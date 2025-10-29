from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Tuple, List, Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import classification_report, confusion_matrix, brier_score_loss, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from config import load_config
from embeddings import Embedder
from normalize import normalize_text


# helpers ----

def load_dataset(path: Path, positive_label: str) -> Tuple[List[str], np.ndarray]:
    """
    csv s stoplcama text & label, ki vrne x (seznam stringov - text) in y (numpy int64 0/1, kjer 1 pomeni TEST)
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    
    df = pd.read_csv(path)
    
    # nujno mora imeti stolpca "text" in "label"
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain text and label columns.")
    
    # x so vhodna besedila (kot seznam stringov)
    x = [normalize_text(t) for t in df["text"].astype(str).tolist()]
    # raw_label so originalne vrednosti label stolpca
    raw_label = df["label"]

    # če so labeli dejansko stringi
    if raw_label.dtype == object:
        pos = str(positive_label).strip().lower()
        # vse label-e spremeni v lower in primerja s pozitivno oznako
        y = raw_label.astype(str).str.strip().str.lower().apply(lambda v: 1 if v == pos else 0).astype(int).to_numpy()
    else: 
        # če že so številske vrednosti
        uniq = sorted(pd.Series(raw_label).unique())
        # mora biti samo 0 in 1
        if set(uniq) != {0,1}:
               raise ValueError(f"Numeric labels must be 0 or 1 only, got: {uniq}")
        y = raw_label.astype(int).to_numpy()

    # morata biti prisotni oba primera v training data
    if not ({0,1} <= set(np.unique(y))):
        raise ValueError("Dataset must contain examples of both classes (0 and 1)")

    return x, y

def plot_reliability(y_true: np.ndarray, p: np.ndarray, out_png: Path): 
    """
    Nariše reliability diagram (png), ki pokaže ali je model confidence realen ali ne.
    Prav tako imenovan "calibration curve" - kaže ali modelove napovedane verjetnosti (confidence) odražajo resnično pogostost.
    """
    # izračuna true in predicted pogostosti za vsak bin napovedane verjetnosti (n_bins=10)
    prob_true, prob_pred = calibration_curve(y_true, p, n_bins=10, strategy="uniform")
    
    # začne nov plot
    plt.figure()
    
    # nariše modelovo krivuljo z krogi na vsaki točki in doda idealno kalibracijo
    plt.plot(prob_pred, prob_true, marker="o", label="model")
    plt.plot([0, 1], [0, 1], linestyle="--", label="ideal")
    
    #poimenovanje (title, x & y os)
    plt.xlabel("napovedana verjetnost (p)")
    plt.ylabel("dejanska frekvenca")
    plt.title("reliability (kalibracija)")
    plt.grid(True)
    
    # pokaže legendo (model vs idealno)
    plt.legend()
    
    # naredi dir za sliko
    out_png.parent.mkdir(parents=True, exist_ok=True)
    
    # shranimo plot kot png (ločljivost 160 dpi) in končamo
    plt.savefig(out_png, dpi=160)
    plt.close()

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, p: np.ndarray) -> Dict[str, Any]:
    """
    Izračuba glavne številske metrike uspešnosti modela.
    - confusion matrix: pokaže koliko primerov je zadel / zgrešil
    - brier score: meri ali so njegove vrednosti (confidence) realne
    - f1 = kako dobro loči test od other
    """
    # classification report
    report = classification_report(
        y_true, y_pred,
        output_dict=True,
        digits=4,
        target_names=["other", "test"]
    )
    # izračuna confusion matrix
    cm = confusion_matrix(y_true, y_pred).tolist()
    # izračuna brier score 
    brier = float(brier_score_loss(y_true, p))
    # izračuna povprečen f1 
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    return {
        "classification_report": report,
        "confusion_matrix": cm,
        "brier_score": brier,
        "macro_f1": macro_f1
    }


# main train ----

def main():
    parser = argparse.ArgumentParser(description="train calibrated logistics classifier over sentance embeddings (test vs other).")
    parser.add_argument("--data", type=str, required=True, help="path to CSV with colums: text,label.")
    parser.add_argument("--positive_label", type=str, default="test", help="string label treated as TEST (1) if labels are stings.")
    parser.add_argument("--test_size", type=float, default=0.2, help="test split size (default 0.2).")
    parser.add_argument("--seed", type=int, default=42, help="random seed.")
    parser.add_argument("--max_iter", type=int, default=1000, help="LogReg max_iter")
    parser.add_argument("--calibration", choices=["isotonic", "sigmoid"], default="isotonic", help="calibrator: isotonic or platt (sigmoid).")
    args = parser.parse_args()

    cfg = load_config()
    paths = cfg.paths

    # 1) data
    x, y = load_dataset(Path(args.data), positive_label=args.positive_label)
    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=args.test_size, stratify=y, random_state=args.seed
    )

    # 2) embeddings
    emb = Embedder(cfg.embedder_name)
    z_tr = emb.encode(x_tr)
    z_te = emb.encode(x_te)
    
    # 3) model + kalibracija
    base = LogisticRegression(
        max_iter=args.max_iter,
        class_weight="balanced",
        n_jobs=1
    )

    # število primerov v najmanjšem razredu (za določitev možnega števila CV folds za kalibracijo)
    class_counts = np.bincount(y_tr)
    min_class_count = int(class_counts.min())

    if min_class_count < 2:
        # premalo primerov v kakšnem razredu za kalibracijo - trenira nenakalibriran model
        print(
            f"[warn] Too few samples in a class for calibration (min_class_count={min_class_count}). "
            "Training without probability calibration."
        )
        clf = base.fit(z_tr, y_tr)
    else:
        n_splits = min(5, min_class_count)
        if n_splits < 2:
            # premalo primerov za vsaj 2-slojni CV - trenira nenakalibriran model
            print(
                f"[warn] Not enough samples for calibration CV (n_splits={n_splits}). "
                "Training without calibration."
            )
            clf = base.fit(z_tr, y_tr)
        else:
            # trenira kalibriran model
            clf = CalibratedClassifierCV(
                base,
                method=args.calibration,
                cv=n_splits
            )
            clf.fit(z_tr, y_tr)

    # 4) eval
    p_te = clf.predict_proba(z_te)[:, 1] # verjetnost za 1 (test)
    y_hat = (p_te >= 0.5).astype(int)

    metrics = compute_metrics(y_te, y_hat, p_te) 
    plot_reliability(y_te, p_te, paths.reliability_png)

    # 5) shrani artifakte
    joblib.dump({"cal_clf": clf}, paths.model_file)
    emb.save(paths.embedder_dir)
    with open(paths.labels_file, "w", encoding="utf-8") as f: 
        json.dump({"0": "other", "1": "test"}, f, ensure_ascii=False, indent=2)
    with open(paths.metrics_file, "w", encoding="utf-8") as f: 
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    
    # 6) konzola
    print("=== eval ===")
    print(json.dumps(metrics["classification_report"], indent=2))
    print(f"confusion: {metrics['confusion_matrix']}")
    print(f"brier: {metrics['brier_score']:.6f} | Macro-F1: {metrics['macro_f1']:.4f}")
    print(f"saved model: {paths.model_file}")
    print(f"saved embedder: {paths.embedder_dir}")
    print(f"saved labels: {paths.labels_file}")
    print(f"saved metrics: {paths.metrics_file}")
    print(f"saved reliability plot: {paths.reliability_png}")

if __name__ == "__main__":
    main()