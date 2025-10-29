from __future__ import annotations
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import classification_report, confusion_matrix, brier_score_loss, f1_score
from sentence_transformers import SentenceTransformer

from config import load_config
from normalize import normalize_text

def load_dataset(path: Path, positive_label: str) -> tuple[list[str], np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain text and label columns.")
    
    # text column
    x = [normalize_text(t) for t in df["text"].astype(str).tolist()]
    # label column
    raw_label = df["label"]
    # preveri ali je label tako kot mora biti (0 ali 1)
    if raw_label.dtype == object: 
        pos = positive_label.strip().lower()
        y = raw_label.astype(str).str.strip().str.lower().apply(lambda v: 1 if v == pos else 0).astype(int).to_numpy()
    else:
        uniq = set(pd.Series(raw_label).unique())
        if uniq != {0, 1}:
            raise ValueError(f"Numeric labels must be 0/1 only, got: {sorted(uniq)}")
        y = raw_label.astype(int).to_numpy()
    return x, y

def plot_reliability(y_true: np.ndarray, p: np.ndarray, out_png: Path):
    prob_true, prob_pred = calibration_curve(y_true, p, n_bins=10, strategy="uniform")
    plt.figure()
    plt.plot(prob_pred, prob_true, marker="o", label="model")
    plt.plot([0, 1], [0, 1], linestyle="--", label="ideal")
    plt.xlabel("napovedana verjetnost (p)")
    plt.ylabel("dejanska frekvenca")
    plt.title("reliability (kalibracija)")
    plt.grid(True)
    plt.legend()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=160)
    plt.close()

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, p: np.ndarray):
    report = classification_report(y_true, y_pred, output_dict=True, digits=4, target_names=["other", "test"])
    cm = confusion_matrix(y_true, y_pred).tolist()
    brier = float(brier_score_loss(y_true, p))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    return {
        "classification_report": report,
        "confusion_matrix": cm,
        "brier_score": brier,
        "macro_f1": macro_f1
    }

def main():
    ap = argparse.ArgumentParser(description="evaluate calibrated classifier on CSV dataset.")
    ap.add_argument("--data", type=str, required=True, help="CSV with columns: text,label")
    ap.add_argument("--positive_label", type=str, default="test", help="string treated as TEST (1) if labels are strings")
    args = ap.parse_args()

    cfg = load_config()
    paths = cfg.paths

    # naloži model in embedder
    bundle = joblib.load(paths.model_file)
    cal_clf = bundle["cal_clf"]
    embedder = SentenceTransformer(str(paths.embedder_dir))

    # naloži podatke
    x_texts, y = load_dataset(Path(args.data), args.positive_label)

    # embedd + probability
    z = np.asarray(embedder.encode(x_texts, normalize_embeddings=True, batch_size=64, convert_to_numpy=True, show_progress_bar=True))
    p = cal_clf.predict_proba(z)[:, 1]
    y_hat = (p >= 0.5).astype(int)

    # metrike
    metrics = compute_metrics(y, y_hat, p)
    with open(paths.metrics_file, "w", encoding="utf-8") as f: 
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # reliability
    plot_reliability(y, p, paths.reliability_png)

    #izpis
    print("=== eval ===")
    print(json.dumps(metrics["classification_report"], indent=2))
    print(f"confusion: {metrics['confusion_matrix']}")
    print(f"brier: {metrics['brier_score']:.6f} | Macro-F1: {metrics['macro_f1']:.4f}")
    print(f"saved metrics: {paths.metrics_file}")
    print(f"saved reliability plot: {paths.reliability_png}")

if __name__ == "__main__":
    main()