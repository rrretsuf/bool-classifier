from __future__ import annotations
import json
import time
from typing import Tuple, Optional

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

from config import load_config
from normalize import normalize_text

# global load (lazy)
_CFG = None 
_CAL_CLF = None
_EMBEDDER = None

def _load_once():
    global _CFG, _CAL_CLF, _EMBEDDER
    if _CFG is None:
        _CFG = load_config()
    if _CAL_CLF is None: 
        bundle = joblib.load(_CFG.paths.model_file)
        _CAL_CLF = bundle["cal_clf"]
    if _EMBEDDER is None:
        # vedno uporabljemo lokalno kopijo embedderja
        _EMBEDDER = SentenceTransformer(str(_CFG.paths.embedder_dir))
    
def _embed(texts):
    return np.asarray(
        _EMBEDDER.encode(
            texts,
            normalize_embeddings=True,
            batch_size=64, 
            convert_to_numpy=True,
            show_progress_bar=False
        )
    )

def _predict_proba(text: str) -> float:
    z = _embed([text])
    p = float(_CAL_CLF.predict_proba(z)[:, 1][0]) # verjetnost za 1 (test)
    return p

def _call_openrouter(text: str) -> Optional[str]:
    """
    Vrne test ali other na podlagi llm klasifikacije. 
    Če pride do napake vrne None.
    """
    if not _CFG.openrouter_api_key:
        return None
    
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=_CFG.openrouter_api_key,
        default_headers={
            "X-Title": "bool-classifier",
        },
    )

    try:
        completion = client.chat.completions.create(
            extra_body={},
            model=_CFG.openrouter_model,
            messages=[
                    {"role": "system", "content": "You are an intent classifier. Reply ONLY with JSON: {\"intent\":\"test\"|\"other\"}."},
                    {"role": "user", "content": text},
            ]
        )
        content = completion.choices[0].message.content
        parsed = json.loads(content)
        intent = parsed.get("intent")
        if intent in ("test", "other"):
            return intent
        return None
    
    except Exception:
        return None

def _log_borderline(payload: dict):
    p = _CFG.paths.borderline_log
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")

def decide(text: str) -> Tuple[str, float, str]:
    """
    Vrne:
    - intent: test ali other
    - confidence: verjetnost za test
    - source: confident ali model ali fallack
    """
    _load_once()
    normalized_text = normalize_text(text)
    p_test = _predict_proba(normalized_text)
    low, high = _CFG.low_thresh, _CFG.high_thresh

    # confident območja
    if p_test <= low:
        # model je prepričan, da ni test
        return "other", p_test, "confident"
    if p_test >= high:
        # model je prepričan, da je test
        return "test", p_test, "confident"
    
    # mejno območje -> poskus fallbacka (openrouter)
    intent_llm = _call_openrouter(normalized_text)
    if intent_llm is not None:
        _log_borderline({
            "time": time.time(),
            "text": normalized_text,
            "probability of test": round(p_test, 6),
            "source": "fallback",
            "final_intent": intent_llm,
            "model": _CFG.openrouter_model
        })
        return intent_llm, p_test, "fallback"
    
    # če fallback ne dela (vrne se None), se vrnemo na lokalni model in logiramo mejni primer
    intent_local = "test" if p_test >= 0.5 else "other"
    _log_borderline({
            "time": time.time(),
            "text": normalized_text,
            "probability of test": round(p_test, 6),
            "source": "model",
            "final_intent": intent_local,
            "note": "fallback_failed_or_missing_key"
    })
    return intent_local, p_test, "model"