import unicodedata

def normalize_text(text: str) -> str:
    """
    Normalizira sporočilo od uporabnika za plast modela in rulesov. 
    Normalize pomeni, da preoblikuje v tako obliko, da bo delovalo na teh plasteh.
    """
    if not isinstance(text, str):
        text = str(text)
    # unicode normalizacija
    text = unicodedata.normalize("NFKC", text)
    # trim odvečen whitespace
    text = " ".join(text.split())
    # lowercasing
    return text.lower()