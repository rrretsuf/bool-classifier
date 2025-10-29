from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

from inference import decide


def run_one(text: str):
    intent, p, src = decide(text)
    print(json.dumps({
        "text": text,
        "intent": intent,
        "confidence": p,
        "source": src
    }, ensure_ascii=False))

def run_file(path: Path):
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if not t:
                continue
            run_one(t)

def main():
    ap = argparse.ArgumentParser(description="quick local interface test")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--text", type=str, help="single input text")
    g.add_argument("--file", type=str, help="path to utf-8 text file (one query per line)")
    args = ap.parse_args()

    if args.text is not None:
        run_one(args.text)
    else:
        run_file(Path(args.file))

if __name__ == "__main__":
    main()