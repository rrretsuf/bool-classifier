# Test Classifier

Boolean classifier for "test" vs "other" intent using sentence embeddings and calibrated logistic regression.

## How It Works

**Architecture**: Text → Normalize → Embed → Logistic Regression → Calibrated Probabilities

**Tech Stack**:
- **Embeddings**: Sentence Transformers (`BAAI/bge-m3` by default) converts text to dense vectors
- **Classifier**: Scikit-learn Logistic Regression with isotonic calibration for probability calibration
- **Features**: Balanced class weights, normalized embeddings, reliability plots

**Training Flow**: CSV data → text normalization → sentence embeddings → calibrated logistic regression → threshold-based classification (low/high confidence zones)

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Training

```bash
cd classifier
python train.py --data data/train.csv --positive_label test
```

Model artifacts saved to `classifier/artifacts/`.

## Evaluation

```bash
python evaluate.py --data data/train.csv --positive_label test
```

## Inference

```bash
python test_infer.py --file data/golder.txt
# or
python test_infer.py --text "naredi test za https://example.com"
```

## Configuration

Optional `.env` variables:
- `EMBEDDER_NAME` (default: `BAAI/bge-m3`)
- `LOW_THRESH` (default: `0.35`)
- `HIGH_THRESH` (default: `0.65`)
- `OPENROUTER_API_KEY` (optional, for fallback)
- `OPENROUTER_MODEL` (default: `x-ai/grok-4-fast`)# classifier-bool