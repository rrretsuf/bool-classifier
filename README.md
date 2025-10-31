# Boolean Classifier

Generic boolean classifier for binary intent classification using sentence embeddings and calibrated logistic regression.

**Current use case**: Distinguishes "test" vs "other" intent, but the architecture is general-purpose and can be adapted for any binary classification task (e.g., "spam" vs "ham", "urgent" vs "normal", etc.).

## How It Works

**Architecture**: Text → Normalize → Embed → Logistic Regression → Calibrated Probabilities

**Tech Stack**:
- **Embeddings**: Sentence Transformers (`BAAI/bge-m3` by default) converts text to dense vectors
- **Classifier**: Scikit-learn Logistic Regression with isotonic calibration for probability calibration
- **Features**: Balanced class weights, normalized embeddings, reliability plots

**Training Flow**: CSV data → text normalization → sentence embeddings → calibrated logistic regression → threshold-based classification (low/high confidence zones)

**Note**: While the current implementation is configured for "test" vs "other" classification, the same architecture works for any binary classification task. See "Adapting for Other Use Cases" below for details.

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
- `OPENROUTER_MODEL` (default: `x-ai/grok-4-fast`)

## Adapting for Other Use Cases

To use this classifier for a different binary classification task (e.g., "spam" vs "ham", "urgent" vs "normal") instead of "test" vs "other", you need to make the following changes:

### Required Code Changes

1. **`classifier/train.py`**:
   - Line 103: Change `target_names=["other", "test"]` to your labels (e.g., `["ham", "spam"]`)
   - Line 192: Change `{"0": "other", "1": "test"}` to your label mapping (e.g., `{"0": "ham", "1": "spam"}`)
   - Line 124: Default `--positive_label` can remain or change to your positive class

2. **`classifier/evaluate.py`**:
   - Line 54: Change `target_names=["other", "test"]` to your labels

3. **`classifier/inference.py`** (most critical):
   - Line 67: Update system prompt from `"test"|\"other\"` to your labels
   - Line 74: Change `if intent in ("test", "other")` to check your labels
   - Line 100-105: Change return values `"other"` and `"test"` to your labels
   - Line 121: Change `"test" if p_test >= 0.5 else "other"` to your labels
   - **Important**: Consider reading labels from `labels.json` (stored during training) instead of hardcoding, so the inference code automatically uses the correct labels from your trained model

4. **Training Command**:
   - Update `--positive_label` argument to match your positive class label in your CSV data

### Example: Spam vs Ham

If adapting for spam detection:
- CSV labels: `"spam"` and `"ham"`
- Train with: `--positive_label spam`
- Update hardcoded strings from `"test"/"other"` to `"spam"/"ham"` in the files above

The core architecture (embeddings → logistic regression → calibration) remains unchanged and works for any binary classification task.