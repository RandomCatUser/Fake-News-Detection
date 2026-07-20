# Fake News Detection

Fake news spreads fast, and telling it apart from real reporting by eye is getting
harder. This project uses **supervised machine learning** to classify news articles
as **REAL** or **FAKE** based on their text content.

It follows the classic approach popularized by the DataFlair
["Detecting Fake News"](https://data-flair.training/blogs/advanced-python-project-detecting-fake-news/)
tutorial: TF-IDF vectorization combined with a `PassiveAggressiveClassifier`.

## How it works

1. **TF-IDF Vectorization** — converts article text into numeric features weighted
   by how important each word is to a document relative to the whole corpus.
2. **PassiveAggressiveClassifier** — an online-learning linear classifier that is
   fast to train and performs well on high-dimensional sparse text data.
3. **Evaluation** — accuracy score, confusion matrix, and a classification report
   (precision/recall/F1) on a held-out test set.

## Project structure

```
Fake News Detection/
├── README.md
├── requirements.txt
├── data/
│   └── news.csv              # sample dataset (see note below)
├── model/
│   ├── train.py              # standalone training script
│   ├── vectorizer.pkl         # saved TF-IDF vectorizer (after training)
│   └── classifier.pkl         # saved classifier (after training)
└── docs/
    └── Fake_News_Detection.ipynb   # step-by-step Jupyter notebook walkthrough
```

## About the dataset

`data/news.csv` currently contains a **small, synthetic sample** (40 rows) so the
whole pipeline can be run immediately and end-to-end as a demo. It was generated
for this project and is **not** a real news dataset.

The dataset link originally referenced for this project points to a Google Drive
file (`news.zip`) that requires an authenticated browser session to download, so it
could not be fetched automatically here. For real results, swap in a full-size
dataset such as the ~6,300-article Kaggle **"Fake News detection"** dataset used in
the original DataFlair tutorial (or any similar dataset), formatted as a CSV with
at least these columns:

| column | description                  |
|--------|-------------------------------|
| title  | headline of the article        |
| text   | full body text of the article  |
| label  | `REAL` or `FAKE`                |

Just drop your CSV in as `data/news.csv` (same column names) and everything else
works unchanged — or pass a different path/column names via command-line flags
(see below).

## Setup

```bash
cd "Fake News Detection"
pip install -r requirements.txt
```

## Usage

### Option 1: Run the training script

```bash
cd model
python train.py --data ../data/news.csv
```

Optional flags:
- `--text-col` — name of the text column (default: `text`)
- `--label-col` — name of the label column (default: `label`)
- `--test-size` — fraction of data held out for testing (default: `0.2`)
- `--out-dir` — where to save `vectorizer.pkl` / `classifier.pkl` (default: current dir)

This prints accuracy, a confusion matrix, and a classification report, then saves
the trained vectorizer and classifier as `.pkl` files.

### Option 2: Explore the Jupyter notebook

```bash
jupyter notebook docs/Fake_News_Detection.ipynb
```

The notebook walks through every step interactively — loading data, TF-IDF
vectorization, training, evaluation (with a confusion matrix heatmap), and testing
the model on a custom headline — with explanations along the way.

### Making a prediction with a saved model

```python
import pickle

with open("model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
with open("model/classifier.pkl", "rb") as f:
    classifier = pickle.load(f)

text = "Your article text goes here..."
prediction = classifier.predict(vectorizer.transform([text]))
print(prediction[0])  # 'REAL' or 'FAKE'
```

## Notes & limitations

- Accuracy numbers from the bundled sample dataset are **not meaningful** — the
  sample is tiny and only exists to prove the pipeline runs. Retrain on a real,
  larger dataset before drawing any conclusions.
- This is a **text-style classifier**, not a fact-checker: it learns patterns in
  wording, tone, and structure that correlate with the labels in its training
  data. It cannot verify real-world facts, and its accuracy is only as good as the
  dataset and labels it's trained on.
- You may see a `FutureWarning` about `PassiveAggressiveClassifier` being
  deprecated in newer versions of scikit-learn (in favor of `SGDClassifier` with
  specific parameters) — this is harmless for current versions and doesn't affect
  results.

## Possible extensions

- Compare against other models: Logistic Regression, Multinomial Naive Bayes, SVM.
- Add cross-validation and hyperparameter tuning (`GridSearchCV`).
- Combine `title` and `text` as input, or engineer additional features
  (article length, punctuation patterns, source metadata).
- Add explainability — inspect which TF-IDF terms most influence each prediction.
- Build a simple web UI (e.g. Flask/Streamlit) around the saved model for live
  predictions.

## Issues

- If you found any errors, please make an issue report; really appreciate it 🤭
