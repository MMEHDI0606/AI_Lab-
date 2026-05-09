# RACE Reading Comprehension & Quiz Generation System

A scikit-learn–based reading comprehension system trained on the RACE dataset. Trains two models to verify answers and generate distractors/hints, then serves them through a Streamlit UI.

Final evaluation and reporting use BLEU, ROUGE, and METEOR on generated text against reference text. Accuracy, Precision, Recall, F1, and related classification metrics may still appear as internal diagnostics for intermediate components, but they are not the final project metrics.

## Setup

```bash
pip install -r requirements.txt
```

Also download NLTK data (run once):
```python
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
```

## Dataset

Download the RACE dataset and place the three splits in `data/raw/`:

```
data/raw/train.csv
data/raw/val.csv
data/raw/test.csv
```

Columns: `id, article, question, A, B, C, D, answer`

## Run notebook (Phase 1 — train models)

```bash
jupyter notebook notebooks/EDA_and_Training.ipynb
# Kernel → Restart & Run All
```

This trains all models and saves artifacts to `models/` and `data/processed/`.

## Run UI

```bash
streamlit run ui/app.py
```

## Run tests

```bash
pytest tests/ -v
```

## Project structure

```
race_rc_project/
├── data/
│   ├── raw/                    # Original RACE CSVs
│   └── processed/              # Feature matrices (X_train.npz etc.) + figures
├── models/
│   ├── model_a/traditional/    # lr_model.pkl, svm_model.pkl, rf_model.pkl
│   └── model_b/traditional/    # distractor_ranker.pkl, hint_scorer.pkl, vectorizer_b.pkl
├── src/
│   ├── preprocessing.py
│   ├── model_a_train.py
│   ├── model_b_train.py
│   ├── inference.py
│   └── evaluate.py
├── ui/
│   └── app.py                  # Streamlit 4-screen application
├── tests/
│   └── test_inference.py       # 6 pytest tests
├── notebooks/
│   └── EDA_and_Training.ipynb
├── report/
│   └── final_report.md
└── requirements.txt
```

## Results summary

### Model A — Answer Verification (test set)

| Model               | BLEU   | ROUGE-1 | ROUGE-L | METEOR |
|---------------------|--------|---------|---------|--------|
| Logistic Regression | 0.3027 | 0.4733  | 0.4664  | 0.4072 |
| SVM (Calibrated)    | 0.3018 | 0.4725  | 0.4655  | 0.4062 |
| Random Forest       | 0.2573 | 0.4268  | 0.4199  | 0.3595 |
| Ensemble (Soft Vote)| 0.2735 | 0.4469  | 0.4401  | 0.3787 |

### Model B — Distractor Generation (val set, 200 samples)

| Task                    | BLEU   | ROUGE-1 | ROUGE-L | METEOR |
|-------------------------|--------|---------|---------|--------|
| Distractor Generation   | 0.0173 | 0.1206  | 0.1065  | 0.0954 |

### Unsupervised / Semi-Supervised (diagnostics only)

| Method                  | Score     |
|-------------------------|-----------|
| K-Means Silhouette      | 0.0947    |
| K-Means Purity          | 0.7469    |
| GMM Silhouette          | 0.0130    |
| GMM Log-Likelihood      | 5208.3940 |
| Label Propagation F1    | 0.4303    |

See [PROJECT_REQUIREMENTS_CHECKLIST.md](c:\Users\hkals\Desktop\UNI-sem5\ai_proj\race_rc_project\PROJECT_REQUIREMENTS_CHECKLIST.md) for a consolidated submission checklist based on the assignment PDF plus the instructor metric update.
