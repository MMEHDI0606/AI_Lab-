# RACE Reading Comprehension Project — Agentic Execution Plan

> **How this file works:**
> This plan is split into two phases for a human-in-the-loop workflow.
> - **Phase 1** — the agent produces a complete Jupyter notebook; the human runs it manually and returns the executed notebook.
> - **Phase 2** — the agent receives the executed notebook, reads all outputs/metrics, and completes all remaining deliverables automatically.

---

## Context & Constraints

- **Dataset:** RACE (ReAding Comprehension from Examinations) — ~87,866 train rows, columns: `id, article, question, A, B, C, D, answer`
- **Dataset path on disk:** `data/raw/train.csv`, `data/raw/test.csv`, `data/raw/val.csv`
- **Primary feature representation:** One-Hot Encoding (TF-IDF is optional/secondary)
- **All models:** scikit-learn only — no deep learning, no transformers, no BERT
- **Hardware target:** standard CPU / Google Colab T4 — inference must complete in <10 s per sample
- **UI framework:** Streamlit (recommended)
- **Output root:** `race_rc_project/`
- **NOTE REGARDING DATASET:** use only one file and split the data yourself in 80-10-10 proportion.
- **Updated evaluation policy (mandatory):** Do not use Accuracy/Precision/Recall/F1/Exact Match as final project evaluation metrics.
- **Final evaluation metrics must be:** BLEU, ROUGE, METEOR (for generated questions/answers/distractors/hints vs references).
---

## Project Folder Structure (agent must create this)

```
race_rc_project/
├── data/
│   ├── raw/                    # Original RACE CSVs — agent must not modify
│   └── processed/              # Feature matrices, splits — agent writes here
├── models/
│   ├── model_a/
│   │   └── traditional/        # Pickled sklearn models for Model A
│   └── model_b/
│       └── traditional/        # Pickled sklearn models for Model B
├── src/
│   ├── preprocessing.py        # Dataset loading & feature engineering
│   ├── model_a_train.py        # Training script for Model A
│   ├── model_b_train.py        # Training script for Model B
│   ├── inference.py            # Unified inference API
│   └── evaluate.py             # Metric computation utilities
├── ui/
│   ├── app.py                  # Streamlit entry point
│   └── components/             # Reusable Streamlit widgets
├── notebooks/
│   ├── EDA_and_Training.ipynb  # ← PHASE 1 OUTPUT (agent writes this)
│   └── experiments.ipynb       # Optional experiment log
├── tests/
│   └── test_inference.py       # Unit tests
├── requirements.txt
├── README.md
└── report/
    └── final_report.pdf        # ← PHASE 2 OUTPUT
```

---

---

# PHASE 1 — Notebook Generation

> **Agent task:** Run import kagglehub

# Download latest version
path = kagglehub.dataset_download("ankitdhiman7/race-dataset")

print("Path to dataset files:", path)

Produce `notebooks/EDA_and_Training.ipynb` in full.
> The human will run this notebook cell-by-cell, then return the executed `.ipynb` (with all cell outputs populated) to the agent to begin Phase 2.

---

## Phase 1 Instructions for the Agent

### What to produce
A single self-contained Jupyter notebook: `notebooks/EDA_and_Training.ipynb`

The notebook must be **runnable top-to-bottom with `Kernel → Restart & Run All`** with no manual intervention beyond having the dataset in `data/raw/`.

---

### Notebook Section 1 — Environment & Data Loading

**Cells to include:**

1. Install / import block:
   ```python
   # All imports at the top — pandas, numpy, sklearn, gensim, joblib, matplotlib, seaborn
   ```

2. Load all three splits from `data/raw/`:
   ```python
   train_df = pd.read_csv('data/raw/train.csv')
   val_df   = pd.read_csv('data/raw/val.csv')
   test_df  = pd.read_csv('data/raw/test.csv')
   print(train_df.shape, val_df.shape, test_df.shape)
   print(train_df.columns.tolist())
   print(train_df.head(2))
   ```

3. Sanity check — assert expected columns exist, no nulls in critical columns.

---

### Notebook Section 2 — Exploratory Data Analysis

**Agent must produce all of the following plots with titles and axis labels. Save each figure to `data/processed/figures/`.**

| Plot | What to show | Chart type |
|------|-------------|------------|
| Answer label distribution | Count of A/B/C/D in train | Bar chart |
| Article length distribution | Word count histogram | Histogram (bins=50) |
| Question length distribution | Word count histogram | Histogram |
| Option length distribution | Avg word count per option vs correct | Grouped bar |
| Question type breakdown | First-word classification (Who/What/Where/When/Why/How/Fill-in) | Horizontal bar |
| Middle vs high school split | Row count by difficulty level | Pie chart |
| Answer label by split | Train vs val vs test balance | Grouped bar |

**Summary statistics cell** — print a markdown table with: mean/median/max passage length, vocab size (unique tokens in train articles), total questions, answer balance %.

---

### Notebook Section 3 — Preprocessing Pipeline

**Agent writes the preprocessing logic inline in the notebook first, then extracts it to `src/preprocessing.py` as a final cell using `%%writefile`.**

Steps to implement in order:

1. **Text cleaning function**
   - Lowercase, remove punctuation, strip extra whitespace
   - Apply to: article, question, all four options

2. **Data expansion** — convert each row (1 question, 4 options) into 4 rows (1 per option), adding:
   - `label` column: 1 if this option == answer, else 0
   - `option_letter` column: A/B/C/D
   - `combined_text` column: `article + " [SEP] " + question + " [SEP] " + option`

3. **One-Hot Encoding**
   - Build vocabulary over `combined_text` in train set (top 5000 tokens by frequency)
   - Fit `sklearn.preprocessing.OneHotEncoder` or `CountVectorizer(binary=True, max_features=5000)`
   - Transform train, val, test — save as sparse `.npz` to `data/processed/`

4. **Cosine similarity feature**
   - For each row compute cosine similarity between article tokens and option tokens (use CountVectorizer, not TF-IDF)
   - Append as a single float column to feature matrix

5. **Handcrafted lexical features** (combine into one array per row):
   - `option_length`: word count of option
   - `question_length`: word count of question
   - `keyword_overlap`: count of tokens shared between question and option
   - `option_in_article`: count of option tokens that appear in article
   - `answer_position`: character position of first option token in article (0 if not found), normalized by article length

6. **Final feature matrix** — hstack: One-Hot sparse + cosine similarity + lexical features
   - Save: `data/processed/X_train.npz`, `X_val.npz`, `X_test.npz`
   - Save labels: `data/processed/y_train.npy`, `y_val.npy`, `y_test.npy`

7. **`%%writefile src/preprocessing.py`** — write the full pipeline as importable functions: `clean_text()`, `expand_df()`, `build_features()`, `load_features()`

---

### Notebook Section 4 — Model A: Traditional ML (Answer Verification)

**Train and evaluate ALL of the following. Each model block must follow this pattern:**
```
[markdown cell: model name + brief description]
[code cell: train]
[code cell: internal diagnostic check (optional; not used for final grading)]
[code cell: save model with joblib]
```

#### Models to train:

**4a. Logistic Regression**
- Features: One-Hot + cosine + lexical
- `sklearn.linear_model.LogisticRegression(max_iter=1000, C=1.0)`
- Report: training diagnostics only (no Accuracy/Precision/Recall/F1/Exact Match in final evaluation tables)
- Save: `models/model_a/traditional/lr_model.pkl`

**4b. Support Vector Machine**
- Features: One-Hot + cosine + lexical
- `sklearn.svm.LinearSVC(max_iter=2000)`
- Wrap with `CalibratedClassifierCV` for probability outputs
- Report: diagnostics only (optional)
- Save: `models/model_a/traditional/svm_model.pkl`

**4c. Naive Bayes**
- Features: CountVectorizer on question text only (question type classification)
- `sklearn.naive_bayes.MultinomialNB()`
- Target: first word of question (Who/What/Where/When/Why/How/Other)
- Report: diagnostics only (optional)
- Save: `models/model_a/traditional/nb_model.pkl`

**4d. Random Forest**
- Features: lexical features only (no One-Hot — too sparse)
- `sklearn.ensemble.RandomForestClassifier(n_estimators=200, random_state=42)`
- Report: diagnostics only (optional)
- Save: `models/model_a/traditional/rf_model.pkl`

**4e. Comparison table cell** — print a formatted table of generation metrics:
```
Model               | BLEU | ROUGE-L | METEOR | Train time (s)
Logistic Regression | ...  | ...     | ...    | ...
SVM                 | ...  | ...     | ...    | ...
Naive Bayes         | ...  | ...     | ...    | ...
Random Forest       | ...  | ...     | ...    | ...
```

---

### Notebook Section 5 — Model A: Unsupervised & Semi-Supervised

**5a. K-Means Clustering**
- Input: One-Hot feature matrix (train set, unlabeled)
- Elbow method: fit K-Means for K = 2..10, plot inertia
- Fit final K-Means with best K
- Report: clustering diagnostics only (not part of final grading metrics)
- Visualize clusters with PCA → 2D scatter plot (color by cluster)

**5b. Gaussian Mixture Model**
- `sklearn.mixture.GaussianMixture(n_components=best_K)`
- Report: diagnostics only (optional)
- Compare soft vs hard assignments in a small table

**5c. Label Propagation (Semi-Supervised)**
- Take 10% of train as labeled, treat rest as unlabeled (label = -1)
- `sklearn.semi_supervised.LabelPropagation(kernel='knn', n_neighbors=7)`
- Report: diagnostics only (optional)
- Comparison table: optional qualitative comparison only

**5d. Ensemble**
- Soft voting: average `predict_proba` from LR + SVM (calibrated) + RF
- `sklearn.ensemble.VotingClassifier` or manual averaging
- Report: BLEU, ROUGE-L, METEOR on generated answer strings vs reference answers
- Save ensemble predictions to `data/processed/ensemble_val_preds.npy`

---

### Notebook Section 6 — Model A: Template-Based Question Generation

**6a. Candidate sentence extraction**
- For each article: split into sentences (simple `.split('.')`)
- Score each sentence by One-Hot keyword overlap with the correct answer string
- Select top-3 candidate sentences

**6b. Wh-word template application**
- For each candidate sentence, attempt each template:
  - Strip subject → "What did [subject] do?" pattern
  - Replace named entity-like tokens (capitalized words) → "Who is ___?"
  - Fill-in-the-blank: blank out the answer token → "The ___ is important because..."
- Produce a list of candidate questions per article

**6c. Question ranker**
- Features per candidate question: length, starts with Wh-word (binary), overlap with article, overlap with answer
- Train `LinearSVC` on a small labeled subset (use existing RACE questions as positive examples, garbled sentences as negatives)
- Select top-1 ranked question per article
- Print 5 example (article snippet, generated question, gold question) triples
- Evaluate generated questions using BLEU, ROUGE-L, METEOR against gold RACE questions

---

### Notebook Section 7 — Model B: Distractor Generation

**7a. Candidate extraction**
- For each (article, correct_answer): retrieve all phrases from article using sliding window (1–3 grams)
- Filter: remove exact matches to correct answer

**7b. Feature engineering per candidate**
- `cosine_sim_to_answer`: One-Hot cosine similarity between candidate and correct answer
- `char_match_score`: character n-gram overlap ratio
- `passage_frequency`: raw count of candidate in article
- `length_ratio`: len(candidate) / len(correct_answer)

**7c. Distractor ranker**
- Label candidates: 1 if this phrase is one of the 3 original distractors, else 0
- Train `LogisticRegression` on the above features
- At inference: select top-3 non-answer candidates (with diversity penalty: cosine sim between selected distractors < 0.8)
- Evaluate generated distractors with BLEU, ROUGE-L, METEOR against reference distractors (B/C/D options excluding gold)

**7d. Frequency-based fallback**
- Identify high-frequency nouns in article (top-20 by count)
- Substitute correct answer with similarly frequent but different terms
- Use as backup when ranker confidence is low

**7e. Extractive hint generator**
- Score each article sentence by keyword overlap with question (bag-of-words, no neural net)
- Rank sentences descending by score
- Hint 1 = lowest-scoring sentence (general), Hint 2 = mid, Hint 3 = highest overlap (most explicit)
- Train `LogisticRegression` on (keyword_overlap, position_in_article, sentence_length) to predict if sentence is the gold hint sentence
- Report: BLEU, ROUGE-L, METEOR for generated hints vs reference hint sentences

**7f. Save Model B artifacts**
- `models/model_b/traditional/distractor_ranker.pkl`
- `models/model_b/traditional/hint_scorer.pkl`
- `models/model_b/traditional/vectorizer_b.pkl`

---

### Notebook Section 8 — Final Evaluation Summary

**Agent must produce this cell as the last executable cell:**

```python
# ── FINAL RESULTS TABLE ──────────────────────────────────────────
# Model A
print("=== MODEL A — QUESTION/ANSWER GENERATION QUALITY ===")
# print BLEU, ROUGE-L, METEOR for: LR, SVM, NB, RF, Ensemble
# on BOTH val and test sets (where references are available)

# Model B
print("=== MODEL B — DISTRACTOR GENERATION ===")
# print BLEU, ROUGE-L, METEOR on test set

print("=== MODEL B — HINT EXTRACTION ===")
# print BLEU, ROUGE-L, METEOR on test set

# Unsupervised
print("=== UNSUPERVISED / SEMI-SUPERVISED ===")
# print optional diagnostics (not used for final grading)
```

**All metric values printed by this cell are what Phase 2 will read.**

---

### Notebook Section 9 — Export src/ scripts

Using `%%writefile`, export the following from the notebook:

- `src/preprocessing.py` — all preprocessing functions
- `src/model_a_train.py` — training logic for all Model A classifiers (callable as `python src/model_a_train.py`)
- `src/model_b_train.py` — training logic for Model B
- `src/evaluate.py` — `compute_generation_metrics(pred_texts, ref_texts)` returning BLEU/ROUGE/METEOR
- `src/inference.py` — `predict_answer(article, question, options)` and `generate_distractors(article, question, answer)` and `get_hints(article, question)` — loads pickled models

---

### Phase 1 Completion Criteria

The agent must verify all of the following before declaring Phase 1 done:

- [ ] Notebook runs top-to-bottom without errors
- [ ] All 7 EDA plots saved to `data/processed/figures/`
- [ ] Feature matrices exist: `X_train.npz`, `X_val.npz`, `X_test.npz`
- [ ] All 4 Model A classifiers saved as `.pkl`
- [ ] Model B distractor ranker + hint scorer saved as `.pkl`
- [ ] Final Results Table (Section 8) prints cleanly
- [ ] All `src/` scripts written via `%%writefile`

**Hand off:** Save the notebook as `notebooks/EDA_and_Training.ipynb`. Notify the human that Phase 1 is complete and the notebook is ready to run.

---

---

# PHASE 2 — Post-Notebook Deliverables

> **Trigger:** The human returns the executed `notebooks/EDA_and_Training.ipynb` with all cell outputs populated.
> The agent reads the notebook outputs, extracts all metric values, and completes the following deliverables autonomously.

---

## Phase 2 Step 1 — Read Notebook Outputs

Before doing anything else, the agent must:

1. Parse `notebooks/EDA_and_Training.ipynb` — extract all printed outputs from cell outputs
2. Extract the following values from the Final Results Table (Section 8 output):
   - Model A: BLEU, ROUGE-L, METEOR for each model (val + test)
   - Model B distractors: BLEU, ROUGE-L, METEOR (test)
   - Model B hints: BLEU, ROUGE-L, METEOR (test)
   - Optional diagnostics: unsupervised/semi-supervised stats (not grading metrics)
3. Extract figure paths from `data/processed/figures/`
4. Store all values in a structured dict — these populate the report and dashboard

---

## Phase 2 Step 2 — Streamlit UI

**File:** `ui/app.py`

Build a 4-screen Streamlit application. Wire it to `src/inference.py` — all model calls go through `predict_answer()`, `generate_distractors()`, `get_hints()`.

### Screen 1 — Article Input
```
st.title("Reading Comprehension Quiz Generator")
[text_area: "Paste a reading passage"]
[button: "Load random RACE sample"]  ← loads from test_df, displays article
[button: "Generate Quiz"]            ← calls Model A (question gen) + Model B (distractors + hints)
[spinner: "Generating quiz..."]      ← shown during inference
```

### Screen 2 — Quiz View
```
st.subheader("Question")
[display: generated or RACE question text]
[radio: options A / B / C / D]       ← one correct answer, three Model B distractors
[button: "Check Answer"]             ← calls Model A verifier
[result: green "Correct!" or red "Incorrect — correct answer was X"]
[expander: "Why?"]                   ← show cosine similarity score or keyword overlap explanation
[notice: "⚠️ AI-generated — errors possible"]
```

### Screen 3 — Hint Panel
```
[expander: "Hint 1 — General clue"]    ← st.expander, collapsed by default
[expander: "Hint 2 — More specific"]
[expander: "Hint 3 — Near-explicit"]
[button: "Reveal Answer"]              ← only shown after all 3 expanders have been opened (use st.session_state)
```

### Screen 4 — Analytics Dashboard
```
st.subheader("Model A Performance")
[table: BLEU | ROUGE-L | METEOR for each model]
[chart: model-wise BLEU/ROUGE/METEOR comparison]

st.subheader("Model B Performance")
[table: BLEU | ROUGE-L | METEOR for distractor and hint generation]

st.subheader("Session Stats")
[metric: total questions answered this session]
[metric: avg BLEU for generated outputs in session]
[metric: avg inference time (ms)]
[button: "Export session log to CSV"]
```

**UX requirements the agent must enforce:**
- All error states (empty input, model load failure) display `st.error("...")` with a friendly message
- `st.spinner` wraps every model call
- Minimum font size: Streamlit default (do not shrink)
- Session state initialized in `if 'log' not in st.session_state` guard

---

## Phase 2 Step 3 — Unit Tests

**File:** `tests/test_inference.py`

Write pytest tests covering:

```python
def test_predict_answer_returns_valid_label():
    # load a single RACE row from test set
    # call predict_answer(article, question, [A,B,C,D])
    # assert result in ['A','B','C','D']

def test_generate_distractors_returns_three():
    # call generate_distractors(article, question, correct_answer)
    # assert len(result) == 3
    # assert correct_answer not in result

def test_get_hints_returns_three():
    # call get_hints(article, question)
    # assert len(result) == 3
    # assert all(isinstance(h, str) for h in result)

def test_no_distractor_equals_correct_answer():
    # for 20 random test samples, assert distractors != correct answer

def test_inference_latency():
    # time a single call to predict_answer + generate_distractors + get_hints
    # assert total < 10 seconds

def test_generation_metrics_return_valid_ranges():
   # call compute_generation_metrics(pred_texts, ref_texts)
   # assert keys {'bleu','rouge_l','meteor'} exist
   # assert each score is in [0.0, 1.0]
```

---

## Phase 2 Step 4 — README

**File:** `README.md`

Structure:

```markdown
# RACE Reading Comprehension & Quiz Generation System

## Setup
pip install -r requirements.txt

## Dataset
Place train.csv / val.csv / test.csv in data/raw/

## Run notebook (Phase 1)
jupyter notebook notebooks/EDA_and_Training.ipynb
# Run all cells — this trains all models and saves artifacts

## Run UI
streamlit run ui/app.py

## Run tests
pytest tests/

## Project structure
[paste folder tree]

## Results summary
[table: BLEU | ROUGE-L | METEOR for Model A and Model B — populated from notebook outputs]
```

---

## Phase 2 Step 5 — Final Report

**File:** `report/final_report.pdf`
Generate as a structured markdown document then convert to PDF.

Sections (use the exact structure from the spec):

1. **Abstract** — ≤200 words. Summarise: dataset, two models, UI, key metric results (fill from notebook outputs)
2. **Introduction & Motivation** — edtech context, RACE dataset significance, what this system does
3. **Related Work** — cite all 8 papers from the spec's suggested references section; one paragraph per paper cluster
4. **Dataset Analysis** — RACE statistics table, embed the EDA figures from `data/processed/figures/`, discuss class balance and linguistic diversity
5. **Model A: Design, Training, Results**
   - Architecture diagram (text-based, using markdown table or ASCII)
   - One-Hot encoding rationale
   - Results table (all 4 classifiers + ensemble, val + test) using BLEU/ROUGE-L/METEOR
   - Unsupervised / semi-supervised results table
   - Generation quality comparison figure (BLEU/ROUGE/METEOR)
   - Discussion: which model performed best and why
6. **Model B: Design, Training, Results**
   - Distractor pipeline description with example (article snippet → 3 distractors)
   - Hint extractor description with example
   - Results: BLEU, ROUGE-L, METEOR
   - Discussion: diversity penalty impact
7. **User Interface Description** — describe all 4 screens, include Streamlit component list
8. **Evaluation & Discussion** — cross-model comparison, latency results, human evaluation summary
9. **Limitations & Future Work**
   - RACE cultural/linguistic bias (Chinese school exam origin)
   - No deployment in real exams without human review
   - One-Hot encoding limitations vs contextual embeddings
   - Future: fine-tune contextual models and add semantic metrics (e.g., BERTScore) alongside BLEU/ROUGE/METEOR
10. **Conclusion**
11. **References** — all 8 cited papers in IEEE format

---

## Phase 2 Step 6 — requirements.txt

Pin exact versions. Must include at minimum:

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
gensim>=4.3
xgboost>=2.0
sentence-transformers>=2.2
streamlit>=1.32
joblib>=1.3
matplotlib>=3.7
seaborn>=0.13
pytest>=8.0
nltk>=3.8
rouge-score>=0.1.2
```

---

## Phase 2 Completion Criteria

The agent must verify all of the following before declaring Phase 2 done:

- [ ] `ui/app.py` runs with `streamlit run ui/app.py` without import errors
- [ ] All 4 screens reachable and wired to real model inference
- [ ] `pytest tests/` passes all 6 tests
- [ ] `README.md` contains accurate BLEU/ROUGE/METEOR values from notebook outputs
- [ ] `report/final_report.pdf` generated with all sections complete and figures embedded
- [ ] `requirements.txt` present with pinned versions
- [ ] All files committed to git with meaningful commit messages

---

## Ethical Checklist (agent must address in report)

- [ ] Cultural/linguistic bias: RACE passages from Chinese school exams — discuss generalization limits
- [ ] Accessibility: Streamlit UI uses sufficient contrast and is keyboard navigable
- [ ] Academic integrity notice: UI displays "AI-generated — errors possible" on quiz screen
- [ ] Model transparency: UI labels which answers are AI-generated
- [ ] No deployment in real exam settings without human review — stated in report limitations

---

*End of plan. Phase 1 produces the notebook. Phase 2 begins when the executed notebook is returned.*
