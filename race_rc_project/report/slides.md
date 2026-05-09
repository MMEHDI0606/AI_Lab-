# RACE Reading Comprehension & Quiz Generation System
## Demo Presentation Slides

**Mehdi (23i0048) • Fahad (23i0614)**  
AI Lab — Semester 5 | Framework: scikit-learn (CPU-only)

---

## Slide 1 — Title

# RACE Reading Comprehension  
# & Quiz Generation System

**Team:** Mehdi (23i0048) • Fahad (23i0614)  
**Course:** AI Lab — Semester 5  
**Stack:** Python · scikit-learn · Streamlit · NLTK · ROUGE

> *Can classical ML models pass a Chinese high-school English exam?*

---

## Slide 2 — Problem Statement

### What We Built

Three tasks in one pipeline:

| # | Task | Input → Output |
|---|------|---------------|
| 1 | **Answer Verification** | Passage + Question + 4 Options → Correct Option |
| 2 | **Distractor Generation** | Passage + Correct Answer → 3 Plausible Wrong Options |
| 3 | **Hint Extraction** | Passage + Question → 3 Graduated Hint Sentences |

**Constraint:** scikit-learn only — no deep learning, CPU inference < 10 s/sample

---

## Slide 3 — Dataset: RACE

### What is RACE?

- **Full name:** ReAding Comprehension from Examinations  
- **Source:** Lai et al., EMNLP 2017  
- **Origin:** Chinese middle & high school English exams  
- **Why hard:** Multi-sentence reasoning, cultural context, fill-in-the-blank heavy

### Split Statistics

| Split | Questions | Unique Articles | Avg Article | Avg Question |
|-------|-----------|-----------------|-------------|--------------|
| Train | ~70,000   | ~27,000         | ~300 words  | ~11 words    |
| Val   | ~4,900    | ~1,900          | ~295 words  | ~11 words    |
| Test  | ~4,900    | ~1,900          | ~297 words  | ~11 words    |

- **Answer balance:** A/B/C/D each ~25% → random baseline = 25%  
- **Difficulty:** ~60% middle school, ~40% high school

---

## Slide 4 — EDA: Exploratory Data Analysis

### What We Explored

**1. Article Length Distribution**
- Right-skewed: median ~250 words, max > 1,000
- Long tail → bag-of-words features remain tractable; TF-IDF over-penalises rare but key words

**2. Question Type Breakdown**
| Type | Share |
|------|-------|
| Fill-in-the-blank | ~35% |
| "What" | ~25% |
| "Which" | ~15% |
| Who / Why / How / Other | ~25% |

**3. Answer Label Balance**
- A: 24.8% · B: 25.1% · C: 24.9% · D: 25.2% → perfectly balanced → no class-weighting needed

**4. Keyword Overlap (article ↔ correct option)**
- Correct options average **2.3×** more keyword matches with article than wrong options → confirms overlap signal is useful

---

## Slide 5 — Statistical Analysis

### Methods Applied

| Method | Purpose |
|--------|---------|
| **Cosine Similarity** (CountVectorizer) | Measure article-option semantic proximity |
| **Term Frequency (unigram count)** | Passage-frequency feature for distractor ranking |
| **Character Match Score** | Lexical surface similarity between candidate and answer |
| **K-Means Clustering** (k=2) | Cluster feature space; check separability |
| **Gaussian Mixture Model** (k=2) | Soft cluster boundaries; log-likelihood of fit |
| **Label Propagation** | Semi-supervised: propagate labels to unlabeled examples |
| **Silhouette Score** | Internal validity of clustering |
| **Purity Score** | How well clusters align with true labels |

---

## Slide 6 — Statistical Analysis: Results & Interpretation

### Unsupervised / Semi-Supervised Diagnostics

| Method | Score | Interpretation |
|--------|-------|---------------|
| K-Means Silhouette | **0.0947** | Weak cluster separation — expected in 5,007-dim sparse space |
| K-Means Purity | **0.7469** | 74.7% of items land in the correct-class cluster |
| GMM Silhouette | **0.0130** | Softer decision boundaries than K-Means |
| GMM Log-Likelihood | **5208.39** | Absolute value; shows GMM fits data but not comparable across datasets |
| Label Propagation F1 | **0.4303** | Semi-supervised reaches 43% macro-F1 using only 10% labelled data |

### Key Takeaways
- Low silhouette is **expected** — high-dimensional binary features produce overlapping clusters
- High purity (74.7%) confirms there **is** separable signal even without supervision
- Label Propagation (43% F1) shows semi-supervised can leverage the large unlabelled RACE corpus

---

## Slide 7 — Data Preprocessing

### Full Pipeline

```
Raw CSV  (article, question, A, B, C, D, answer)
    │
    ▼
1. Text Cleaning
   • lowercase  →  remove punctuation  →  strip extra whitespace
   • Applied identically to prediction AND reference (_clean_for_eval)
    │
    ▼
2. Feature Engineering (per option row)
   • One-Hot encoding  — CountVectorizer, top-5000 tokens, binary
     (concatenation of article + question + option text)
   • Cosine similarity — article vector · option vector
   • 5 Lexical features:
       option_len        (token count of option)
       question_len      (token count of question)
       keyword_overlap   (option tokens ∩ article tokens / question tokens)
       option_in_article (fraction of option tokens found in article)
       answer_position   (position of first option match in article)
    │
    ▼
3. Label Construction
   • y = 1 if option == gold answer, else 0
   • 4 rows per question → 4 binary classification instances
    │
    ▼
4. Train / Val / Test Split (pre-split in RACE CSV files)
   • X_train.npz / X_test.npz stored as scipy sparse matrices
   • y_train.npy / y_test.npy stored as NumPy arrays
```

**Final feature dimension:** ~5,007 (5,000 One-Hot + 1 cosine + 5 lexical)

---

## Slide 8 — Model Selection: Why These Models?

### Candidate Evaluation Rationale

| Model | Why Considered | Why Selected / Rejected |
|-------|---------------|------------------------|
| **Logistic Regression** | Linear, interpretable, fast, handles sparse high-dim | ✅ **Selected** — L2 regularisation excels in 5,007-dim space |
| **SVM (LinearSVC + Calibrated)** | Strong for text, good margin maximisation | ✅ **Selected** — near-LR performance; adds diversity to ensemble |
| **Random Forest** | Non-linear, robust to irrelevant features | ⚠️ **Limited** — restricted to 5 lexical features (can't scale to 5,007-dim sparse); used as standalone only |
| **TF-IDF** | Standard text weighting | ❌ **Rejected** — IDF penalises diagnostically important common words ("the answer is...") |
| **Neural / BERT** | State-of-the-art on RACE | ❌ **Out of scope** — CPU-only constraint; would exceed 10 s/sample |
| **XGBoost** | Gradient boosting, powerful | ❌ **Rejected** — dense feature input required; memory overhead too large at 5,007 dims |

### Ensemble Design Decision
- Initial: equal-weight average LR + SVM + RF → **worse** than LR alone
- Root cause: RF's 5-feature predictions dilute LR/SVM's 5,007-feature signals
- Fix: **LR × 0.7 + SVM × 0.3** (RF excluded) → preserves LR's lead, adds marginal SVM benefit

---

## Slide 9 — Model Training

### Model A — Answer Verification

```python
# Binary classifier: label=1 if option is correct answer
# Trained on 4 rows per question

LogisticRegression(solver='saga', C=1.0, max_iter=1000, class_weight=None)
CalibratedClassifierCV(LinearSVC(max_iter=2000), cv=3, method='isotonic')
RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)
                       # trained on X_lex only (5 lexical features)
```

**Weighted Ensemble (inference only):**
```python
_ens_p = 0.7 * lr_model.predict_proba(X)[:, 1]
       + 0.3 * svm_model.predict_proba(X)[:, 1]
# Pick option with highest _ens_p among 4 options per question
```

### Model B — Distractor Ranker + Hint Scorer

```python
# Distractor ranker: trained on (candidate_ngram, gold_distractor) pairs
# Features: [cosine_to_answer, cosine_to_article, passage_freq, char_match, length_ratio]
LogisticRegression(solver='saga', C=1.0)   →  distractor_ranker.pkl

# Hint scorer: trained on (sentence, relevance_label) pairs
# Features: [question_overlap, 0.0, position, sentence_length]
LogisticRegression(solver='saga', C=1.0)   →  hint_scorer.pkl
```

**Training data size:** ~280,000 option rows (train split × 4 options/question)

---

## Slide 10 — Model Testing & Evaluation

### Evaluation Methodology

- **Metric rationale:** Output is **generated text** compared against reference text  
  → BLEU, ROUGE-1, ROUGE-L, METEOR (no Accuracy/F1 as final metrics)
- **Preprocessing symmetry:** `_clean_for_eval()` applied identically to both prediction and reference  
  → lowercase + strip punctuation + normalise whitespace
- **Test set:** held-out RACE test CSV (~4,900 questions, never seen during training)
- **Model B eval:** 200 sampled val questions; hint proxy = passage sentence with max keyword overlap with answer

### Model A Results (Test Set)

| Model | BLEU | ROUGE-1 | ROUGE-L | METEOR |
|-------|------|---------|---------|--------|
| Logistic Regression | **0.3042** | **0.4703** | **0.4637** | **0.4136** |
| SVM (Calibrated) | 0.3033 | 0.4694 | 0.4628 | 0.4126 |
| Random Forest | 0.2591 | 0.4239 | 0.4174 | 0.3662 |
| Ensemble (LR×0.7 + SVM×0.3) | 0.3038 | 0.4698 | 0.4633 | 0.4132 |

### Model B Results (Val Set, n=200)

| Task | BLEU | ROUGE-1 | ROUGE-L | METEOR |
|------|------|---------|---------|--------|
| Distractor Generation | 0.0067 | 0.0762 | 0.0748 | 0.0313 |
| Hint Generation (proxy ref) | 0.0605 | 0.1537 | **0.1375** | 0.1130 |

---

## Slide 11 — Performance Metrics: Interpretation

### Model A — Answer Verification

- **LR BLEU 0.3042** — predicts the correct answer text with ~30% n-gram precision; strong for a bag-of-words system on a hard multi-choice benchmark
- **LR ROUGE-L 0.4637** — the longest common subsequence between predicted and gold answer covers ~46% of the reference; confirms the model selects semantically similar options
- **LR METEOR 0.4136** — paraphrase-aware score of 41%; METEOR is more forgiving of synonym matches than BLEU
- **Ensemble vs LR:** only 0.0004 BLEU below LR — nearly identical, validating the weighted design

### Model B — Distractor & Hints

- **Distractor BLEU 0.0067 (low, expected):** n-gram extracts rarely match full-sentence gold distractors word-for-word; ROUGE-1 0.0762 shows better unigram overlap
- **Hint ROUGE-L 0.1375:** extracted hint sentences share ~14% longest-common-subsequence overlap with the target-sentence proxy reference — meaningful given no gold hints exist in RACE
- **Hint BLEU 0.0605 vs Distractor BLEU 0.0067:** hints are full sentences (longer, more n-gram matches) vs distractors are short 1-2 gram extracts

### Overall System Assessment
| Criterion | Result |
|-----------|--------|
| Best single model | Logistic Regression |
| Ensemble benefit | Marginal but not harmful |
| Inference latency | 0.3 – 1.5 s (CPU) ✅ |
| Unit tests | 7 / 7 pass ✅ |

---

## Slide 12 — System Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                  RACE CSV Files                      │
│         train.csv / val.csv / test.csv               │
└────────────────────┬────────────────────────────────┘
                     │ preprocessing.py
                     ▼
┌─────────────────────────────────────────────────────┐
│  Feature Matrix  X.npz  (scipy sparse, ~5007 cols)  │
│  Labels          y.npy                              │
└────────┬────────────────────┬───────────────────────┘
         │  model_a_train.py  │  model_b_train.py
         ▼                    ▼
   ┌───────────┐      ┌──────────────────────┐
   │ lr_model  │      │  distractor_ranker   │
   │ svm_model │      │  hint_scorer         │
   │ rf_model  │      │  vectorizer_b        │
   └───────────┘      └──────────────────────┘
         │                    │
         └──────┬─────────────┘
                │ inference.py (unified API)
                ▼
   ┌────────────────────────┐
   │   Streamlit  ui/app.py │
   │  ┌──────────────────┐  │
   │  │  Article Input   │  │
   │  │  Quiz View       │  │
   │  │  Hint Panel      │  │
   │  │  Analytics Dash  │  │
   │  └──────────────────┘  │
   └────────────────────────┘
```

---

## Slide 13 — Streamlit UI: 4 Screens

### Screen 1 — Article Input
- Paste any passage **or** click *Load random RACE sample*
- Click *Generate Quiz* → triggers `predict_answer()` + `generate_distractors()` + `get_hints()`

### Screen 2 — Quiz View
- Question text displayed
- `st.radio` for options A / B / C / D
- *Check Answer* → `st.success` (correct) or `st.error` (wrong)
- *"Why?"* expander → keyword overlap explanation

### Screen 3 — Hint Panel
- Hint 1 (general) → Hint 2 (intermediate) → Hint 3 (near-explicit)
- Each in `st.expander`; *Reveal Answer* only visible after all 3 opened
- Graduated disclosure: 0th, 33rd, 66th percentile of hint score distribution

### Screen 4 — Analytics Dashboard
- Model A metrics table + BLEU/ROUGE-L bar chart
- Model B distractor & hint tables
- Unsupervised diagnostics (K-Means, GMM, Label Propagation)
- Session stats via `st.metric` + CSV export

---

## Slide 14 — Key Design Decisions (Summary)

| Decision | Choice | Reason |
|----------|--------|--------|
| Encoding | One-Hot (binary CountVectorizer) | Preserves presence without IDF bias on common Q-words |
| Vocab size | 5,000 tokens | Balances expressivity vs. memory/latency |
| Ensemble | LR×0.7 + SVM×0.3 | RF's 5-feature output adds noise; weighted 2-model keeps gains |
| Preprocessing | `_clean_for_eval()` on both sides | Removes case/punctuation bias from metric computation |
| Hint proxy | Max-overlap article sentence | RACE has no gold hints; keyword overlap is a principled proxy |
| Distractor selection | Top-3 + cosine diversity penalty (< 0.8) | Prevents adjacent n-grams dominating top-3 |
| Metrics | BLEU / ROUGE-1 / ROUGE-L / METEOR | Task is text generation; classification metrics not applicable |

---

## Slide 15 — Limitations & Future Work

### Current Limitations
- **N-gram distractors** are grammatically awkward (e.g., "million km", "5 million")
- **Bag-of-words** ignores word order and long-range coreference
- **Hint proxy reference** is approximate — RACE has no gold hint annotations
- **Cultural domain** — all passages from Chinese school exams; may not generalise

### Future Directions
- Replace n-gram distractor extraction with **T5 / GPT-2** fine-tuned generation
- Replace bag-of-words with **sentence-transformers** (SBERT) embeddings for cosine features
- Add **BERTScore** evaluation alongside BLEU/ROUGE for semantic-level scoring
- Deploy on **Streamlit Cloud** with caching (`@st.cache_resource`) for vectoriser

---

## Slide 16 — Live Demo Outline

### Demo Flow (≈ 3 minutes)

```
1. Open Streamlit app  →  http://localhost:8501

2. Screen 1 — Article Input
   • Click "Load random RACE sample"
   • Show article + auto-generated question

3. Screen 2 — Quiz View
   • Select a wrong option → show st.error feedback
   • Select the correct option → show st.success + "Why?" explanation

4. Screen 3 — Hint Panel
   • Open Hint 1 (general sentence)
   • Open Hint 2 (more specific)
   • Open Hint 3 (near-explicit)
   • Click "Reveal Answer"

5. Screen 4 — Analytics Dashboard
   • Show Model A metrics table + bar chart
   • Show Model B distractor & hint metrics
   • Export CSV

6. Paste a CUSTOM passage (not from RACE)
   • Show the system generating a quiz from unseen text
```

---

## Slide 17 — Results Summary (One-Pager)

### Model A — Answer Verification

| Model | BLEU | ROUGE-L | METEOR |
|-------|------|---------|--------|
| **Logistic Regression** ⭐ | **0.3042** | **0.4637** | **0.4136** |
| SVM (Calibrated) | 0.3033 | 0.4628 | 0.4126 |
| Random Forest | 0.2591 | 0.4174 | 0.3662 |
| Ensemble (LR×0.7+SVM×0.3) | 0.3038 | 0.4633 | 0.4132 |

### Model B — Generation

| Task | BLEU | ROUGE-1 | ROUGE-L | METEOR |
|------|------|---------|---------|--------|
| Distractor Generation | 0.0067 | 0.0762 | 0.0748 | 0.0313 |
| Hint Generation | 0.0605 | 0.1537 | 0.1375 | 0.1130 |

### System Stats
- **7 / 7** unit tests passing
- **< 1.5 s** inference per sample (CPU)
- **4-screen** Streamlit UI
- **~87,866** training questions from RACE

---

## Slide 18 — Thank You

# Thank You

**GitHub:** https://github.com/MMEHDI0606/AI_Lab-  
**Stack:** Python 3.11 · scikit-learn · NLTK · rouge-score · Streamlit

### Questions?

*Key numbers to remember:*  
- LR BLEU **0.3042** — best Model A  
- Hint ROUGE-L **0.1375** — first-ever hint evaluation on this pipeline  
- Ensemble = LR × 0.7 + SVM × 0.3 (no RF)
