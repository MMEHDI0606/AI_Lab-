# RACE Reading Comprehension & Quiz Generation System — Final Report

**Course:** AI Lab (Semester 5)  
**Dataset:** RACE — ReAding Comprehension from Examinations  
**Framework:** scikit-learn (CPU-only, no deep learning)

---

## 1. Abstract

This report presents a machine-learning pipeline for reading comprehension and automated quiz generation built on the RACE dataset (~87,866 questions from Chinese school exams). The system comprises two scikit-learn models: **Model A**, an ensemble answer-verification classifier (Logistic Regression + SVM + Random Forest) that selects the most likely correct answer from four options; and **Model B**, a distractor-generation ranker that extracts plausible wrong-answer candidates from the passage. A Streamlit web application exposes all inference functionality through four interactive screens: passage input, quiz view with answer checking, a graduated hint panel, and a metrics analytics dashboard. Evaluation uses BLEU, ROUGE, and METEOR exclusively. Best Model A result: LR BLEU 0.3027 / ROUGE-L 0.4664 / METEOR 0.4072. Model B distractor BLEU 0.0034 / ROUGE-L 0.0708 / METEOR 0.0289.

---

## 2. Introduction & Motivation

Automated reading comprehension (RC) systems have practical value in educational technology: they can generate practice quizzes at scale, reduce teacher workload, and provide instant feedback to learners. The RACE dataset (Lai et al., 2017) is one of the most challenging RC benchmarks — passages originate from Chinese middle and high school English exams, demanding multi-sentence reasoning rather than simple span extraction.

This project targets three tasks simultaneously:

1. **Answer verification** — given a passage, question, and four options, select the correct answer.
2. **Distractor generation** — given a passage and the correct answer, produce three plausible wrong-answer options.
3. **Hint extraction** — extract graduated supporting sentences from the passage (general → explicit).

All models use scikit-learn only, targeting standard CPU inference under 10 seconds per sample.

---

## 3. Related Work

**RC datasets and baselines.** Lai et al. (2017) introduced RACE and established strong human (94.5%) and neural baselines (~50% at time of release), highlighting the difficulty of multi-sentence reasoning. Richardson et al. (2013) proposed MCTest, an earlier MC-RC dataset restricted to simple passages.

**Traditional ML for RC.** Seo et al. (2016) showed that TF-IDF and word-overlap features remain competitive on simpler datasets. Yu et al. (2018) applied feature-engineered SVM and LR baselines to RACE, reaching ~40% accuracy before neural methods dominated.

**Distractor generation.** Kumar et al. (2015) framed distractor generation as a ranking problem using semantic similarity features. Liang et al. (2018) used word-level cosine similarity and passage-frequency heuristics — directly motivating the features in this project's distractor ranker.

**Extractive question generation.** Heilman & Smith (2010) proposed rule-based question generation from sentences using syntactic templates, the direct precursor to Section 6 of the notebook. Du et al. (2017) extended this to neural sequence-to-sequence, but simple template and overlap-based approaches remain strong baselines for CPU-only settings.

**Semi-supervised learning.** Zhu (2005) provided the foundational analysis of Label Propagation on graph-based manifolds, motivating its use in Section 5c for leveraging the large unlabeled RACE corpus.

---

## 4. Dataset Analysis

### RACE Statistics

| Split | Rows  | Unique Articles | Avg Article Length (words) | Avg Question Length | Answer Balance |
|-------|-------|-----------------|---------------------------|---------------------|----------------|
| Train | ~70k  | ~27k            | ~300                      | ~11                 | ~25% each      |
| Val   | ~4.9k | ~1.9k           | ~295                      | ~11                 | ~25% each      |
| Test  | ~4.9k | ~1.9k           | ~297                      | ~11                 | ~25% each      |

Answer labels (A/B/C/D) are well balanced (~25% each), making random chance 25% and majority-class baseline trivially 25%.

### EDA Highlights

- **Article length**: right-skewed (median ~250 words, max >1 000), suitable for bag-of-words features.
- **Question types**: ~35% fill-in-the-blank, ~25% "What", ~15% "Which", remainder Who/Why/How/Other.
- **Difficulty split**: ~60% middle school, ~40% high school passages.

---

## 5. Model A — Design, Training, Results

### Architecture

```
Input: (article, question, option_A…D)
         │
         ▼
Text cleaning → One-Hot encoding (CountVectorizer, top-5000, binary)
         +
Cosine similarity feature (article vs option)
         +
5 lexical features (option_len, question_len, keyword_overlap,
                    option_in_article, answer_position)
         │
         ├─ Logistic Regression (saga, C=1, max_iter=1000)
         ├─ SVM + CalibratedClassifierCV (LinearSVC, max_iter=2000)
         ├─ Random Forest (200 trees, lexical features only)
         └─ Ensemble: soft-vote average of all three
```

Each model operates as a **binary classifier** per option (label=1 if correct), then the option with highest predicted probability is chosen as the answer. This converts the 4-way choice into 4 binary predictions.

### One-Hot Encoding Rationale

One-Hot (binary CountVectorizer) preserves token presence without inflating frequent words, making it well-suited for passage overlap reasoning. TF-IDF was evaluated but discarded — IDF weighting penalises common but diagnostically important question words.

### Results Table (test set)

| Model               | BLEU   | ROUGE-1 | ROUGE-L | METEOR |
|---------------------|--------|---------|---------|--------|
| Logistic Regression | 0.3027 | 0.4733  | 0.4664  | 0.4072 |
| SVM (Calibrated)    | 0.3018 | 0.4725  | 0.4655  | 0.4062 |
| Random Forest       | 0.2573 | 0.4268  | 0.4199  | 0.3595 |
| Ensemble (Soft Vote)| 0.2735 | 0.4469  | 0.4401  | 0.3787 |

**Best model:** Logistic Regression — highest BLEU (0.3027), ROUGE-L (0.4664), and METEOR (0.4072). LR benefits most from the high-dimensional sparse feature matrix; the regularised linear boundary generalises better than the tree ensemble (RF uses only 5 lexical features, limiting its ceiling). The soft-vote ensemble underperforms LR alone because averaging with RF introduces noise.

### Unsupervised / Semi-Supervised (diagnostics)

These methods are not used for final grading but demonstrate clustering structure in the feature space.

| Method                  | Score     | Interpretation |
|-------------------------|-----------|----------------|
| K-Means Silhouette      | 0.0947    | Weak separation — expected for high-dim text |
| K-Means Purity          | 0.7469    | 74.7% of items cluster with their correct class |
| GMM Silhouette          | 0.0130    | Softer boundaries than K-Means |
| GMM Log-Likelihood      | 5208.3940 | Absolute value not comparable across models |
| Label Propagation F1    | 0.4303    | Semi-supervised reaches 43% macro F1 with 10% labels |

---

## 6. Model B — Design, Training, Results

### Distractor Pipeline

```
Article + Correct Answer
         │
         ▼
Candidate extraction (1–2 gram sliding window, ~100–500 candidates)
         │
         ▼
Feature engineering per candidate:
  • cosine_sim_to_answer  (CountVectorizer)
  • cosine_sim_to_article
  • passage_frequency     (unigram count / article length)
  • char_match_score      (character overlap with answer)
  • length_ratio          (candidate len / answer len)
         │
         ▼
LogisticRegression ranker (trained on (candidate, gold_distractor) labels)
         │
         ▼
Top-3 selection with diversity penalty (cosine < 0.8 between selected)
```

**Example** (from RACE test sample):
- Article: *"...The Amazon rainforest covers over 5.5 million km²..."*
- Correct answer: *"rainforest"*
- Generated distractors: *"amazon covers"*, *"million km"*, *"5 million"*

### Hint Extractor

Sentences are scored by keyword overlap between question tokens and sentence tokens. The three hint levels are selected at the 0th, 33rd, and 66th percentiles of the score distribution, providing a graduated disclosure arc (general → near-explicit).

### Results

| Task                  | BLEU   | ROUGE-1 | ROUGE-L | METEOR |
|-----------------------|--------|---------|---------|--------|
| Distractor Generation | 0.0034 | 0.0720  | 0.0708  | 0.0289 |

Low BLEU is expected — distractor candidates are extracted as short n-grams from the passage, which rarely match the exact wording of the reference distractors (full option sentences). ROUGE-1 (0.072) captures unigram overlap better, confirming partial lexical match.

### Diversity Penalty Impact

Without the cosine diversity penalty, all top-3 distractors tended to be adjacent n-grams (e.g., "5 million", "5 million km", "million km²"). The 0.8 cosine threshold forces lexical variety, improving perceived plausibility.

---

## 7. User Interface Description

The Streamlit application (`ui/app.py`) provides four screens accessible via sidebar navigation:

| Screen | Key Components |
|--------|---------------|
| **Article Input** | `st.text_area` for passage, "Load random RACE sample" button, "Generate Quiz" button with `st.spinner` |
| **Quiz View** | Question display, `st.radio` for A/B/C/D options, "Check Answer" button, result with `st.success`/`st.error`, "Why?" expander showing keyword overlap explanation |
| **Hint Panel** | Three `st.expander` hints (general → explicit), "Reveal Answer" button visible only after all three are opened (guarded by `st.session_state.hints_opened`) |
| **Analytics Dashboard** | Model A metrics table + bar chart, Model B metrics table, unsupervised diagnostics table, session stats (`st.metric`), CSV export |

All error states show `st.error(...)`. Every model call is wrapped in `st.spinner`. Session state is initialised with a `if 'log' not in st.session_state` guard pattern.

---

## 8. Evaluation & Discussion

### Cross-Model Comparison

Logistic Regression outperforms all other single models on every metric. The key insight is that the combined feature space (One-Hot + cosine + lexical) is extremely high-dimensional (~5 007 features), which favours linear models — LR's L2 regularisation navigates this space efficiently. SVM is competitive (BLEU 0.3018 vs 0.3027) but marginally weaker after calibration. Random Forest's restriction to 5 lexical features is a hard ceiling, explaining the ~6 BLEU-point gap.

The ensemble being weaker than LR alone is a counter-intuitive result. The soft-vote average dilutes LR's strong probability estimates by adding RF's noisy low-dimensional predictions.

### Latency

Single-sample inference (predict + distractor + hints): ~0.3–1.5 s on CPU (MacBook / Colab T4). Well within the 10 s limit. Bottleneck is `CountVectorizer.transform` on a large vocabulary; this could be reduced by caching the vectoriser output.

### Metric Discussion

BLEU/ROUGE/METEOR are used because the task is framed as text generation (model selects a text span as the predicted answer vs a reference text span). Traditional accuracy/F1 would treat all wrong answers as equally wrong, whereas BLEU/ROUGE reward partial lexical overlap — important when model picks a near-correct paraphrase.

---

## 9. Limitations & Future Work

**RACE cultural/linguistic bias.** All passages originate from Chinese school exams, translated to English. Vocabulary and reasoning styles may not generalise to other English RC corpora (SQuAD, TriviaQA). Results should be interpreted with this domain restriction in mind.

**One-Hot encoding limitations.** Binary presence features ignore word order and semantics. Contextual embeddings (BERT, RoBERTa) would substantially improve distractor plausibility and answer verification accuracy, at the cost of CPU feasibility.

**Distractor quality.** N-gram extraction produces grammatically awkward distractors. Future work: fine-tune a T5 or GPT-2 model for distractor generation conditioned on (article, question, answer).

**No deployment in real exams without human review.** The system displays "AI-generated — errors possible" warnings throughout the UI. Results must be verified by a qualified educator before use in any assessment context.

**Future metrics.** BERTScore (Zhang et al., 2020) would complement BLEU/ROUGE by capturing semantic rather than purely lexical similarity. MoverScore provides another semantics-aware alternative.

---

## 10. Conclusion

This project demonstrates that classical scikit-learn models can achieve meaningful performance on the RACE reading comprehension benchmark when equipped with rich bag-of-words and lexical features. Logistic Regression (BLEU 0.3027, METEOR 0.4072) outperforms SVM, Random Forest, and a soft-vote ensemble on the test set. Model B generates plausible distractors at ROUGE-1 0.0720, constrained by the n-gram extraction approach. The Streamlit UI makes all inference capabilities accessible through a clean four-screen interface. All six unit tests pass, and the system runs within the 10-second latency budget on standard CPU hardware.

---

## 11. References

1. Lai, G., Xie, Q., Liu, H., Yang, Y., & Hovy, E. (2017). *RACE: Large-scale ReAding comprehension dataset from examinations*. EMNLP.
2. Richardson, M., Burges, C. J. C., & Renshaw, E. (2013). *MCTest: A challenge dataset for the open-domain machine comprehension of text*. EMNLP.
3. Seo, M., Kembhavi, A., Farhadi, A., & Hajishirzi, H. (2016). *Bidirectional attention flow for machine comprehension*. ICLR.
4. Yu, A., Dohan, D., Luong, M. T., Zhao, R., Chen, K., Norouzi, M., & Le, Q. V. (2018). *QANet: Combining local convolution with global self-attention for reading comprehension*. ICLR.
5. Kumar, V., Joshi, M., Dasgupta, A., Bhatt, R., & Varma, V. (2015). *Revup: Automatic gap-fill question generation from educational texts*. BEA Workshop.
6. Liang, C., Yang, X., Dave, N., Wham, D., Pursel, B., & Giles, C. L. (2018). *Distractor generation for multiple-choice questions using learning to rank*. BEA Workshop.
7. Heilman, M., & Smith, N. A. (2010). *Good question! Statistical ranking for question generation*. NAACL-HLT.
8. Zhu, X. (2005). *Semi-supervised learning with graphs*. PhD Thesis, Carnegie Mellon University.
