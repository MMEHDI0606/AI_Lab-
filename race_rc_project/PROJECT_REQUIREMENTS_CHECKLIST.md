# Project Requirements Checklist

This checklist combines the requirements in `AL2002_LabProject.pdf` with the instructor update that final evaluation and reporting must use BLEU, ROUGE, and METEOR for generated text.

## Metric Policy

- [x] Final evaluation/reporting uses BLEU score.
- [x] Final evaluation/reporting uses ROUGE score.
- [x] Final evaluation/reporting uses METEOR score.
- [x] Accuracy, Precision, Recall, F1, Exact Match, and Confusion Matrix are treated only as optional internal diagnostics, not final project metrics.
- [x] Evaluation focuses on similarity between generated output and reference text in wording, meaning, and structure.

## Core System Requirements

- [x] Use the RACE dataset with columns `id, article, question, A, B, C, D, answer`.
- [x] Build Model A for question/answer generation or verification using traditional ML.
- [x] Build Model B for distractor generation and hint generation using traditional ML.
- [x] Use One-Hot / bag-of-words style features as the primary classical representation.
- [x] Provide a user-facing UI integrating both models.
- [x] Keep inference under 10 seconds per sample.
- [x] Keep the training pipeline reproducible from the notebook / scripts.

## Model A Requirements

- [x] Implement at least two traditional ML models and compare them.
- [x] Include Logistic Regression.
- [x] Include SVM.
- [x] Include Random Forest.
- [x] Include Naive Bayes for question-type classification.
- [x] Include an ensemble strategy.
- [x] Include at least one unsupervised or semi-supervised approach.
- [x] Report BLEU/ROUGE/METEOR for final answer-generation/verification evaluation.
- [ ] Ensure every saved training script reports text-level metrics only from predicted answer text vs reference answer text.

## Model B Requirements

- [x] Generate three plausible distractors.
- [x] Apply feature engineering for candidate distractor ranking.
- [x] Use a classical ML ranker for distractor selection.
- [x] Generate graduated hints.
- [x] Use BLEU/ROUGE/METEOR for final distractor/hint evaluation and reporting.
- [ ] Add an explicit text-level evaluation artifact for generated hints against reference hints or chosen gold supporting sentences.

## UI Requirements

- [x] Screen 1: article input.
- [x] Screen 1: random RACE sample loader.
- [x] Screen 2: quiz view with four options and answer checking.
- [x] Screen 3: graduated hint panel.
- [x] Screen 3: reveal-answer gate after hints are used.
- [x] Screen 4: analytics / developer dashboard.
- [x] Friendly error handling is present.
- [x] Loading indicator is present during inference.
- [x] UI warns that AI-generated answers may contain errors.
- [ ] Keyboard navigation and other accessibility details are not explicitly documented.

## Repository / Submission Requirements

- [x] `requirements.txt` exists.
- [ ] `requirements.txt` is not fully pinned to exact versions.
- [x] `README.md` exists with setup and run instructions.
- [x] Notebook exists.
- [x] Trained model checkpoints exist under `models/`.
- [x] Final report content exists in Markdown.
- [ ] Final report PDF is not present in the repository.
- [x] Tests exist.
- [ ] `src/preprocessing.py` is currently empty and should contain the reusable preprocessing pipeline described in the brief.
- [ ] Human evaluation form is not present in the repository.
- [ ] Demo video / live demo artifact is not present in the repository.
- [ ] Clean commit history was not audited here.

## Final Report Structure

- [x] Abstract.
- [x] Introduction & Motivation.
- [x] Related Work.
- [x] Dataset Analysis.
- [x] Model A: Design, Training, Results.
- [x] Model B: Design, Training, Results.
- [x] User Interface Description.
- [x] Evaluation & Discussion.
- [x] Limitations & Future Work.
- [x] Conclusion.
- [x] References.
- [x] At least 5 references are included.

## Recommended Next Fixes

- [ ] Regenerate or patch the notebook so it no longer computes BLEU/ROUGE/METEOR on binary labels inside training cells.
- [ ] Export the report to PDF and add it under `report/`.
- [ ] Implement `src/preprocessing.py` so the training scripts are self-contained outside the notebook.
- [ ] Add or attach the required human-evaluation rubric/form if the course still expects it.