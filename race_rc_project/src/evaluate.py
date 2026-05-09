"""evaluate.py — Metric computation utilities."""
import re
import string

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, r2_score
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer as _rouge_mod


def compute_metrics(y_true, y_pred, y_proba=None, n_options=4):
    metrics = {
        'accuracy':  accuracy_score(y_true, y_pred),
        'macro_f1':  f1_score(y_true, y_pred, average='macro', zero_division=0),
        'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall':    recall_score(y_true, y_pred, average='macro', zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
    }
    if y_proba is not None:
        n = len(y_true) // n_options
        correct = sum(
            np.argmax(y_proba[i*n_options:(i+1)*n_options]) == np.argmax(y_true[i*n_options:(i+1)*n_options])
            for i in range(n)
            if y_true[i*n_options:(i+1)*n_options].sum() > 0
        )
        metrics['exact_match'] = correct / max(n, 1)
    return metrics


# ── Generation metric helpers ──────────────────────────────────────────────

def _clean_for_eval(text):
    """Lowercase, strip punctuation, normalise whitespace before scoring."""
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return re.sub(r'\s+', ' ', text).strip()


def compute_generation_metrics(pred_texts, ref_texts):
    """Compute corpus-average BLEU, ROUGE-1, ROUGE-L, and METEOR in [0, 1].

    Both sides are cleaned (lowercase + no punctuation) before scoring to
    ensure symmetry between prediction and reference preprocessing.
    """
    assert len(pred_texts) == len(ref_texts), "pred_texts and ref_texts must have same length"
    if not pred_texts:
        return {"bleu": 0.0, "rouge_1": 0.0, "rouge_l": 0.0, "meteor": 0.0}

    smooth = SmoothingFunction().method1
    scorer = _rouge_mod.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

    bleu_vals   = []
    rouge1_vals = []
    rougel_vals = []
    meteor_vals = []

    for pred, ref in zip(pred_texts, ref_texts):
        pred = _clean_for_eval(pred)
        ref  = _clean_for_eval(ref)
        pred_tok = pred.split()
        ref_tok  = ref.split()

        bleu  = sentence_bleu([ref_tok], pred_tok, smoothing_function=smooth) if pred_tok and ref_tok else 0.0
        rouge = scorer.score(ref, pred)
        met   = meteor_score([ref_tok], pred_tok) if pred_tok and ref_tok else 0.0

        bleu_vals.append(float(bleu))
        rouge1_vals.append(float(rouge["rouge1"].fmeasure))
        rougel_vals.append(float(rouge["rougeL"].fmeasure))
        meteor_vals.append(float(met))

    n = len(pred_texts)
    return {
        "bleu":    sum(bleu_vals)   / n,
        "rouge_1": sum(rouge1_vals) / n,
        "rouge_l": sum(rougel_vals) / n,
        "meteor":  sum(meteor_vals) / n,
    }
