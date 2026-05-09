"""tests/test_inference.py — Unit tests for RACE inference API."""
import os
import sys
import time
import random

import pytest
import pandas as pd

# ── Path setup ──────────────────────────────────────────────────────────────
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT      = os.path.dirname(_TESTS_DIR)
_SRC       = os.path.join(_ROOT, 'src')
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# Evict cached HuggingFace evaluate if present
for _mod in list(sys.modules.keys()):
    if _mod == 'evaluate' or _mod.startswith('evaluate.'):
        del sys.modules[_mod]

from inference import predict_answer, generate_distractors, get_hints
from evaluate import compute_generation_metrics

# ── Load test set once ──────────────────────────────────────────────────────
_TEST_CSV = os.path.join(_ROOT, 'data', 'raw', 'test.csv')

@pytest.fixture(scope='module')
def test_df():
    if not os.path.exists(_TEST_CSV):
        pytest.skip('test.csv not found — run the notebook to generate data.')
    return pd.read_csv(_TEST_CSV)

@pytest.fixture(scope='module')
def sample_row(test_df):
    return test_df.sample(1, random_state=42).iloc[0]

# ── Tests ────────────────────────────────────────────────────────────────────

def test_predict_answer_returns_valid_label(sample_row):
    article  = str(sample_row['article'])
    question = str(sample_row['question'])
    options  = [str(sample_row[o]) for o in ['A', 'B', 'C', 'D']]
    result   = predict_answer(article, question, options)
    assert result in ['A', 'B', 'C', 'D'], f"Expected A/B/C/D, got {result!r}"


def test_generate_distractors_returns_three(sample_row):
    article = str(sample_row['article'])
    question = str(sample_row['question'])
    answer  = str(sample_row[sample_row['answer']])
    result  = generate_distractors(article, question, answer)
    assert isinstance(result, list), "generate_distractors must return a list"
    assert len(result) == 3, f"Expected 3 distractors, got {len(result)}"


def test_generate_distractors_excludes_correct_answer(sample_row):
    article  = str(sample_row['article'])
    question = str(sample_row['question'])
    answer   = str(sample_row[sample_row['answer']])
    result   = generate_distractors(article, question, answer)
    for d in result:
        assert d.strip().lower() != answer.strip().lower(), \
            f"Distractor {d!r} matches the correct answer {answer!r}"


def test_get_hints_returns_three(sample_row):
    article  = str(sample_row['article'])
    question = str(sample_row['question'])
    result   = get_hints(article, question)
    assert isinstance(result, list), "get_hints must return a list"
    assert len(result) == 3, f"Expected 3 hints, got {len(result)}"
    assert all(isinstance(h, str) for h in result), "All hints must be strings"


def test_no_distractor_equals_correct_answer(test_df):
    sample = test_df.sample(min(20, len(test_df)), random_state=0)
    for _, row in sample.iterrows():
        article  = str(row['article'])
        question = str(row['question'])
        answer   = str(row[row['answer']])
        distractors = generate_distractors(article, question, answer)
        for d in distractors:
            assert d.strip().lower() != answer.strip().lower(), \
                f"Row answer {answer!r} appeared as a distractor"


def test_inference_latency(sample_row):
    article  = str(sample_row['article'])
    question = str(sample_row['question'])
    options  = [str(sample_row[o]) for o in ['A', 'B', 'C', 'D']]
    answer   = str(sample_row[sample_row['answer']])

    t0 = time.time()
    predict_answer(article, question, options)
    generate_distractors(article, question, answer)
    get_hints(article, question)
    elapsed = time.time() - t0

    assert elapsed < 10.0, f"Inference took {elapsed:.2f}s (limit: 10s)"


def test_generation_metrics_return_valid_ranges():
    preds = ['the cat sat on the mat', 'it was a sunny day', 'she went to the store']
    refs  = ['a cat rested on the rug',  'the weather was bright', 'she visited the shop']
    m = compute_generation_metrics(preds, refs)
    required_keys = {'bleu', 'rouge_1', 'rouge_l', 'meteor'}
    assert required_keys.issubset(m.keys()), f"Missing keys: {required_keys - m.keys()}"
    for key in required_keys:
        assert 0.0 <= m[key] <= 1.0, f"{key}={m[key]} is out of [0, 1]"


def test_generation_metrics_reject_class_labels():
    with pytest.raises(ValueError):
        compute_generation_metrics(['1', '0', '1'], ['1', '1', '0'])
