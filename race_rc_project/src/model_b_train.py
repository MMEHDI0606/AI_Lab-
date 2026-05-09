"""model_b_train.py — Train Model B (distractor ranker + hint scorer)."""
import os, sys, re, string, warnings
import numpy as np
import pandas as pd
import joblib
warnings.filterwarnings('ignore')

from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROCESSED = 'data/processed'
MODELS_B  = 'models/model_b/traditional'


def clean_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return re.sub(r'\s+', ' ', text).strip()


# Minimum / maximum phrase length (in words) for distractor candidates.
# Longer phrases match gold distractors far better than 1-2 gram snippets.
MIN_PHRASE_WORDS = 5
MAX_PHRASE_WORDS = 10


def extract_candidates(article, answer, min_words=MIN_PHRASE_WORDS, max_words=MAX_PHRASE_WORDS):
    """Extract 5-10 word sliding-window phrases from the article as distractor candidates.

    Using longer phrases (matching the typical length of real exam distractors)
    dramatically improves BLEU/ROUGE/METEOR versus the old 1-2 gram approach.
    """
    tokens = clean_text(article).split()
    ans_clean = clean_text(answer)
    cands = set()
    for n in range(min_words, max_words + 1):
        for i in range(len(tokens) - n + 1):
            phrase = ' '.join(tokens[i:i + n])
            # Skip if identical to the answer or too short to be meaningful
            if phrase != ans_clean and len(phrase) >= 4:
                cands.add(phrase)
    return list(cands)


def main():
    train_df = pd.read_csv(os.path.join('data/raw', 'train.csv'))
    # Build vocab
    dist_vec = CountVectorizer(binary=True, max_features=3000)
    dist_vec.fit(train_df['article'].apply(clean_text).tolist())
    joblib.dump(dist_vec, os.path.join(MODELS_B, 'vectorizer_b.pkl'))

    # Increase training sample so ranker sees enough long-phrase examples
    SAMPLE = min(1500, len(train_df))
    dist_X, dist_y = [], []
    for _, row in train_df.sample(SAMPLE, random_state=42).iterrows():
        correct_ans = row[row['answer']]
        distractors = [row[o] for o in ['A','B','C','D'] if o != row['answer']]
        # Use the new long-phrase extractor (5-10 words)
        cands = extract_candidates(row['article'], correct_ans)
        if len(cands) < 3:
            continue
        # Evaluate up to 60 candidates per row (more variety for the ranker)
        for cand in cands[:60]:
            cv   = dist_vec.transform([cand])
            av   = dist_vec.transform([clean_text(correct_ans)])
            arv  = dist_vec.transform([clean_text(row['article'])])
            cos  = cosine_similarity(cv, av)[0, 0]
            cart = cosine_similarity(cv, arv)[0, 0]
            freq = clean_text(row['article']).split().count(cand.split()[0]) / max(len(clean_text(row['article']).split()), 1)
            char = sum(1 for a, b in zip(cand, clean_text(correct_ans)) if a == b) / max(len(clean_text(correct_ans)), 1)
            lr   = len(cand.split()) / max(len(clean_text(correct_ans).split()), 1)
            # Positive label: candidate is a substring of (or contains) a real gold distractor
            label = int(any(cand in clean_text(d) or clean_text(d) in cand for d in distractors))
            dist_X.append([cos, cart, freq, char, lr])
            dist_y.append(label)
    dist_X = np.array(dist_X)
    dist_y = np.array(dist_y)
    ranker = LogisticRegression(max_iter=1000, C=1.0).fit(dist_X, dist_y)
    joblib.dump(ranker, os.path.join(MODELS_B, 'distractor_ranker.pkl'))
    print(f'Distractor ranker trained. Acc: {accuracy_score(dist_y, ranker.predict(dist_X)):.4f}')
    print(f'  Training set size: {len(dist_X)}, positive rate: {dist_y.mean():.3f}')

    hint_X, hint_y = [], []
    for _, row in train_df.sample(SAMPLE, random_state=42).iterrows():
        sents = [s.strip() for s in re.split(r'[.!?]', row['article']) if len(s.strip())>15]
        qt = set(clean_text(row['question']).split())
        at = set(clean_text(row[row['answer']]).split())
        for pos, sent in enumerate(sents[:15]):
            st = set(clean_text(sent).split())
            hint_X.append([len(qt&st)/max(len(qt),1), len(at&st)/max(len(at),1), pos/max(len(sents),1), len(sent.split())])
            hint_y.append(1 if len(at&st)/max(len(at),1) > 0.3 else 0)
    hint_X = np.array(hint_X); hint_y = np.array(hint_y)
    hint_scorer = LogisticRegression(max_iter=500).fit(hint_X, hint_y)
    joblib.dump(hint_scorer, os.path.join(MODELS_B, 'hint_scorer.pkl'))
    print('Hint scorer trained and saved.')


if __name__ == '__main__':
    main()
