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


def extract_candidates(article, answer, max_ngram=2):
    tokens = clean_text(article).split()
    ans_clean = clean_text(answer)
    cands = set()
    for n in range(1, max_ngram+1):
        for i in range(len(tokens)-n+1):
            p = ' '.join(tokens[i:i+n])
            if p != ans_clean and len(p) > 2:
                cands.add(p)
    return list(cands)


def main():
    train_df = pd.read_csv(os.path.join('data/raw', 'train.csv'))
    # Build vocab
    dist_vec = CountVectorizer(binary=True, max_features=3000)
    dist_vec.fit(train_df['article'].apply(clean_text).tolist())
    joblib.dump(dist_vec, os.path.join(MODELS_B, 'vectorizer_b.pkl'))

    SAMPLE = min(500, len(train_df))
    dist_X, dist_y = [], []
    for _, row in train_df.sample(SAMPLE, random_state=42).iterrows():
        correct_ans = row[row['answer']]
        distractors = [row[o] for o in ['A','B','C','D'] if o != row['answer']]
        cands = extract_candidates(row['article'], correct_ans)
        if len(cands) < 3: continue
        for cand in cands[:30]:
            cv = dist_vec.transform([cand])
            av = dist_vec.transform([clean_text(correct_ans)])
            cos = cosine_similarity(cv, av)[0,0]
            freq = clean_text(row['article']).split().count(cand.split()[0]) / max(len(clean_text(row['article']).split()),1)
            char = sum(1 for a,b in zip(cand, clean_text(correct_ans)) if a==b)/max(len(clean_text(correct_ans)),1)
            lr = len(cand.split())/max(len(clean_text(correct_ans).split()),1)
            cart = cosine_similarity(cv, dist_vec.transform([clean_text(row['article'])]))[0,0]
            label = int(any(cand in clean_text(d) or clean_text(d) in cand for d in distractors))
            dist_X.append([cos, cart, freq, char, lr])
            dist_y.append(label)
    dist_X = np.array(dist_X); dist_y = np.array(dist_y)
    ranker = LogisticRegression(max_iter=500).fit(dist_X, dist_y)
    joblib.dump(ranker, os.path.join(MODELS_B, 'distractor_ranker.pkl'))
    print(f'Distractor ranker trained. Acc: {accuracy_score(dist_y, ranker.predict(dist_X)):.4f}')

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
