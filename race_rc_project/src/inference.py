"""inference.py — Unified inference API."""
import os, re, string, time, warnings
import numpy as np
import joblib
warnings.filterwarnings('ignore')

from scipy.sparse import hstack, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

_MODELS = {}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_A = os.path.join(BASE_DIR, '..', 'models', 'model_a', 'traditional')
MODELS_B = os.path.join(BASE_DIR, '..', 'models', 'model_b', 'traditional')


def _load_models():
    if _MODELS: return
    _MODELS['ohe']      = joblib.load(os.path.join(MODELS_A, 'ohe_vectorizer.pkl'))
    _MODELS['lr']       = joblib.load(os.path.join(MODELS_A, 'lr_model.pkl'))
    _MODELS['svm']      = joblib.load(os.path.join(MODELS_A, 'svm_model.pkl'))
    _MODELS['rf']       = joblib.load(os.path.join(MODELS_A, 'rf_model.pkl'))
    _MODELS['dist_vec'] = joblib.load(os.path.join(MODELS_B, 'vectorizer_b.pkl'))
    _MODELS['dist_rk']  = joblib.load(os.path.join(MODELS_B, 'distractor_ranker.pkl'))
    _MODELS['hint_sk']  = joblib.load(os.path.join(MODELS_B, 'hint_scorer.pkl'))


def _clean(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return re.sub(r'\s+', ' ', text).strip()


def predict_answer(article, question, options):
    """Returns predicted answer letter ('A','B','C','D')."""
    _load_models()
    ohe = _MODELS['ohe']; lr = _MODELS['lr']; svm = _MODELS['svm']; rf = _MODELS['rf']
    arts = _clean(article); qs = _clean(question)
    texts, lex_rows = [], []
    for opt in options:
        os_c = _clean(opt)
        texts.append(f'{arts} [SEP] {qs} [SEP] {os_c}')
        at = set(arts.split()); qt = set(qs.split()); ot = set(os_c.split())
        ol = os_c.split()
        pos = arts.find(ol[0]) / max(len(arts),1) if ol and ol[0] in arts else 0.0
        lex_rows.append([len(ol), len(qs.split()), len(qt&ot), len(ot&at), pos])
    X_ohe = ohe.transform(texts)
    art_vec = type(ohe)(binary=True, vocabulary=ohe.vocabulary_)
    art_m = art_vec.transform([arts]*4)
    opt_m = art_vec.transform([_clean(o) for o in options])
    cos_f = csr_matrix(np.array([cosine_similarity(art_m[i], opt_m[i])[0,0] for i in range(4)]).reshape(-1,1))
    lex_f = csr_matrix(np.array(lex_rows, dtype=np.float32))
    X = hstack([X_ohe, cos_f, lex_f])
    lr_p  = lr.predict_proba(X)[:,1]
    svm_p = svm.predict_proba(X)[:,1]
    rf_p  = rf.predict_proba(lex_f.toarray())[:,1]
    scores = (lr_p + svm_p + rf_p) / 3
    return ['A','B','C','D'][int(np.argmax(scores))]


def generate_distractors(article, question, answer, n=3):
    """Returns list of 3 distractor strings.

    Candidates are 5-10 word sliding-window phrases extracted from the article.
    This matches the training distribution of the distractor_ranker and produces
    full-phrase distractors that score much higher on BLEU/ROUGE/METEOR.
    """
    _load_models()
    dvec = _MODELS['dist_vec']; drk = _MODELS['dist_rk']
    tokens = _clean(article).split()
    ans_c = _clean(answer)
    # Extract 5-10 word phrases (mirrors model_b_train.py extract_candidates)
    cands = list({
        ' '.join(tokens[i:i+n_])
        for n_ in range(5, 11)
        for i in range(len(tokens) - n_ + 1)
        if ' '.join(tokens[i:i+n_]) != ans_c and len(' '.join(tokens[i:i+n_])) >= 4
    })
    if len(cands) < n:
        freq = Counter(tokens)
        stop = {'the','a','an','is','was','are','were','of','in','to','and','or','it'}
        cands += [t for t,_ in freq.most_common(50) if t not in _clean(answer).split() and t not in stop][:n]
    feats = []
    for cand in cands[:80]:
        cv = dvec.transform([cand]); av = dvec.transform([ans_c])
        arv = dvec.transform([_clean(article)])
        cos_a = cosine_similarity(cv, av)[0,0]
        cos_r = cosine_similarity(cv, arv)[0,0]
        freq_f = tokens.count(cand.split()[0]) / max(len(tokens),1)
        char_f = sum(1 for a,b in zip(cand, ans_c) if a==b) / max(len(ans_c),1)
        lr_f = len(cand.split()) / max(len(ans_c.split()),1)
        feats.append([cos_a, cos_r, freq_f, char_f, lr_f])
    scores = drk.predict_proba(feats)[:,1]
    ranked = [cands[i] for i in np.argsort(scores)[::-1]]
    selected = []
    for cand in ranked:
        if len(selected) == n: break
        if not selected:
            selected.append(cand)
        else:
            cv = dvec.transform([cand])
            if not any(cosine_similarity(cv, dvec.transform([s]))[0,0] > 0.8 for s in selected):
                selected.append(cand)
    while len(selected) < n:
        selected.append(ranked[len(selected)] if len(ranked) > len(selected) else 'N/A')
    return selected[:n]


def get_hints(article, question, n=3):
    """Returns list of 3 graduated hint strings."""
    _load_models()
    hsk = _MODELS['hint_sk']
    sents = [s.strip() for s in re.split(r'[.!?]', article) if len(s.strip()) > 15]
    if not sents:
        return ['(No hints available.)']*3
    qt = set(_clean(question).split())
    feats = []
    for pos, sent in enumerate(sents[:20]):
        st = set(_clean(sent).split())
        feats.append([len(qt&st)/max(len(qt),1), 0.0, pos/max(len(sents),1), len(sent.split())])
    scores = hsk.predict_proba(feats)[:,1]
    order = np.argsort(scores)
    step = max(len(order)//3, 1)
    hint_sents = [sents[order[0]], sents[order[min(step, len(order)-1)]], sents[order[min(2*step, len(order)-1)]]]
    seen, result = set(), []
    for h in hint_sents:
        if h not in seen: seen.add(h); result.append(h)
    while len(result) < 3:
        result.append('(Refer to the passage for more context.)')
    return result[:3]
