"""model_a_train.py — Train all Model A classifiers."""
import os, sys, time, warnings
import numpy as np
import joblib
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import load_features
from evaluate import compute_metrics

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import load_npz

PROCESSED = 'data/processed'
MODELS_A  = 'models/model_a/traditional'


def main():
    X_tr, X_va, X_te, y_tr, y_va, y_te = load_features(PROCESSED)
    # Lexical cols are last 5
    X_lex_tr = X_tr[:, -5:].toarray()
    X_lex_va = X_va[:, -5:].toarray()
    X_lex_te = X_te[:, -5:].toarray()

    classifiers = [
        ('lr',  LogisticRegression(max_iter=1000, C=1.0, solver='saga', n_jobs=-1), X_tr, X_va),
        ('svm', CalibratedClassifierCV(LinearSVC(max_iter=2000), cv=3), X_tr, X_va),
        ('rf',  RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1), X_lex_tr, X_lex_va),
    ]
    for name, clf, Xtr, Xva in classifiers:
        t0 = time.time()
        clf.fit(Xtr, y_tr)
        elapsed = time.time() - t0
        preds = clf.predict(Xva)
        m = compute_metrics(y_va, preds)
        print(f'{name}: acc={m["accuracy"]:.4f}, f1={m["macro_f1"]:.4f}, time={elapsed:.1f}s')
        joblib.dump(clf, os.path.join(MODELS_A, f'{name}_model.pkl'))
    print('All Model A classifiers trained and saved.')


if __name__ == '__main__':
    main()
