"""Reusable preprocessing pipeline for RACE model training."""

import os
import re
import string

import joblib
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, hstack, load_npz, save_npz
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DEFAULT_RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
DEFAULT_PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
DEFAULT_MODELS_DIR = os.path.join(PROJECT_ROOT, 'models', 'model_a', 'traditional')


def _resolve_path(path_value):
	if os.path.isabs(path_value):
		return path_value
	return os.path.join(PROJECT_ROOT, path_value)


def clean_text(text):
	"""Lowercase, remove punctuation, strip extra whitespace."""
	text = str(text).lower()
	text = text.translate(str.maketrans('', '', string.punctuation))
	return re.sub(r'\s+', ' ', text).strip()


def prepare_text_columns(df):
	"""Return a copy with cleaned article/question/option columns added."""
	result = df.copy()
	result['article_clean'] = result['article'].apply(clean_text)
	result['question_clean'] = result['question'].apply(clean_text)
	for option in ['A', 'B', 'C', 'D']:
		result[f'{option}_clean'] = result[option].apply(clean_text)
	return result


def expand_df(df):
	"""Expand one MCQ row into four option rows for binary verification."""
	rows = []
	for _, row in df.iterrows():
		article_clean = row.get('article_clean', clean_text(row['article']))
		question_clean = row.get('question_clean', clean_text(row['question']))
		for option in ['A', 'B', 'C', 'D']:
			option_clean = row.get(f'{option}_clean', clean_text(row[option]))
			rows.append(
				{
					'article': article_clean,
					'question': question_clean,
					'option': option_clean,
					'option_letter': option,
					'label': 1 if row['answer'] == option else 0,
					'combined_text': f'{article_clean} [SEP] {question_clean} [SEP] {option_clean}',
					'article_raw': row['article'],
					'question_raw': row['question'],
					'A_raw': row['A'],
					'B_raw': row['B'],
					'C_raw': row['C'],
					'D_raw': row['D'],
					'answer': row['answer'],
				}
			)
	return pd.DataFrame(rows)


def load_raw_splits(raw_dir=DEFAULT_RAW_DIR):
	"""Load train/val/test CSV splits from disk."""
	raw_dir = _resolve_path(raw_dir)
	train_df = pd.read_csv(os.path.join(raw_dir, 'train.csv'))
	val_df = pd.read_csv(os.path.join(raw_dir, 'val.csv'))
	test_df = pd.read_csv(os.path.join(raw_dir, 'test.csv'))
	return train_df, val_df, test_df


def _cosine_feature(df_expanded, vocabulary):
	article_vectorizer = CountVectorizer(binary=True, vocabulary=vocabulary)
	article_matrix = article_vectorizer.transform(df_expanded['article'])
	option_matrix = article_vectorizer.transform(df_expanded['option'])
	values = np.array(
		[cosine_similarity(article_matrix[i], option_matrix[i])[0, 0] for i in range(article_matrix.shape[0])],
		dtype=np.float32,
	).reshape(-1, 1)
	return csr_matrix(values)


def _lexical_features(df_expanded):
	feature_rows = []
	for _, row in df_expanded.iterrows():
		article_tokens = set(row['article'].split())
		question_tokens = set(row['question'].split())
		option_tokens = set(row['option'].split())
		option_words = row['option'].split()
		article_text = row['article']
		position = 0.0
		if option_words and option_words[0] in article_text:
			position = article_text.find(option_words[0]) / max(len(article_text), 1)
		feature_rows.append(
			[
				len(option_words),
				len(row['question'].split()),
				len(question_tokens & option_tokens),
				len(option_tokens & article_tokens),
				position,
			]
		)
	return csr_matrix(np.array(feature_rows, dtype=np.float32))


def build_features(
	train_exp,
	val_exp,
	test_exp,
	save_dir=DEFAULT_PROCESSED_DIR,
	models_dir=DEFAULT_MODELS_DIR,
	max_features=5000,
):
	"""Build and save sparse feature matrices plus labels."""
	save_dir = _resolve_path(save_dir)
	models_dir = _resolve_path(models_dir)
	os.makedirs(save_dir, exist_ok=True)
	os.makedirs(models_dir, exist_ok=True)

	vectorizer = CountVectorizer(binary=True, max_features=max_features, min_df=2)
	train_ohe = vectorizer.fit_transform(train_exp['combined_text'])
	val_ohe = vectorizer.transform(val_exp['combined_text'])
	test_ohe = vectorizer.transform(test_exp['combined_text'])
	joblib.dump(vectorizer, os.path.join(models_dir, 'ohe_vectorizer.pkl'))

	feature_sets = [
		('train', train_exp, train_ohe),
		('val', val_exp, val_ohe),
		('test', test_exp, test_ohe),
	]
	for split_name, expanded_df, sparse_ohe in feature_sets:
		full_matrix = hstack(
			[
				sparse_ohe,
				_cosine_feature(expanded_df, vectorizer.vocabulary_),
				_lexical_features(expanded_df),
			]
		)
		save_npz(os.path.join(save_dir, f'X_{split_name}.npz'), full_matrix)
		np.save(os.path.join(save_dir, f'y_{split_name}.npy'), expanded_df['label'].to_numpy())

	return vectorizer


def save_expanded_splits(train_exp, val_exp, test_exp, save_dir=DEFAULT_PROCESSED_DIR):
	"""Persist expanded row-per-option dataframes for inspection/debugging."""
	save_dir = _resolve_path(save_dir)
	os.makedirs(save_dir, exist_ok=True)
	train_exp.to_csv(os.path.join(save_dir, 'train_exp.csv'), index=False)
	val_exp.to_csv(os.path.join(save_dir, 'val_exp.csv'), index=False)
	test_exp.to_csv(os.path.join(save_dir, 'test_exp.csv'), index=False)


def preprocess_and_build(raw_dir=DEFAULT_RAW_DIR, save_dir=DEFAULT_PROCESSED_DIR, models_dir=DEFAULT_MODELS_DIR):
	"""Full raw-CSV to saved-feature pipeline for script-based training."""
	train_df, val_df, test_df = load_raw_splits(raw_dir)
	train_exp = expand_df(prepare_text_columns(train_df))
	val_exp = expand_df(prepare_text_columns(val_df))
	test_exp = expand_df(prepare_text_columns(test_df))
	save_expanded_splits(train_exp, val_exp, test_exp, save_dir=save_dir)
	build_features(train_exp, val_exp, test_exp, save_dir=save_dir, models_dir=models_dir)
	return train_exp, val_exp, test_exp


def load_features(processed_dir=DEFAULT_PROCESSED_DIR):
	"""Load saved train/val/test sparse features and labels."""
	processed_dir = _resolve_path(processed_dir)
	X_train = load_npz(os.path.join(processed_dir, 'X_train.npz'))
	X_val = load_npz(os.path.join(processed_dir, 'X_val.npz'))
	X_test = load_npz(os.path.join(processed_dir, 'X_test.npz'))
	y_train = np.load(os.path.join(processed_dir, 'y_train.npy'))
	y_val = np.load(os.path.join(processed_dir, 'y_val.npy'))
	y_test = np.load(os.path.join(processed_dir, 'y_test.npy'))
	return X_train, X_val, X_test, y_train, y_val, y_test


__all__ = [
	'build_features',
	'clean_text',
	'expand_df',
	'load_features',
	'load_raw_splits',
	'prepare_text_columns',
	'preprocess_and_build',
	'save_expanded_splits',
]
