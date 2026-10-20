"""Nạp mô hình đã huấn luyện và thực hiện dự đoán cho ứng dụng Streamlit.

Mô hình phân loại có sẵn trong models/. Mô hình hồi quy và phân cụm được
huấn luyện lại theo đúng công thức trong notebook 06/05 nếu file chưa tồn tại.
"""

from __future__ import annotations

import re
import warnings
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data' / 'processed'
RAW_DIR = ROOT / 'data' / 'raw'
MODELS_DIR = ROOT / 'models'

CLASSIFICATION_PATH = MODELS_DIR / 'classification' / 'classification_pipeline.pkl'
REGRESSION_PATH = MODELS_DIR / 'regression' / 'regression_pipeline.pkl'
CLUSTERING_DIR = MODELS_DIR / 'clustering'
PCA_PATH = CLUSTERING_DIR / 'pca_model.pkl'
KMEANS_PATH = CLUSTERING_DIR / 'kmeans_model.pkl'
NOTEBOOK_CLUSTER_FILES = {
    'preprocess': CLUSTERING_DIR / 'preprocessing.joblib',
    'pca': CLUSTERING_DIR / 'pca.joblib',
    'kmeans': CLUSTERING_DIR / 'kmeans_with_pca.joblib',
}

RANDOM_STATE = 42
POS_LABEL = 'High'

GENRE_LIST = [
    'Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime',
    'Documentary', 'Drama', 'Family', 'Fantasy', 'History', 'Horror',
    'Music', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Sport',
    'Thriller', 'War', 'Western',
]
GENRE_FLAG_COLS = ['genre_' + g.lower().replace('-', '_') for g in GENRE_LIST]

NUM_CLASSIFICATION = ['year', 'duration_minutes', 'metascore', 'genre_count',
                      'cast_count', 'title_length']
CAT_CLASSIFICATION = ['primary_genre', 'certificates']
CLASSIFICATION_FEATURES = NUM_CLASSIFICATION + GENRE_FLAG_COLS + CAT_CLASSIFICATION

REGRESSION_FEATURES = ['year', 'duration_minutes', 'metascore',
                       'certificates', 'primary_genre']

CLUSTER_NUM_COLS = ['rating', 'metascore', 'duration', 'year']
CERT_MAP = {'R': 'R', 'PG-13': 'PG-13', 'PG': 'PG', 'G': 'G'}

CLUSTER_NAMES = {
    0: 'Giải trí đại chúng điểm thấp',
    1: 'Đại tác phẩm dài, điểm cao',
    2: 'Phim cổ điển (trước 1996)',
    3: 'Phim hiện đại chất lượng khá',
}
PROFILE_ANCHORS = {
    0: (5.92, 51.08, 102.23, 2009.99),
    1: (7.70, 74.22, 155.37, 1996.98),
    2: (6.78, 66.54, 108.20, 1974.03),
    3: (7.12, 68.21, 110.25, 2012.19),
}

_SOURCES: dict[str, str] = {}


def _genre_flag_value(selected: set[str], genre: str) -> int:
    return int(genre.casefold() in selected)


def _derived(values: dict) -> dict:
    title = str(values.get('title') or '').strip()
    genres = [g for g in (values.get('genres') or []) if g]
    cast_raw = str(values.get('star_cast') or '').strip()
    cast_tokens = [t for t in re.split(r'[,;|]', cast_raw) if t.strip()] if cast_raw else []
    selected = {g.casefold() for g in genres}
    flags = {col: _genre_flag_value(selected, genre)
             for col, genre in zip(GENRE_FLAG_COLS, GENRE_LIST)}
    return {
        'title': title,
        'title_length': len(title),
        'genres': genres,
        'primary_genre': genres[0] if genres else np.nan,
        'genre_count': len(genres),
        'cast_count': float(len(cast_tokens)) if cast_tokens else np.nan,
        'flags': flags,
    }


def build_classification_frame(values: dict) -> pd.DataFrame:
    d = _derived(values)
    row = {
        'year': values.get('year'),
        'duration_minutes': values.get('duration_minutes'),
        'metascore': values.get('metascore'),
        'genre_count': d['genre_count'],
        'cast_count': d['cast_count'],
        'title_length': d['title_length'],
        **d['flags'],
        'primary_genre': d['primary_genre'],
        'certificates': values.get('certificates') or np.nan,
    }
    frame = pd.DataFrame([row])
    return frame[CLASSIFICATION_FEATURES].astype({
        'year': 'float64', 'duration_minutes': 'float64', 'metascore': 'float64',
        'genre_count': 'float64', 'cast_count': 'float64', 'title_length': 'float64',
        **{c: 'float64' for c in GENRE_FLAG_COLS},
        'primary_genre': 'object', 'certificates': 'object',
    })


def build_regression_frame(values: dict) -> pd.DataFrame:
    d = _derived(values)
    row = {
        'year': values.get('year'),
        'duration_minutes': values.get('duration_minutes'),
        'metascore': values.get('metascore'),
        'certificates': values.get('certificates') or np.nan,
        'primary_genre': d['primary_genre'],
    }
    frame = pd.DataFrame([row])
    return frame[REGRESSION_FEATURES].astype({
        'year': 'float64', 'duration_minutes': 'float64', 'metascore': 'float64',
        'certificates': 'object', 'primary_genre': 'object',
    })


def build_clustering_frame(values: dict, bundle: dict) -> pd.DataFrame:
    d = _derived(values)
    keep = set(bundle['genre_keep'])
    cert_map = bundle['cert_map']
    genre = d['primary_genre']
    genre_grp = genre if isinstance(genre, str) and genre in keep else 'Other'
    cert = values.get('certificates')
    cert_grp = cert_map.get(cert, 'Other') if isinstance(cert, str) else 'Other'
    raw = pd.DataFrame([{
        'rating': values.get('imdb_rating'),
        'metascore': values.get('metascore'),
        'duration': values.get('duration_minutes'),
        'year': values.get('year'),
        'genre_grp': genre_grp,
        'cert_grp': cert_grp,
    }])
    num_cols = bundle['num_cols']
    cat_cols = [c for c in bundle['feature_names'] if c not in num_cols]
    x_num = pd.DataFrame(bundle['scaler'].transform(raw[num_cols]), columns=num_cols)
    x_cat = pd.get_dummies(raw[['genre_grp', 'cert_grp']], prefix=['g', 'c']).astype(float)
    x_cat = x_cat.reindex(columns=cat_cols, fill_value=0.0)
    x = pd.concat([x_num, x_cat], axis=1)
    return x[bundle['feature_names']]


def _load_pipeline(path: Path):
    obj = joblib.load(path)
    if isinstance(obj, dict):
        obj = obj.get('pipeline') or obj.get('model') or obj
    return obj


def _train_classification() -> Pipeline:
    df = pd.read_csv(DATA_DIR / 'imdb_for_classification.csv')
    genre_cols = [c for c in df.columns if c.startswith('genre_') and c != 'genre_count']
    features = NUM_CLASSIFICATION + genre_cols + CAT_CLASSIFICATION
    x = df[features].copy()
    y = df['rating_class'].copy()
    for col in CAT_CLASSIFICATION:
        x[col] = x[col].astype(object).where(x[col].notna(), np.nan)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    preprocessor = ColumnTransformer([
        ('numerical', Pipeline([
            ('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
            ('scaler', StandardScaler()),
        ]), NUM_CLASSIFICATION + genre_cols),
        ('categorical', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent', keep_empty_features=True)),
            ('onehot', OneHotEncoder(handle_unknown='ignore')),
        ]), CAT_CLASSIFICATION),
    ])
    pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('model', DecisionTreeClassifier(max_depth=5, min_samples_leaf=50,
                                         class_weight='balanced',
                                         random_state=RANDOM_STATE)),
    ])
    pipeline.fit(x_train, y_train)
    acc = accuracy_score(y_test, pipeline.predict(x_test))
    warnings.warn(f'Đã huấn luyện lại mô hình phân loại (test accuracy={acc:.4f}).')
    CLASSIFICATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, CLASSIFICATION_PATH)
    return pipeline


def _train_regression() -> Pipeline:
    df = pd.read_csv(DATA_DIR / 'imdb_cleaned.csv')
    prepared = df[['imdb_rating', 'year', 'duration_minutes', 'metascore',
                   'certificates', 'genres']].copy()
    prepared['imdb_rating'] = pd.to_numeric(prepared['imdb_rating'], errors='coerce')
    for col in ['year', 'duration_minutes', 'metascore']:
        prepared[col] = pd.to_numeric(prepared[col], errors='coerce')
    genres = prepared['genres'].astype('string').str.strip()
    prepared['primary_genre'] = genres.str.split(',').str[0].str.strip()
    for col in ['primary_genre', 'certificates']:
        prepared[col] = prepared[col].astype(object).replace('', np.nan)
    target_valid = (prepared['imdb_rating'].notna()
                    & np.isfinite(prepared['imdb_rating'])
                    & prepared['imdb_rating'].between(0, 10))
    prepared = prepared.loc[target_valid].reset_index(drop=True)
    x = prepared[REGRESSION_FEATURES].copy()
    y = prepared['imdb_rating'].copy()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, random_state=RANDOM_STATE)
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ])
    preprocessor = ColumnTransformer([
        ('numeric', numeric_transformer, ['year', 'duration_minutes', 'metascore']),
        ('categorical', categorical_transformer, ['certificates', 'primary_genre']),
    ])
    candidates = {
        'Linear Regression': Pipeline([('preprocessing', preprocessor),
                                       ('model', LinearRegression())]),
        'Decision Tree': Pipeline([
            ('preprocessing', preprocessor),
            ('model', DecisionTreeRegressor(random_state=RANDOM_STATE,
                                            max_depth=12, min_samples_leaf=5)),
        ]),
    }
    scores = {}
    for name, pipeline in candidates.items():
        pipeline.fit(x_train, y_train)
        pred = pipeline.predict(x_test)
        scores[name] = float(np.sqrt(mean_squared_error(y_test, pred)))
    best_name = min(scores, key=scores.get)
    best = candidates[best_name]
    r2 = float(r2_score(y_test, best.predict(x_test)))
    warnings.warn(f'Đã huấn luyện lại mô hình hồi quy: {best_name} '
                  f'(RMSE={scores[best_name]:.4f}, R2={r2:.4f}).')
    REGRESSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, REGRESSION_PATH)
    return best


def _clustering_raw_frame() -> pd.DataFrame:
    paths = sorted(RAW_DIR.glob('*.csv'))
    if not paths:
        raise FileNotFoundError(f'Không tìm thấy CSV dữ liệu thô trong {RAW_DIR}')
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True, sort=False)
    df = df.rename(columns={
        'IMDb Rating': 'rating', 'Duration (minutes)': 'duration',
        'Certificates': 'certificate', 'MetaScore': 'metascore',
        'Genre': 'genre', 'Title': 'title', 'Year': 'year',
    })
    for col in ['Poster-src', 'Second_Genre', 'Third_Genre']:
        if col in df.columns:
            df = df.drop(columns=col)
    for col in CLUSTER_NUM_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=CLUSTER_NUM_COLS)
    df = df.drop_duplicates(subset=['title', 'year']).reset_index(drop=True)
    return df


def _match_cluster_names(means: np.ndarray, cluster_ids: list[int]) -> dict[int, str]:
    anchor_ids = sorted(PROFILE_ANCHORS)
    anchors = np.array([PROFILE_ANCHORS[i] for i in anchor_ids], dtype=float)
    names_by_anchor = {idx: CLUSTER_NAMES[i] for idx, i in enumerate(anchor_ids)}
    scale = np.maximum(np.ptp(anchors, axis=0), 1.0)
    cost = (np.abs(means[:, None, :] - anchors[None, :, :]) / scale).sum(axis=2)
    rows, cols = linear_sum_assignment(cost)
    matched = {int(cluster_ids[r]): names_by_anchor[int(c)] for r, c in zip(rows, cols)}
    for cid in cluster_ids:
        matched.setdefault(int(cid), f'Nhóm {cid}')
    return matched


def _train_clustering() -> dict:
    df = _clustering_raw_frame()
    genre_share = df['genre'].value_counts(normalize=True)
    genre_keep = list(genre_share[genre_share >= 0.015].index)
    cert_map = dict(CERT_MAP)
    df['genre_grp'] = np.where(df['genre'].isin(genre_keep), df['genre'], 'Other')
    df['cert_grp'] = df['certificate'].map(cert_map).fillna('Other')
    x_cat = pd.get_dummies(df[['genre_grp', 'cert_grp']], prefix=['g', 'c']).astype(float)
    cat_cols = list(x_cat.columns)
    scaler = StandardScaler()
    x_num = pd.DataFrame(scaler.fit_transform(df[CLUSTER_NUM_COLS]),
                         columns=CLUSTER_NUM_COLS, index=df.index)
    x = pd.concat([x_num, x_cat], axis=1)
    feature_names = list(x.columns)
    pca_full = PCA(random_state=RANDOM_STATE).fit(x)
    cumulative = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.argmax(cumulative >= 0.90) + 1)
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE).fit(x)
    z = pca.transform(x)
    best_k, best_silhouette = 2, -1.0
    for k in range(2, 11):
        km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE).fit(z)
        score = float(silhouette_score(z, km.labels_))
        if score > best_silhouette:
            best_k, best_silhouette = k, score
    kmeans = KMeans(n_clusters=best_k, n_init=20, random_state=RANDOM_STATE).fit(z)
    labels = kmeans.labels_
    means = df.groupby(labels)[CLUSTER_NUM_COLS].mean().reindex(range(best_k)).to_numpy()
    cluster_names = _match_cluster_names(means, list(range(best_k)))
    profiles = []
    for cid in range(best_k):
        mask = labels == cid
        group = df.loc[mask]
        profiles.append({
            'cluster': cid,
            'name': cluster_names[cid],
            'n': int(mask.sum()),
            'share': float(mask.mean()),
            'rating': float(group['rating'].mean()),
            'metascore': float(group['metascore'].mean()),
            'duration': float(group['duration'].mean()),
            'year': float(group['year'].mean()),
        })
    points = pd.DataFrame({'pc1': z[:, 0], 'pc2': z[:, 1], 'cluster': labels})
    pca_bundle = {
        'pca': pca, 'scaler': scaler, 'feature_names': feature_names,
        'num_cols': list(CLUSTER_NUM_COLS), 'genre_keep': genre_keep,
        'cert_map': cert_map,
    }
    kmeans_bundle = {
        'kmeans': kmeans, 'cluster_names': cluster_names,
        'profiles': profiles, 'points': points,
    }
    CLUSTERING_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pca_bundle, PCA_PATH)
    joblib.dump(kmeans_bundle, KMEANS_PATH)
    warnings.warn(f'Đã huấn luyện lại PCA ({n_components} thành phần) và '
                  f'KMeans (k={best_k}, silhouette={best_silhouette:.3f}).')
    return {**pca_bundle, **kmeans_bundle}


@lru_cache(maxsize=None)
def load_classification_model() -> Pipeline:
    if CLASSIFICATION_PATH.exists():
        try:
            model = _load_pipeline(CLASSIFICATION_PATH)
            _SOURCES['classification'] = 'file đã lưu'
            return model
        except Exception as exc:
            warnings.warn(f'Không nạp được file phân loại: {exc}')
    _SOURCES['classification'] = 'huấn luyện lại (notebook 07)'
    return _train_classification()


@lru_cache(maxsize=None)
def load_regression_model() -> Pipeline:
    if REGRESSION_PATH.exists():
        try:
            model = _load_pipeline(REGRESSION_PATH)
            _SOURCES['regression'] = 'file đã lưu'
            return model
        except Exception as exc:
            warnings.warn(f'Không nạp được file hồi quy: {exc}')
    _SOURCES['regression'] = 'huấn luyện lại (notebook 06)'
    return _train_regression()


def _cluster_bundle_keys_ok(pca_bundle, kmeans_bundle) -> bool:
    need_pca = {'pca', 'scaler', 'feature_names', 'num_cols', 'genre_keep', 'cert_map'}
    need_km = {'kmeans', 'cluster_names'}
    return (need_pca <= set(pca_bundle) and need_km <= set(kmeans_bundle))


@lru_cache(maxsize=None)
def load_clustering_models() -> dict:
    if PCA_PATH.exists() and KMEANS_PATH.exists():
        try:
            pca_bundle = joblib.load(PCA_PATH)
            kmeans_bundle = joblib.load(KMEANS_PATH)
            if isinstance(pca_bundle, dict) and isinstance(kmeans_bundle, dict) \
                    and _cluster_bundle_keys_ok(pca_bundle, kmeans_bundle):
                _SOURCES['clustering'] = 'file đã lưu'
                return {**pca_bundle, **kmeans_bundle}
        except Exception as exc:
            warnings.warn(f'Không nạp được file phân cụm: {exc}')
    if all(path.exists() for path in NOTEBOOK_CLUSTER_FILES.values()):
        try:
            pre = joblib.load(NOTEBOOK_CLUSTER_FILES['preprocess'])
            pca = joblib.load(NOTEBOOK_CLUSTER_FILES['pca'])
            kmeans = joblib.load(NOTEBOOK_CLUSTER_FILES['kmeans'])
            bundle = {
                'pca': pca, 'kmeans': kmeans, 'scaler': pre['scaler'],
                'num_cols': list(pre['num_cols']),
                'feature_names': list(pre['feature_names']),
                'genre_keep': list(pre['genre_keep']), 'cert_map': dict(pre['cert_map']),
                'cluster_names': dict(CLUSTER_NAMES),
                'profiles': None, 'points': None,
            }
            _SOURCES['clustering'] = 'bộ file notebook 05'
            return bundle
        except Exception as exc:
            warnings.warn(f'Không nạp được bộ file notebook 05: {exc}')
    _SOURCES['clustering'] = 'huấn luyện lại (notebook 05)'
    return _train_clustering()


def predict_classification(values: dict) -> dict:
    model = load_classification_model()
    frame = build_classification_frame(values)
    label = str(model.predict(frame)[0])
    proba = model.predict_proba(frame)[0]
    classes = [str(c) for c in model.classes_]
    p_high = float(proba[classes.index(POS_LABEL)])
    return {
        'label': label,
        'label_vi': 'Điểm cao (High)' if label == POS_LABEL else 'Điểm thấp (Low)',
        'p_high': p_high,
        'p_low': 1.0 - p_high,
        'threshold_note': POS_LABEL,
        'features': frame,
    }


def predict_regression(values: dict) -> dict:
    model = load_regression_model()
    frame = build_regression_frame(values)
    rating = float(model.predict(frame)[0])
    return {
        'rating': rating,
        'rating_clipped': float(np.clip(rating, 0.0, 10.0)),
        'features': frame,
    }


def predict_clustering(values: dict) -> dict:
    bundle = load_clustering_models()
    frame = build_clustering_frame(values, bundle)
    z = bundle['pca'].transform(frame)
    cluster_id = int(bundle['kmeans'].predict(z)[0])
    cluster_name = bundle['cluster_names'].get(cluster_id, f'Nhóm {cluster_id}')
    profile = next((p for p in (bundle.get('profiles') or [])
                    if p['cluster'] == cluster_id), None)
    return {
        'cluster_id': cluster_id,
        'cluster_name': cluster_name,
        'profile': profile,
        'profiles': bundle.get('profiles'),
        'points': bundle.get('points'),
        'pc1': float(z[0][0]),
        'pc2': float(z[0][1]),
        'n_components': int(bundle['pca'].n_components_),
        'summary': {
            'rating': float(values.get('imdb_rating')),
            'metascore': float(values.get('metascore')),
            'duration': float(values.get('duration_minutes')),
            'year': int(values.get('year')),
        },
        'features': frame,
    }


def list_certificate_options() -> list[str]:
    featured = DATA_DIR / 'imdb_featured.csv'
    if featured.exists():
        df = pd.read_csv(featured, usecols=['certificates'])
        options = sorted(str(v) for v in df['certificates'].dropna().unique())
        if options:
            return options
    return ['PG-13', 'PG', 'R', 'G', 'Not Rated', 'Unrated']


def list_genre_options() -> list[str]:
    return list(GENRE_LIST)


def model_status() -> list[dict]:
    specs = [
        ('Phân loại (Classification)', load_classification_model,
         [CLASSIFICATION_PATH], 'classification'),
        ('Hồi quy (Regression)', load_regression_model,
         [REGRESSION_PATH], 'regression'),
        ('Phân cụm (PCA + K-Means)', load_clustering_models,
         [PCA_PATH, KMEANS_PATH], 'clustering'),
    ]
    rows = []
    for name, loader, paths, key in specs:
        exists = all(p.exists() for p in paths)
        size_kb = sum(p.stat().st_size for p in paths if p.exists()) / 1024
        row = {
            'name': name,
            'paths': ' + '.join(str(p.relative_to(ROOT)) for p in paths),
            'exists': exists,
            'size_kb': round(size_kb, 1),
            'source': None,
            'kind': None,
            'ok': True,
            'error': None,
        }
        try:
            model = loader()
            row['source'] = _SOURCES.get(key)
            if isinstance(model, dict):
                row['kind'] = (f"PCA({model['pca'].n_components_} thành phần) + "
                               f"KMeans(k={model['kmeans'].n_clusters})")
            else:
                row['kind'] = type(model).__name__
        except Exception as exc:
            row['ok'] = False
            row['error'] = str(exc)
        rows.append(row)
    return rows
