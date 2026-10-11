"""Tạo đặc trưng trước khi nhóm xây mô hình; không fit encoder/scaler trên toàn tập."""
import numpy as np
import pandas as pd


def create_features(cleaned):
    df = cleaned.copy()
    df['release_decade'] = ((df['year'] // 10) * 10).astype('Int64')
    df['title_length'] = df['title'].astype('string').str.len().astype('Int64')
    df['genre_count'] = df['genres'].astype('string').str.split(',').apply(
        lambda x: len([i for i in x if i.strip()]) if isinstance(x, list) else np.nan
    )
    df['primary_genre'] = df['genres'].astype('string').str.split(',').str[0].str.strip()
    df['cast_count'] = df['star_cast'].astype('string').str.split(r',|;|\|').apply(
        lambda x: len([i for i in x if i.strip()]) if isinstance(x, list) else np.nan
    )
    df['duration_group'] = pd.cut(df['duration_minutes'], bins=[0, 90, 120, np.inf],
                                  labels=['Ngan (<=90p)', 'Vua (91-120p)', 'Dai (>120p)'])
    # Nhãn định trước, không xác định theo phân vị của toàn dataset.
    df['rating_class'] = pd.Series(pd.NA, index=df.index, dtype='string')
    has_rating = df['imdb_rating'].notna()
    df.loc[has_rating, 'rating_class'] = np.where(df.loc[has_rating, 'imdb_rating'] >= 7.0,
                                                 'Cao (>=7)', 'Thap (<7)')
    return df


def model_datasets(feature_df):
    df = feature_df.copy()
    # Để nhóm 3/4 tự fit imputer, scaler, encoder trên tập train.
    reg = df.loc[df['imdb_rating'].notna()].drop(columns=['rating_class']).copy()
    clf = df.loc[df['rating_class'].notna()].copy()
    # Clustering: chỉ giữ metadata + numerical; các cột missing giữ NaN.
    cluster_cols = ['title', 'year', 'duration_minutes', 'metascore',
                    'imdb_rating', 'genre_count', 'cast_count', 'primary_genre']
    clu = df[cluster_cols].copy()
    return {'imdb_cleaned': df, 'imdb_for_regression': reg,
            'imdb_for_classification': clf, 'imdb_for_clustering': clu}
