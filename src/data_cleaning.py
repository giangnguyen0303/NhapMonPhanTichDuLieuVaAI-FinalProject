"""Tiền xử lý IMDb, lưu NaN để tránh rò rỉ dữ liệu (leakage) giữa train/test."""
import re
import unicodedata
import numpy as np
import pandas as pd

# Các alias có thể gặp khi người phát hành đổi cách đặt tên cột.
ALIASES = {
    'title': ['Title', 'Movie Name', 'Movie Title', 'Name', 'primaryTitle'],
    'imdb_rating': ['IMDb Rating', 'IMDB Rating', 'IMDb_Rating', 'Rating', 'IMDB_Rating', 'averageRating'],
    'year': ['Year', 'Release Year', 'Released Year', 'startYear'],
    'duration_minutes': ['Duration (minutes)', 'Duration', 'Runtime', 'Runtime (Minutes)', 'runtimeMinutes'],
    'metascore': ['MetaScore', 'Metascore', 'Meta Score'],
    'certificates': ['Certificates', 'Certificate', 'Age Rating', 'Certification'],
    'director': ['Director', 'Directors'],
    'star_cast': ['Star Cast', 'StarCast', 'Cast', 'Actors', 'Stars'],
    'poster_url': ['Poster-src', 'Poster URL', 'Poster', 'Poster-src URL'],
    'genres': ['Genres'],
    'genre_1': ['Genre', 'Genre 1', 'First_Genre', 'First Genre'],
    'genre_2': ['Second_Genre', 'Second Genre', 'Genre 2'],
    'genre_3': ['Third_Genre', 'Third Genre', 'Genre 3'],
}


def _key(name):
    name = unicodedata.normalize('NFKD', str(name))
    return re.sub(r'[^a-z0-9]', '', name.lower())


def normalize_columns(df):
    """Đổi các cột theo tên chuẩn; báo lỗi khi thiếu cột bắt buộc."""
    found = {_key(c): c for c in df.columns}
    renames = {}
    for standard, aliases in ALIASES.items():
        possibilities = [standard] + aliases
        for alias in possibilities:
            existing = found.get(_key(alias))
            if existing is not None and (existing == standard or standard not in df.columns):
                renames[existing] = standard
                break
    df = df.rename(columns=renames).copy()
    for needed in ['title', 'imdb_rating', 'year', 'duration_minutes', 'metascore',
                   'certificates', 'director', 'star_cast', 'poster_url',
                   'genres', 'genre_1', 'genre_2', 'genre_3']:
        if needed not in df:
            df[needed] = np.nan
    if df['title'].isna().all():
        raise ValueError(f'Không nhận ra cột tên phim. Các cột hiện có: {list(df.columns)}')
    return df


def parse_number(values):
    """Trích xuất một giá trị số từ các chuỗi như '8.5/10', '142 min', '1,200'."""
    s = values.astype('string').str.replace(',', '', regex=False)
    extracted = s.str.extract(r'([-+]?\d+(?:\.\d+)?)', expand=False)
    return pd.to_numeric(extracted, errors='coerce')


def parse_year(values):
    s = values.astype('string')
    four_digit = s.str.extract(r'\b((?:18|19|20|21)\d{2})\b', expand=False)
    return pd.to_numeric(four_digit, errors='coerce').astype('Int64')


def norm_genres(row):
    result = []
    for col in ['genres', 'genre_1', 'genre_2', 'genre_3']:
        val = row.get(col)
        if pd.isna(val):
            continue
        for entry in re.split(r'[,|;/]', str(val)):
            token = re.sub(r'\s+', ' ', entry).strip().title()
            if token and token.lower() not in {'nan','none','unknown','n/a'} and token not in result:
                result.append(token)
    return ', '.join(result) if result else pd.NA


def clean_data(raw):
    """Làm sạch có kiểm soát; kèm bản ghi chất lượng trước/sau."""
    df = normalize_columns(raw)
    before_count = len(df)
    raw_missing = df.isna().sum().sort_values(ascending=False)
    for c in ['title', 'certificates', 'director', 'star_cast', 'poster_url']:
        df[c] = df[c].astype('string').str.replace(r'\s+', ' ', regex=True).str.strip()
        df[c] = df[c].replace({'': pd.NA, 'nan': pd.NA, 'None': pd.NA, 'N/A': pd.NA, '-': pd.NA})
    df['imdb_rating'] = parse_number(df['imdb_rating'])
    df['metascore'] = parse_number(df['metascore'])
    df['duration_minutes'] = parse_number(df['duration_minutes'])
    df['year'] = parse_year(df['year'])
    for c, lo, hi in [('imdb_rating', 0, 10), ('metascore', 0, 100), ('duration_minutes', 1, 10000)]:
        df.loc[~df[c].between(lo, hi), c] = np.nan
    # Year > current year: có thể là phim sắp ra mắt; không mặc định xóa, chỉ bỏ năm vô lý.
    df.loc[~df['year'].between(1888, 2100), 'year'] = pd.NA
    df['genres'] = df.apply(norm_genres, axis=1)
    missing_title = df['title'].isna()
    df = df.loc[~missing_title].copy()
    # Khử trùng theo tên + năm, riêng thiếu năm chỉ trùng với thiếu năm.
    # Để bảo toàn hàng có nhiều thông tin hơn.
    quality_cols = ['imdb_rating', 'year', 'duration_minutes', 'metascore',
                    'genres', 'director', 'star_cast', 'certificates']
    df['_quality'] = df[quality_cols].notna().sum(axis=1)
    df['_title_key'] = df['title'].str.casefold().str.replace(r'[^\w]+', ' ', regex=True).str.strip()
    df = df.sort_values('_quality', ascending=False, kind='stable')
    dup = df.duplicated(subset=['_title_key', 'year'], keep='first')
    duplicate_count = int(dup.sum())
    df = df.loc[~dup].drop(columns=['_quality', '_title_key']).reset_index(drop=True)
    for c in ['imdb_rating', 'duration_minutes', 'metascore', 'year', 'genres']:
        df[f'{c}_missing'] = df[c].isna().astype('int8')
    report = {
        'raw_rows': int(before_count),
        'rows_missing_title_dropped': int(missing_title.sum()),
        'duplicate_title_year_dropped': duplicate_count,
        'clean_rows': int(len(df)),
        'original_missing_by_column': {str(k): int(v) for k, v in raw_missing.items()},
        'clean_missing_by_column': {str(k): int(v) for k, v in df.isna().sum().items()},
        'notes': [
            'Chưa điền missing numeric bằng trung bình/trung vị vì phải fit imputer trên train.',
            'Chưa loại IQR outliers tự động vì có thể là phim hợp lệ.',
            'Chưa kết luận MCAR/MAR/MNAR từ dữ liệu quan sát.'
        ],
    }
    return df, report
