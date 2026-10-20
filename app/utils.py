"""Kiểm tra dữ liệu nhập và định dạng kết quả hiển thị."""

from __future__ import annotations

import datetime as _dt
import re

import streamlit as st

YEAR_MIN = 1888
YEAR_MAX = _dt.date.today().year + 5

DEMO_SAMPLES = {
    'classification': {
        'title': 'Inception',
        'year': 2010,
        'duration_minutes': 148,
        'metascore': 74,
        'genres': ['Action', 'Sci-Fi', 'Thriller'],
        'certificates': 'PG-13',
        'star_cast': 'Leonardo DiCaprio, Joseph Gordon-Levitt, Elliot Page',
    },
    'regression': {
        'year': 2010,
        'duration_minutes': 148,
        'metascore': 74,
        'certificates': 'PG-13',
        'genres': ['Action', 'Sci-Fi', 'Thriller'],
    },
    'clustering': {
        'title': 'Inception',
        'year': 2010,
        'duration_minutes': 148,
        'metascore': 74,
        'imdb_rating': 8.8,
        'genres': ['Action', 'Sci-Fi', 'Thriller'],
        'certificates': 'PG-13',
    },
}


def seed_demo(page: str, demo: bool) -> None:
    """Tiền điền dữ liệu mẫu khi chạy ở chế độ demo (?demo=1)."""
    if not demo:
        return
    for field, value in DEMO_SAMPLES.get(page, {}).items():
        key = f'{page}_{field}'
        if key not in st.session_state:
            st.session_state[key] = value


def validate_movie_input(values: dict, *, need_title: bool = False,
                         need_rating: bool = False, need_cast: bool = False) -> list[str]:
    """Trả về danh sách lỗi (tiếng Việt) của dữ liệu người dùng nhập."""
    errors: list[str] = []

    if need_title and not str(values.get('title') or '').strip():
        errors.append('Vui lòng nhập tên phim.')

    year = values.get('year')
    if year is None or not YEAR_MIN <= int(year) <= YEAR_MAX:
        errors.append(f'Năm phát hành phải nằm trong khoảng {YEAR_MIN}–{YEAR_MAX}.')

    duration = values.get('duration_minutes')
    if duration is None or not 1 <= float(duration) <= 600:
        errors.append('Thời lượng phải nằm trong khoảng 1–600 phút.')

    metascore = values.get('metascore')
    if metascore is not None and not 0 <= float(metascore) <= 100:
        errors.append('MetaScore phải nằm trong khoảng 0–100.')

    if need_rating:
        rating = values.get('imdb_rating')
        if rating is None or not 0 <= float(rating) <= 10:
            errors.append('Điểm IMDb phải nằm trong khoảng 0–10.')

    if not values.get('genres'):
        errors.append('Vui lòng chọn ít nhất một thể loại.')

    if not values.get('certificates'):
        errors.append('Vui lòng chọn chứng nhận phim (certificate).')

    if need_cast:
        cast = str(values.get('star_cast') or '').strip()
        tokens = [t for t in re.split(r'[,;|]', cast) if t.strip()] if cast else []
        if cast and not tokens:
            errors.append('Dàn diễn viên không hợp lệ — hãy dùng tên phân cách bằng dấu phẩy.')

    return errors


def format_rating(value: float) -> str:
    return f'{float(value):.2f} / 10'


def format_pct(probability: float) -> str:
    return f'{float(probability) * 100:.1f}%'


def rating_band(value: float) -> str:
    value = float(value)
    if value >= 7.0:
        return 'Cao (≥ 7.0)'
    if value >= 5.0:
        return 'Trung bình (5.0 – 6.9)'
    return 'Thấp (< 5.0)'
