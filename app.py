"""Giao diện chính của ứng dụng Streamlit — IMDb Analytics & AI.

Chạy:  streamlit run app.py
Demo:  streamlit run app.py rồi mở ?page=classification&demo=1 (tương tự các trang khác).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from app import components, utils
from src import model_prediction as mp

st.set_page_config(page_title='IMDb Analytics & AI', page_icon='🎬', layout='wide')

PAGE_KEYS = ['home', 'classification', 'regression', 'clustering']
PAGE_LABELS = {
    'home': '🏠 Trang chủ',
    'classification': '🌲 Phân loại: điểm cao hay thấp?',
    'regression': '📈 Hồi quy: dự đoán điểm IMDb',
    'clustering': '🔍 Phân cụm: PCA + K-Means',
}


def _query_page() -> str:
    page = st.query_params.get('page', 'home')
    return page if page in PAGE_KEYS else 'home'


def _demo_mode() -> bool:
    return st.query_params.get('demo', '') == '1'


def render_home() -> None:
    components.page_header(
        'Tổng quan dự án — Phân tích dữ liệu & AI',
        'Dự đoán điểm IMDb, phân loại điểm cao/thấp và phân cụm phim · dữ liệu Kaggle IMDb 2024')
    components.model_status_cards()
    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric('Phân loại — Accuracy', '72.85%',
              help='Decision Tree trên tập test 744 phim (notebook 07)')
    c2.metric('Hồi quy — RMSE', '0.694',
              help='Linear Regression, R² = 0.359 (notebook 06)')
    c3.metric('Phân cụm — Silhouette', '0.192',
              help='KMeans k=4 trên không gian PCA 9 chiều (notebook 05)')
    st.divider()
    components.dataset_overview()
    with st.expander('🔎 Luồng xử lý dữ liệu & mô hình'):
        st.markdown(
            '1. **Nạp & làm sạch** (notebook 01–02): gộp 3 file CSV thô (4.573 dòng), '
            'chuẩn hoá cột, xử lý giá trị sai, khử trùng lặp theo (tên phim, năm) → **3.716 phim**.\n\n'
            '2. **EDA** (notebook 03): thống kê thiếu, phân bố, IQR.\n\n'
            '3. **Feature engineering** (notebook 04): `genre_count`, `cast_count`, '
            '`title_length`, `primary_genre`, 21 cột nhị phân thể loại; '
            'nhãn `rating_class` = High nếu điểm IMDb ≥ 7.0.\n\n'
            '4. **Mô hình:**\n'
            '   - **Classification** (07): Decision Tree (depth=5, leaf=50) trong pipeline đầy '
            'dụ 29 đặc trưng → `models/classification/classification_pipeline.pkl`.\n'
            '   - **Regression** (06): Linear Regression dự đoán `imdb_rating` từ năm, thời '
            'lượng, MetaScore, certificates, primary_genre → `models/regression/regression_pipeline.pkl`.\n'
            '   - **Clustering** (05): StandardScaler → PCA (9 chiều, 90.7% phương sai) → '
            'KMeans k=4 → `models/clustering/pca_model.pkl` + `kmeans_model.pkl`.\n\n'
            '5. **Ứng dụng (thành viên 5):** `src/model_prediction.py` nạp mô hình và dựng '
            'đặc trưng đầu vào; `app/utils.py`, `app/components.py`, `app.py` dựng giao diện Streamlit.')
    with st.expander('▶️ Cách cài đặt & chạy ứng dụng'):
        st.code('pip install -r requirements.txt\nstreamlit run app.py', language='bash')
        st.markdown(
            'Chế độ demo (tiền điền dữ liệu mẫu và tự động dự đoán): thêm `?page=classification&demo=1` '
            'hoặc `?page=regression&demo=1`, `?page=clustering&demo=1` vào địa chỉ app.')


def render_classification() -> None:
    utils.seed_demo('classification', _demo_mode())
    components.page_header(
        'Phân loại: phim điểm cao hay thấp?',
        'Nhãn **High** = điểm IMDb ≥ 7.0 · Mô hình: Decision Tree trong pipeline đầy đủ '
        '(29 đặc trưng, không dùng điểm IMDb để tránh rò rỉ nhãn)')
    left, right = st.columns([1.05, 1], gap='large')
    with left:
        st.markdown('##### 📝 Nhập thông tin phim')
        title = st.text_input('Tên phim', max_chars=120, key='classification_title')
        c1, c2, c3 = st.columns(3)
        year = c1.number_input('Năm phát hành', min_value=utils.YEAR_MIN,
                               max_value=utils.YEAR_MAX, value=2015, step=1,
                               key='classification_year')
        duration = c2.number_input('Thời lượng (phút)', min_value=1, max_value=600,
                                   value=110, step=1, key='classification_duration_minutes')
        metascore = c3.number_input('MetaScore', min_value=0, max_value=100,
                                    value=65, step=1, key='classification_metascore')
        genres = st.multiselect(
            'Thể loại (chọn 1–3 thể loại; thể loại được chọn đầu tiên = thể loại chính)',
            mp.list_genre_options(), key='classification_genres')
        certificates = st.selectbox('Chứng nhận (certificate)',
                                    mp.list_certificate_options(),
                                    key='classification_certificates')
        star_cast = st.text_input(
            'Dàn diễn viên (phân cách bằng dấu phẩy)',
            key='classification_star_cast',
            placeholder='VD: Leonardo DiCaprio, Joseph Gordon-Levitt')
        submitted = st.button('🔍 Dự đoán', type='primary',
                              use_container_width=True, key='classification_submit')
    with right:
        result_slot = st.empty()
        st.divider()
        components.show_notebook_metrics('classification')
    values = {
        'title': title, 'year': int(year), 'duration_minutes': float(duration),
        'metascore': float(metascore), 'genres': genres, 'certificates': certificates,
        'star_cast': star_cast,
    }
    if submitted or _demo_mode():
        errors = utils.validate_movie_input(values, need_title=True, need_cast=True)
        if errors:
            for error in errors:
                st.error(error)
        else:
            with result_slot.container():
                with st.spinner('Đang nạp mô hình và dự đoán...'):
                    result = mp.predict_classification(values)
                components.show_classification_result(result)


def render_regression() -> None:
    utils.seed_demo('regression', _demo_mode())
    components.page_header(
        'Hồi quy: dự đoán điểm IMDb',
        'Mô hình: Linear Regression trong pipeline (năm, thời lượng, MetaScore, certificates, '
        'primary_genre) · RMSE 0.694, R² 0.359 trên tập test')
    left, right = st.columns([1.05, 1], gap='large')
    with left:
        st.markdown('##### 📝 Nhập thông tin phim')
        c1, c2, c3 = st.columns(3)
        year = c1.number_input('Năm phát hành', min_value=utils.YEAR_MIN,
                               max_value=utils.YEAR_MAX, value=2015, step=1,
                               key='regression_year')
        duration = c2.number_input('Thời lượng (phút)', min_value=1, max_value=600,
                                   value=110, step=1, key='regression_duration_minutes')
        metascore = c3.number_input('MetaScore', min_value=0, max_value=100,
                                    value=65, step=1, key='regression_metascore')
        certificates = st.selectbox('Chứng nhận (certificate)',
                                    mp.list_certificate_options(),
                                    key='regression_certificates')
        genres = st.multiselect(
            'Thể loại (chọn 1 thể loại chính, có thể chọn thêm)',
            mp.list_genre_options(), key='regression_genres')
        submitted = st.button('🔍 Dự đoán', type='primary',
                              use_container_width=True, key='regression_submit')
    with right:
        result_slot = st.empty()
        st.divider()
        components.show_notebook_metrics('regression')
    values = {
        'year': int(year), 'duration_minutes': float(duration),
        'metascore': float(metascore), 'certificates': certificates, 'genres': genres,
    }
    if submitted or _demo_mode():
        errors = utils.validate_movie_input(values)
        if errors:
            for error in errors:
                st.error(error)
        else:
            with result_slot.container():
                with st.spinner('Đang nạp mô hình và dự đoán...'):
                    result = mp.predict_regression(values)
                dataset_mean = float(
                    components.load_processed('imdb_for_regression.csv')['imdb_rating'].mean())
                components.show_regression_result(result, dataset_mean)


def render_clustering() -> None:
    utils.seed_demo('clustering', _demo_mode())
    components.page_header(
        'Phân cụm phim — PCA + K-Means',
        'StandardScaler → PCA 9 chiều (giữ 90.7% phương sai) → KMeans k=4 · '
        'mỗi cụm mang một ý nghĩa diễn giải riêng')
    left, right = st.columns([1.05, 1], gap='large')
    with left:
        st.markdown('##### 📝 Nhập thông tin phim')
        title = st.text_input('Tên phim', max_chars=120, key='clustering_title')
        c1, c2, c3 = st.columns(3)
        year = c1.number_input('Năm phát hành', min_value=utils.YEAR_MIN,
                               max_value=utils.YEAR_MAX, value=2015, step=1,
                               key='clustering_year')
        duration = c2.number_input('Thời lượng (phút)', min_value=1, max_value=600,
                                   value=110, step=1, key='clustering_duration_minutes')
        metascore = c3.number_input('MetaScore', min_value=0, max_value=100,
                                    value=65, step=1, key='clustering_metascore')
        imdb_rating = st.slider('Điểm IMDb của phim', min_value=0.0, max_value=10.0,
                                value=6.5, step=0.1, key='clustering_imdb_rating',
                                help='Phân cụm là bài toán không giám sát — điểm IMDb là '
                                     'một trong các đặc trưng của phim.')
        genres = st.multiselect(
            'Thể loại (chọn 1–3 thể loại; thể loại được chọn đầu tiên = thể loại chính)',
            mp.list_genre_options(), key='clustering_genres')
        certificates = st.selectbox('Chứng nhận (certificate)',
                                    mp.list_certificate_options(),
                                    key='clustering_certificates')
        submitted = st.button('🔍 Phân cụm', type='primary',
                              use_container_width=True, key='clustering_submit')
    with right:
        result_slot = st.empty()
        st.divider()
        components.show_notebook_metrics('clustering')
    values = {
        'title': title, 'year': int(year), 'duration_minutes': float(duration),
        'metascore': float(metascore), 'imdb_rating': float(imdb_rating),
        'genres': genres, 'certificates': certificates,
    }
    if submitted or _demo_mode():
        errors = utils.validate_movie_input(values, need_title=True, need_rating=True)
        if errors:
            for error in errors:
                st.error(error)
        else:
            with result_slot.container():
                with st.spinner('Đang nạp mô hình và phân cụm...'):
                    result = mp.predict_clustering(values)
                components.show_clustering_result(result)


def main() -> None:
    with st.sidebar:
        st.title('🎬 IMDb Analytics & AI')
        st.caption('Nhập môn Phân tích Dữ liệu & AI — Final Project')
        st.divider()
        page = st.radio('Chức năng', PAGE_KEYS,
                        index=PAGE_KEYS.index(_query_page()),
                        format_func=lambda key: PAGE_LABELS[key], key='nav_page')       
        st.divider()
        st.caption('APP ĐƯỢC HOÀN THIỆN VÀ PHÁT TRIỂN TỪ NHÓM 7 🩵💙💚💛🧡🩷❤️')
    renderers = {
        'home': render_home,
        'classification': render_classification,
        'regression': render_regression,
        'clustering': render_clustering,
    }
    renderers[page]()


main()
