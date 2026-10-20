"""Thành phần giao diện: biểu đồ, bảng chỉ số và khối hiển thị kết quả."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app import utils
from src import model_prediction as mp

DATA_DIR = mp.DATA_DIR

CLUSTER_DESC = {
    'Giải trí đại chúng điểm thấp':
        'Điểm IMDb trung bình ~5.9, Metascore ~51, thời lượng ~102 phút — phim đại chúng, '
        'giải trí, chất lượng trung bình thấp.',
    'Đại tác phẩm dài, điểm cao':
        'Điểm IMDb trung bình ~7.7, Metascore ~74, thời lượng ~155 phút — phim dài, '
        'được giới chuyên môn đánh giá cao (đại tác phẩm).',
    'Phim cổ điển (trước 1996)':
        'Năm phát hành trung bình ~1974, điểm IMDb ~6.8 — phim cổ điển trước thập niên 1996.',
    'Phim hiện đại chất lượng khá':
        'Năm phát hành trung bình ~2012, điểm IMDb ~7.1, thời lượng ~110 phút — '
        'phim hiện đại, chất lượng khá.',
}

CLF_METRICS = pd.DataFrame([
    {'Mô hình': 'Baseline (most_frequent)', 'Accuracy': 0.5618, 'F1-macro': 0.3597,
     'ROC-AUC': 0.5000, 'Ghi chú': 'Đoạn tham chiếu'},
    {'Mô hình': 'Logistic Regression', 'Accuracy': 0.7164, 'F1-macro': 0.7090,
     'ROC-AUC': 0.7723, 'Ghi chú': 'C=1, class_weight=balanced'},
    {'Mô hình': 'Decision Tree', 'Accuracy': 0.7285, 'F1-macro': 0.7197,
     'ROC-AUC': 0.7802, 'Ghi chú': 'depth=5, leaf=50 — mô hình được lưu'},
])

REG_METRICS = pd.DataFrame([
    {'Mô hình': 'Linear Regression', 'MSE': 0.4810, 'RMSE': 0.6935, 'R²': 0.3592,
     'Ghi chú': 'Tốt nhất — mô hình được lưu'},
    {'Mô hình': 'Decision Tree', 'MSE': 0.5089, 'RMSE': 0.7133, 'R²': 0.3221,
     'Ghi chú': 'max_depth=12, min_samples_leaf=5'},
])

CLU_METRICS = pd.DataFrame([
    {'Chỉ số': 'Số chiều đầu vào', 'Có PCA': '9', 'Không PCA': '20'},
    {'Chỉ số': 'Phương sai giữ lại', 'Có PCA': '90.7%', 'Không PCA': '100%'},
    {'Chỉ số': 'Silhouette', 'Có PCA': '0.192', 'Không PCA': '0.168'},
    {'Chỉ số': 'Davies-Bouldin (↓)', 'Có PCA': '1.548', 'Không PCA': '1.692'},
    {'Chỉ số': 'Calinski-Harabasz (↑)', 'Có PCA': '779.4', 'Không PCA': '668.5'},
])


@st.cache_data
def load_processed(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / name)


def page_header(title: str, subtitle: str | None = None) -> None:
    st.title(title)
    if subtitle:
        st.caption(subtitle)


def model_status_cards() -> None:
    st.markdown('##### Trạng thái nạp mô hình')
    rows = mp.model_status()
    cols = st.columns(len(rows))
    for col, row in zip(cols, rows):
        with col:
            if row['ok']:
                st.success(row['name'])
                st.caption(f"`{row['paths']}`")
                st.caption(f"{row['kind']} · {row['size_kb']} KB · {row['source']}")
            else:
                st.error(row['name'])
                st.caption(row['error'])


def show_notebook_metrics(kind: str) -> None:
    st.markdown('**Chỉ số trên tập test (theo notebook nhóm)**')
    if kind == 'classification':
        st.dataframe(CLF_METRICS, use_container_width=True, hide_index=True)
        st.caption('Tập test 744 phim, stratify, random_state=42 — notebook 07.')
    elif kind == 'regression':
        st.dataframe(REG_METRICS, use_container_width=True, hide_index=True)
        st.caption('Tập test 744 phim, random_state=42 — notebook 06.')
    else:
        st.dataframe(CLU_METRICS, use_container_width=True, hide_index=True)
        st.caption('K=4 theo silhouette; PCA giữ 90.7% phương sai — notebook 05.')


def _features_expander(frame: pd.DataFrame, columns_title: str) -> None:
    with st.expander(f'🧮 Đặc trưng đầu vào ({columns_title})'):
        st.dataframe(frame.T, use_container_width=True)


def show_classification_result(result: dict) -> None:
    st.markdown('#### 🎯 Kết quả dự đoán')
    if result['label'] == 'High':
        st.success(f"**{result['label_vi']}** — phim được dự đoán có điểm IMDb ≥ 7.0")
    else:
        st.warning(f"**{result['label_vi']}** — phim được dự đoán có điểm IMDb < 7.0")
    c1, c2 = st.columns(2)
    c1.metric('Xác suất High', utils.format_pct(result['p_high']))
    c2.metric('Xác suất Low', utils.format_pct(result['p_low']))
    fig = go.Figure(go.Bar(
        x=['High (≥ 7.0)', 'Low (< 7.0)'],
        y=[result['p_high'], result['p_low']],
        marker_color=['#2e7d32', '#c62828'],
        text=[utils.format_pct(result['p_high']), utils.format_pct(result['p_low'])],
        textposition='outside',
    ))
    fig.update_layout(
        height=280, showlegend=False,
        yaxis={'tickformat': '.0%', 'range': [0, 1.08]},
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.info('Nhãn **High** nghĩa là điểm IMDb thực tế ≥ 7.0. Mô hình không dùng điểm IMDb '
            'làm đầu vào để tránh rò rỉ nhãn (leakage).')
    _features_expander(result['features'], '29 cột của pipeline phân loại')


def show_regression_result(result: dict, dataset_mean: float) -> None:
    rating = result['rating_clipped']
    st.markdown('#### 🎯 Kết quả dự đoán')
    st.metric('Điểm IMDb dự đoán', utils.format_rating(rating),
              delta=f'{rating - dataset_mean:+.2f} so với trung bình dữ liệu ({dataset_mean:.2f})')
    gauge = go.Figure(go.Indicator(
        mode='gauge+number', value=rating,
        number={'suffix': ' / 10', 'font': {'size': 36}},
        gauge={
            'axis': {'range': [0, 10], 'tickwidth': 1},
            'bar': {'color': '#1f77b4', 'thickness': 0.25},
            'steps': [
                {'range': [0, 5], 'color': '#f8d7da'},
                {'range': [5, 7], 'color': '#fff3cd'},
                {'range': [7, 10], 'color': '#d4edda'},
            ],
            'threshold': {'line': {'color': 'red', 'width': 3},
                          'thickness': 0.8, 'value': 7.0},
        },
    ))
    gauge.update_layout(height=280, margin=dict(l=20, r=20, t=10, b=10))
    st.plotly_chart(gauge, use_container_width=True)
    st.caption(f'Xếp loại theo khoảng điểm: **{utils.rating_band(rating)}** · '
               'vạch đỏ = ngưỡng 7.0 của nhãn High.')
    reg_df = load_processed('imdb_for_regression.csv')
    hist = px.histogram(reg_df, x='imdb_rating', nbins=40,
                        title='Phân bố điểm IMDb thực tế trong dữ liệu',
                        color_discrete_sequence=['#90caf9'])
    hist.add_vline(x=rating, line_color='#d32f2f', line_width=2,
                   annotation_text=f'Dự đoán: {rating:.2f}',
                   annotation_position='top right')
    hist.add_vline(x=7.0, line_color='#2e7d32', line_dash='dot', line_width=2,
                   annotation_text='Ngưỡng 7.0', annotation_position='top left')
    hist.update_layout(height=320, showlegend=False,
                       xaxis_title='Điểm IMDb', yaxis_title='Số phim',
                       margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(hist, use_container_width=True)
    st.info('MetaScore là một đầu vào của mô hình, nên đây là ước tính điểm IMDb dựa trên '
            'dữ liệu phim — không phải dự báo trước ngày ra mắt.')
    _features_expander(result['features'], '5 cột của pipeline hồi quy')


def show_clustering_result(result: dict) -> None:
    st.markdown('#### 🎯 Kết quả phân cụm')
    st.success(f"**Cụm {result['cluster_id']}: {result['cluster_name']}**")
    st.write(CLUSTER_DESC.get(result['cluster_name'], ''))
    c1, c2, c3 = st.columns(3)
    c1.metric('PC1', f"{result['pc1']:.2f}")
    c2.metric('PC2', f"{result['pc2']:.2f}")
    c3.metric('Số chiều PCA', f"{result['n_components']}",
              help='PCA giữ lại khoảng 90.7% phương sai ban đầu.')
    profiles = result.get('profiles')
    if profiles:
        pdf = pd.DataFrame(profiles).copy()
        pdf['Tỷ lệ'] = (pdf['share'] * 100).round(1).astype(str) + '%'
        pdf['Cụm'] = pdf['cluster'].astype(str) + ': ' + pdf['name']
        for col in ['rating', 'metascore', 'duration', 'year']:
            pdf[col] = pdf[col].round(2)
        pdf = pdf.rename(columns={
            'n': 'Số phim', 'rating': 'Điểm IMDb TB', 'metascore': 'Metascore TB',
            'duration': 'Thời lượng TB (phút)', 'year': 'Năm phát hành TB',
        })
        pdf = pdf[['Cụm', 'Số phim', 'Tỷ lệ', 'Điểm IMDb TB', 'Metascore TB',
                   'Thời lượng TB (phút)', 'Năm phát hành TB']]
        summary = result.get('summary') or {}
        pdf = pd.concat([pdf, pd.DataFrame([{
            'Cụm': 'Phim của bạn', 'Số phim': '—', 'Tỷ lệ': '—',
            'Điểm IMDb TB': summary.get('rating'),
            'Metascore TB': summary.get('metascore'),
            'Thời lượng TB (phút)': summary.get('duration'),
            'Năm phát hành TB': summary.get('year'),
        }])], ignore_index=True)
        st.markdown('**Hồ sơ các cụm phim**')
        st.dataframe(pdf, use_container_width=True, hide_index=True)
    points = result.get('points')
    if points is not None:
        names = {p['cluster']: p['name'] for p in (profiles or [])} or dict(mp.CLUSTER_NAMES)
        cloud = points.copy()
        cloud['Cụm'] = cloud['cluster'].map(
            lambda c: f"{c}: {names.get(c, f'Nhóm {c}')}")
        fig = px.scatter(cloud, x='pc1', y='pc2', color='Cụm', opacity=0.45,
                         title='Không gian PCA (2 chiều đầu tiên) — phim của bạn là ngôi sao ⭐',
                         color_discrete_sequence=px.colors.qualitative.Set2)
        fig.add_scatter(
            x=[result['pc1']], y=[result['pc2']], mode='markers',
            marker=dict(size=20, symbol='star', color='#111',
                        line=dict(width=1.5, color='white')),
            name='Phim của bạn',
        )
        fig.update_layout(height=430, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
    _features_expander(result['features'], '20 cột không gian PCA')


def dataset_overview() -> None:
    st.markdown('##### Dữ liệu phim IMDb (3.716 phim sau làm sạch)')
    df = load_processed('imdb_for_classification.csv')
    c1, c2 = st.columns(2, gap='large')
    counts = df['rating_class'].value_counts()
    labels = counts.index.map({'High': 'Cao (≥ 7.0)', 'Low': 'Thấp (< 7.0)'}).tolist()
    fig1 = px.bar(x=labels, y=counts.values.tolist(), color=labels,
                  color_discrete_map={'Cao (≥ 7.0)': '#2e7d32', 'Thấp (< 7.0)': '#c62828'},
                  title='Phân bố nhãn điểm cao / thấp', text=counts.values.tolist())
    fig1.update_layout(showlegend=False, height=330,
                       xaxis_title=None, yaxis_title='Số phim',
                       margin=dict(l=10, r=10, t=40, b=10))
    genre_sums = df[mp.GENRE_FLAG_COLS].sum().sort_values(ascending=False).head(10)
    y_labels = [g.replace('genre_', '').replace('_', ' ').title() for g in genre_sums.index]
    fig2 = px.bar(x=genre_sums.values.tolist(), y=y_labels, orientation='h',
                  title='10 thể loại phổ biến nhất',
                  color=genre_sums.values.tolist(), color_continuous_scale='Blues')
    fig2.update_layout(showlegend=False, height=330,
                       xaxis_title='Số phim', yaxis_title=None,
                       yaxis={'autorange': 'reversed'},
                       margin=dict(l=10, r=10, t=40, b=10))
    c1.plotly_chart(fig1, use_container_width=True)
    c2.plotly_chart(fig2, use_container_width=True)
    st.dataframe(
        df[['title', 'year', 'duration_minutes', 'metascore',
            'primary_genre', 'certificates', 'rating_class']].head(8),
        use_container_width=True, hide_index=True)
