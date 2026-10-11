"""Đọc/gộp CSV IMDb. Không chỉnh sửa dữ liệu gốc."""
from pathlib import Path
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / 'data' / 'raw'


def load_raw_data(raw_dir=RAW_DIR):
    raw_dir = Path(raw_dir)
    paths = sorted(raw_dir.glob('*.csv'))
    if not paths:
        raise FileNotFoundError(
            f'Không có CSV trong {raw_dir}. Hãy tải dataset Kaggle và chép mọi file CSV vào đây.'
        )
    parts, inventory = [], []
    for path in paths:
        try:
            frame = pd.read_csv(path, low_memory=False, encoding='utf-8-sig', na_values=['\\N', 'N/A', 'null', 'NULL', '-', '--'])
        except UnicodeDecodeError:
            frame = pd.read_csv(path, low_memory=False, encoding='cp1252', na_values=['\\N', 'N/A', 'null', 'NULL', '-', '--'])
        if frame.empty:
            inventory.append({'file': path.name, 'rows': 0, 'columns': len(frame.columns), 'status': 'Bỏ qua: file rỗng'})
            continue
        inventory.append({'file': path.name, 'rows': len(frame), 'columns': len(frame.columns), 'status': 'Đã đọc'})
        frame['source_file'] = path.name
        parts.append(frame)
    if not parts:
        raise ValueError('Không có dòng dữ liệu trong các CSV.')
    # Chuẩn hóa tiêu đề trước khi concat: các file có thể đặt tên cột khác dấu/cách.
    df = pd.concat(parts, ignore_index=True, sort=False)
    return df, pd.DataFrame(inventory)
