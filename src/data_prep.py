import pandas as pd
from typing import Tuple, List


# Целевые события (по результатам EDA)
TARGET_EVENT_ACTIONS: List[str] = [
    'click_on_subscription',
]


def load_sessions_hits(
    sessions_path: str,
    hits_path: str,
    sep: str = ','
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Чтение CSV сессий и хитов.
    """
    sessions = pd.read_csv(sessions_path, sep=sep, low_memory=False)
    hits = pd.read_csv(hits_path, sep=sep, low_memory=False)
    return sessions, hits


def basic_clean_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Базовая очистка таблицы ga_sessions:
    - приведение типов даты/времени,
    - удаление дубликатов,
    - заполнение категориальных NaN значением 'unknown',
    - приведение visit_number к int.
    """
    df = df.copy()

    # даты/время
    if 'visit_date' in df.columns:
        df['visit_date'] = pd.to_datetime(df['visit_date'], errors='coerce')

    if 'visit_time' in df.columns:
        df['visit_time'] = df['visit_time'].astype(str)

    # полные дубликаты
    df = df.drop_duplicates()

    # категориальные признаки
    cat_cols = [
        'utm_source', 'utm_medium', 'utm_campaign', 'utm_adcontent',
        'utm_keyword', 'device_category', 'device_os', 'device_brand',
        'device_model', 'device_screen_resolution', 'device_browser',
        'geo_country', 'geo_city'
    ]

    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna('unknown').astype(str)

    # номер визита
    if 'visit_number' in df.columns:
        df['visit_number'] = df['visit_number'].fillna(1).astype(int)

    return df


def basic_clean_hits(df: pd.DataFrame) -> pd.DataFrame:
    """
    Базовая очистка таблицы ga_hits:
    - приведение типов даты/времени,
    - удаление дубликатов.
    """
    df = df.copy()

    if 'hit_date' in df.columns:
        df['hit_date'] = pd.to_datetime(df['hit_date'], errors='coerce')

    if 'hit_time' in df.columns:
        df['hit_time'] = pd.to_datetime(df['hit_time'], errors='coerce').dt.time

    df = df.drop_duplicates()

    return df


def build_session_target(hits: pd.DataFrame) -> pd.Series:
    hits = hits.copy()

    mask = hits['event_action'].isin(TARGET_EVENT_ACTIONS)

    target_by_session = (
        hits[mask]
        .groupby('session_id')
        .size()
        .rename('target')
        .gt(0)
        .astype(int)
    )

    return target_by_session


def aggregate_hits_features(hits: pd.DataFrame) -> pd.DataFrame:
    """
    Строит агрегаты по хитовому уровню на уровне session_id:
    - hits_count: max(hit_number)
    - unique_pages: количество уникальных страниц
    - target_hits: количество целевых хитов
    - target_hits_share: доля целевых хитов
    """
    hits = hits.copy()

    hits['is_target'] = hits['event_action'].isin(TARGET_EVENT_ACTIONS).astype(int)

    agg = hits.groupby('session_id').agg(
        hits_count=('hit_number', 'max'),
        unique_pages=('hit_page_path', 'nunique'),
        target_hits=('is_target', 'sum')
    ).reset_index()

    agg['target_hits_share'] = agg['target_hits'] / agg['hits_count'].clip(lower=1)

    return agg


def build_training_table(
    sessions: pd.DataFrame,
    hits: pd.DataFrame
) -> pd.DataFrame:
    """
    Собирает финальную обучающую таблицу train_df на уровне session_id:
    - очищенные сессии,
    - агрегаты по хитам,
    - таргет.
    """
    sessions_clean = basic_clean_sessions(sessions)
    hits_clean = basic_clean_hits(hits)

    target_by_session = build_session_target(hits_clean)
    hits_agg = aggregate_hits_features(hits_clean)

    df = (
        sessions_clean
        .merge(hits_agg, on='session_id', how='left')
        .merge(target_by_session, on='session_id', how='left')
    )

    # числовые пропуски → 0
    num_cols = ['hits_count', 'unique_pages', 'target_hits', 'target_hits_share']
    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # таргет: NaN → 0
    df['target'] = df['target'].fillna(0).astype(int)

    return df