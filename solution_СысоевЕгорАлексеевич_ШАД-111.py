import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


INPUT_FILE = 'part-00000-6c86f944-e574-4f90-90af-62f188113fcf.c000.snappy.parquet'
OUTPUT_DIR = 'output'
PLOTS_DIR = os.path.join(OUTPUT_DIR, 'plots')


IQR_MULTIPLIER = 3.0


def load_data(filepath: str) -> pd.DataFrame:
    """Загрузка данных."""
    print(f"Загрузка данных из {filepath}...")
    df = pd.read_parquet(filepath)
    print(f"Загружено строк: {len(df)}")
    print(f"Столбцы: {list(df.columns)}")
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Фильтрация:
    1. BrandinDelivery == 1
    2. CategoryNameDelivery не пустой
    """
    initial_len = len(df)

    # Приводим к числу на всякий случай
    df['BrandinDelivery'] = pd.to_numeric(df['BrandinDelivery'], errors='coerce')

    # Фильтр
    df_filtered = df[
        (df['BrandinDelivery'] == 1) &
        (df['CategoryNameDelivery'].notna()) &
        (df['CategoryNameDelivery'] != '')
        ].copy()

    print(f"Строк после фильтрации: {len(df_filtered)} из {initial_len}")
    return df_filtered


def calculate_daily_ots(df: pd.DataFrame) -> pd.DataFrame:
    """
    Расчет daily_ots(i, j, k) = Weight * count_rows
    Группировка по: SubjectID, BrandID, CategoryNameDelivery, researchdate
    """
    print("Расчет daily_ots...")

    # 1. Считаем количество строк (запросов) для каждой группы
    counts = df.groupby(['SubjectID', 'BrandID', 'CategoryNameDelivery', 'researchdate']).size().reset_index(
        name='count_rows')

    # 2. Берем вес респондента за день. Он уникален для пары SubjectID + researchdate
    weights = df[['SubjectID', 'researchdate', 'Weight']].drop_duplicates(subset=['SubjectID', 'researchdate'])

    # 3. Объединяем
    ots_df = counts.merge(weights, on=['SubjectID', 'researchdate'], how='left')

    # ВАЖНО: Приводим Weight к float, чтобы избежать ошибок с Decimal
    ots_df['Weight'] = pd.to_numeric(ots_df['Weight'], errors='coerce').astype(float)

    # 4. Считаем OTS
    ots_df['daily_ots'] = ots_df['Weight'] * ots_df['count_rows']

    # 5. Добавляем название бренда для отчетов
    if 'Brand' in df.columns:
        brand_names = df[['BrandID', 'Brand']].drop_duplicates(subset=['BrandID'])
        ots_df = ots_df.merge(brand_names, on='BrandID', how='left')
    else:
        ots_df['Brand'] = ots_df['BrandID']

    print(f"Рассчитано OTS для {len(ots_df)} групп")
    return ots_df


def detect_anomalies(ots_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Поиск аномалий методом IQR с правилом Тьюки (Tukey's fences).

    Правило Тьюки: выбросы = значения > Q3 + 1.5 * IQR
    Множитель 1.5 — классическая статистическая константа, не требующая настройки.

    Дополнительно: аномалией считается только значение выше медианы,
    чтобы исключить ложные срабатывания на малых OTS.
    """
    print("Поиск аномалий методом IQR (правило Тьюки)...")

    anomalies_reasons = []

    # Группируем по Бренду, Категории и Дню (максимальная детализация)
    grouped = ots_df.groupby(['BrandID', 'CategoryNameDelivery', 'researchdate'])

    for (brand_id, category, date), group in grouped:
        # Защита от малых выборок
        if len(group) < 5:
            continue

        vals = group['daily_ots'].astype(float)

        q1 = vals.quantile(0.25)
        q3 = vals.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            continue

        # Классическое правило Тьюки: множитель 1.5
        threshold = q3 + 1.5 * iqr
        median = vals.median()

        # Только аномально высокие значения (выше порога И выше медианы)
        outliers = group[(group['daily_ots'] > threshold) & (group['daily_ots'] > median)]

        if not outliers.empty:
            for _, row in outliers.iterrows():
                anomalies_reasons.append({
                    'SubjectID': row['SubjectID'],
                    'researchdate': row['researchdate'],
                    'BrandID': row['BrandID'],
                    'Brand': row.get('Brand', 'Unknown'),
                    'CategoryNameDelivery': row['CategoryNameDelivery'],
                    'daily_ots': float(row['daily_ots']),
                    'score': float(row['daily_ots']),
                    'threshold': float(threshold),
                    'reason': f'Daily OTS {row["daily_ots"]:.2f} > Q3 + 1.5*IQR = {threshold:.2f}'
                })

    if not anomalies_reasons:
        print("Аномалии не найдены.")
        empty_cols = ['SubjectID', 'researchdate', 'BrandID', 'Brand', 'CategoryNameDelivery', 'daily_ots', 'score',
                      'threshold', 'reason']
        return pd.DataFrame(columns=['SubjectID', 'researchdate']), pd.DataFrame(columns=empty_cols)

    reasons_df = pd.DataFrame(anomalies_reasons)

    # Формируем итоговый список на удаление: уникальные пары (SubjectID, researchdate)
    anomalies_final = reasons_df[['SubjectID', 'researchdate']].drop_duplicates()

    print(f"Найдено {len(anomalies_final)} уникальных пар (SubjectID, date) для удаления.")
    return anomalies_final, reasons_df


def generate_plots(original_df: pd.DataFrame, cleaned_df: pd.DataFrame, anomalies_df: pd.DataFrame):
    """Генерация графиков."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    original_df['Weight'] = pd.to_numeric(original_df['Weight'], errors='coerce')
    cleaned_df['Weight'] = pd.to_numeric(cleaned_df['Weight'], errors='coerce')

    # 1. Total OTS Before/After
    print("Построение графика Total OTS...")
    daily_before = original_df.groupby('researchdate')['Weight'].sum().reset_index(name='Total_OTS_Before')
    daily_after = cleaned_df.groupby('researchdate')['Weight'].sum().reset_index(name='Total_OTS_After')

    daily_ots = daily_before.merge(daily_after, on='researchdate', how='left').fillna(0)

    plt.figure(figsize=(12, 6))
    plt.plot(daily_ots['researchdate'], daily_ots['Total_OTS_Before'], label='Before', color='blue')
    plt.plot(daily_ots['researchdate'], daily_ots['Total_OTS_After'], label='After', color='red')
    plt.title('Total Daily OTS Before and After Cleaning')
    plt.xlabel('Date')
    plt.ylabel('Total OTS')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'total_ots_before_after.png'))
    plt.close()

    # 2. Category OTS Change
    print("Построение графика Category OTS Change...")
    cat_before = original_df.groupby('CategoryNameDelivery')['Weight'].sum()
    cat_after = cleaned_df.groupby('CategoryNameDelivery')['Weight'].sum()

    cat_change = pd.DataFrame({'Before': cat_before, 'After': cat_after}).fillna(0)
    cat_change['Change_Pct'] = ((cat_change['After'] - cat_change['Before']) / cat_change['Before']) * 100
    cat_change = cat_change.reset_index()

    plt.figure(figsize=(12, 6))
    sns.barplot(data=cat_change, x='CategoryNameDelivery', y='Change_Pct', palette='viridis')
    plt.title('Percentage Change in OTS by CategoryNameDelivery')
    plt.xlabel('CategoryNameDelivery')
    plt.ylabel('Change (%)')
    plt.xticks(rotation=45, ha='right')
    plt.axhline(0, color='black', linewidth=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'category_ots_change.png'))
    plt.close()

    # 3. Daily Anomaly Count
    print("Построение графика Daily Anomaly Count...")
    if not anomalies_df.empty:
        daily_counts = anomalies_df.groupby('researchdate').size().reset_index(name='Anomaly_Count')

        plt.figure(figsize=(12, 6))
        sns.barplot(data=daily_counts, x='researchdate', y='Anomaly_Count', color='salmon')
        plt.title('Number of Anomalous Respondents per Day')
        plt.xlabel('Date')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'daily_anomaly_count.png'))
        plt.close()


def save_results(anomalies_df: pd.DataFrame, reasons_df: pd.DataFrame):
    """Сохранение CSV."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    anomalies_df[['SubjectID', 'researchdate']].drop_duplicates().to_csv('output/anomalies.csv', index=False)
    reasons_df.to_csv(os.path.join(OUTPUT_DIR, 'anomaly_reasons.csv'), index=False)
    print(f"Файлы сохранены в {OUTPUT_DIR}")


def get_cleaned_data(original_df: pd.DataFrame, anomalies_df: pd.DataFrame) -> pd.DataFrame:
    """Удаление строк аномальных респондентов за соответствующие дни."""
    if anomalies_df.empty:
        return original_df

    # Создаем ключ для быстрой фильтрации
    anomalies_df['_key'] = anomalies_df['SubjectID'].astype(str) + '_' + anomalies_df['researchdate'].astype(str)
    original_df = original_df.copy()
    original_df['_key'] = original_df['SubjectID'].astype(str) + '_' + original_df['researchdate'].astype(str)

    # Оставляем только те строки, которых НЕТ в списке аномалий
    cleaned_df = original_df[~original_df['_key'].isin(anomalies_df['_key'])].copy()

    # Чистим временные колонки
    original_df.drop(columns=['_key'], inplace=True)
    cleaned_df.drop(columns=['_key'], inplace=True)

    return cleaned_df


# --- Аналитические функции (по заданию) ---

def plot_demographic_split(original_df, cleaned_df, col_name):
    """График до/после по демографии."""
    before = original_df.groupby(col_name)['Weight'].sum()
    after = cleaned_df.groupby(col_name)['Weight'].sum()
    df_comp = pd.DataFrame({'Before': before, 'After': after}).fillna(0)
    df_comp['Change_Pct'] = ((df_comp['After'] - df_comp['Before']) / df_comp['Before']) * 100

    plt.figure(figsize=(10, 5))
    df_comp['Change_Pct'].plot(kind='bar', color='teal')
    plt.title(f'OTS Change by {col_name}')
    plt.ylabel('Change (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def show_anomaly_queries(original_df, subject_id, date):
    """Показать запросы конкретного аномального респондента."""
    mask = (original_df['SubjectID'] == subject_id) & (original_df['researchdate'] == date)
    queries = original_df.loc[mask, ['QueryText', 'Brand', 'CategoryNameDelivery', 'Weight']]
    print(f"Queries for Subject {subject_id} on {date}:")
    print(queries)


def main():
    print("=== Start Solution ===")

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Файл {INPUT_FILE} не найден.")

    # 1. Загрузка
    df = load_data(INPUT_FILE)

    # 2. Предобработка
    df_filtered = preprocess_data(df)

    # 3. Расчет OTS
    ots_df = calculate_daily_ots(df_filtered)

    # 4. Поиск аномалий
    anomalies_df, reasons_df = detect_anomalies(ots_df)

    # 5. Очистка данных
    cleaned_df = get_cleaned_data(df_filtered, anomalies_df)

    # 6. Сохранение
    save_results(anomalies_df, reasons_df)

    # 7. Графики
    generate_plots(df_filtered, cleaned_df, anomalies_df)

    print("=== Solution Completed Successfully ===")


if __name__ == "__main__":
    main()