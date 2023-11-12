import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from tqdm import tqdm
import os

def prepare_price_data(df, time_periods):
    combined_df = pd.DataFrame()
    for asset in tqdm(df.columns, desc='Preparing data across assets'):
        asset_data = df[asset].dropna()
        if len(asset_data) < max(time_periods) + 1:
            continue
        data_dict = {}
        min_length = len(asset_data) - max(time_periods)
        for period in time_periods:
            past_data = []
            for i in range(len(asset_data) - period):
                cumulative_return = np.prod(1 + asset_data.iloc[i:i + period]) - 1
                past_data.append(cumulative_return)
            data_dict[f'past_{period}'] = past_data[-min_length:]
        next_month_returns = asset_data.iloc[max(time_periods):].tolist()[-min_length:]
        data_dict['next_month'] = next_month_returns
        asset_df = pd.DataFrame(data_dict)
        combined_df = pd.concat([combined_df, asset_df], ignore_index=True)
    combined_df.to_csv('prepared_trend_regression_data.csv')
    return combined_df

def run_sign_regression_price(file_path='all_data.csv', time_periods=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120], column_filters=[]):
    df = pd.read_csv(file_path, parse_dates=True, index_col=0)
    df = df[df.columns[df.columns.str.contains('|'.join(column_filters))]]
    combined_df = prepare_price_data(df, time_periods)
    t_stats_dict = {}
    for period in tqdm(time_periods, desc='Running sign regression'):
        X = np.where(combined_df[f'past_{period}'] > 0, 1, 0)
        y = np.where(combined_df['next_month'] > 0, 1, 0)
        X = sm.add_constant(X)
        model = sm.Logit(y, X).fit(disp=0)
        t_stats_dict[period] = model.tvalues[1]
    plt.figure(figsize=(10, 6))
    x_values = range(len(t_stats_dict.keys()))
    plt.bar(x_values, list(t_stats_dict.values()), color='blue')
    plt.xlabel('Lookback Period (Months)')
    plt.ylabel('T-statistic')
    plt.title('T-statistics for Sign of Past Returns by Lookback Period')
    plt.xticks(x_values, list(t_stats_dict.keys()))
    plt.savefig('t_stats_bar_chart_sign.png')
    
def run_magnitude_regression_price(file_path='all_data.csv', time_periods=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120], column_filters=[]):
    df = pd.read_csv(file_path, parse_dates=True, index_col=0)
    df = df[df.columns[df.columns.str.contains('|'.join(column_filters))]]
    combined_df = prepare_price_data(df, time_periods)
    t_stats_dict = {}
    for period in tqdm(time_periods):
        X = combined_df[f'past_{period}']
        y = combined_df['next_month']
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        t_stats_dict[period] = model.tvalues[1]
    plt.figure(figsize=(10, 6))
    x_values = range(len(t_stats_dict.keys()))
    plt.bar(x_values, list(t_stats_dict.values()), color='blue')
    plt.xlabel('Lookback Period (Months)')
    plt.ylabel('T-statistic')
    plt.title('T-statistics for Magnitude of Past Returns by Lookback Period')
    plt.xticks(x_values, list(t_stats_dict.keys()))
    plt.savefig('t_stats_bar_chart_magnitude.png')

def find_common_countries(predictor_prefix, target_prefix, df):
    predictor_countries = {col.replace(predictor_prefix + '_', '') for col in df.columns if col.startswith(predictor_prefix)}
    target_countries = {col.replace(target_prefix + '_', '') for col in df.columns if col.startswith(target_prefix)}
    common_countries = predictor_countries.intersection(target_countries)
    return list(common_countries)

def process_fundamental_data(data, asset_type, indicator_type, lag):
    df = pd.read_csv(data)
    asset_cols = [col for col in df.columns if asset_type in col]
    indicator_cols = [col for col in df.columns if indicator_type in col]
    df = df[asset_cols + indicator_cols]
    common_countries = find_common_countries(indicator_type, asset_type, df)
    common_cols = [col for col in df.columns if any(country in col for country in common_countries)]
    df = df[common_cols]
    for col in df.columns:
        if indicator_type in col:
            if indicator_type == 'exchange_rate':
                df[col] = (df[col] - df[col].shift(12)) / df[col].shift(12)
            else:
                df[col] = df[col] - df[col].shift(12)
    X, Y = [], []
    for country in common_countries:
        indicator_col = f'{indicator_type}_{country}'
        asset_col = f'{asset_type}_{country}'
        pair_df = df[[indicator_col, asset_col]].dropna()
        pair_df[indicator_col] = pair_df[indicator_col].shift(lag)  # Shift the asset data by 'lag' places
        pair_df = pair_df.dropna()  # Drop the rows with NaN values resulting from the shift
        X.extend(pair_df[indicator_col])
        Y.extend(pair_df[asset_col])
    return np.array(X), np.array(Y)

def run_sign_regression(data, asset_type, indicator_type, lag):
    X, Y = process_fundamental_data(data, asset_type, indicator_type, lag)
    Y = np.sign(Y)
    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit()
    return model.tvalues[1]

def run_magnitude_regression_fundamentals(data, asset_type, indicator_type, lag):
    X, Y = process_fundamental_data(data, asset_type, indicator_type, lag)
    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit()
    return model.tvalues[1]

def generate_t_stat_charts(data):
    asset_types = ['equity_index', 'government_bond', 'currency']
    indicator_types = ['gdp', 'cpi', 'interest_rate', 'exchange_rate']
    lags = [0, {'gdp': 6, 'cpi': 3, 'interest_rate': 1, 'exchange_rate': 1}, 12]
    for i, lag in enumerate(lags):
        t_stats = []
        labels = []
        for asset_type in asset_types:
            for indicator_type in indicator_types:
                if isinstance(lag, dict):
                    current_lag = lag.get(indicator_type, 1)
                else:
                    current_lag = lag
                sign_t_stat = run_sign_regression(data, asset_type, indicator_type, current_lag)
                magnitude_t_stat = run_magnitude_regression_fundamentals(data, asset_type, indicator_type, current_lag)
                t_stats.extend([sign_t_stat, magnitude_t_stat])
                labels.extend([f'Sign {asset_type} {indicator_type}', f'Magnitude {asset_type} {indicator_type}'])
                print(asset_type, indicator_type, lag)
        plt.figure(figsize=(10, 6))
        plt.bar(labels, t_stats, color='blue')
        plt.xlabel('Asset and Indicator Type')
        plt.ylabel('T-statistic')
        plt.title(f'T-statistics for Sign and Magnitude by Asset and Indicator Type')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(f't_stats_bar_chart_{i}.png')