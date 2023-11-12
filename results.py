import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import kurtosis, skew
import matplotlib.pyplot as plt
import regressions as rg
from collections import OrderedDict

def risk_parity_weights(cov_matrix):
    num_assets = len(cov_matrix)
    initial_weights = [1./num_assets] * num_assets
    weight_sum_constraint = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
    bounds = [(0, 1) for _ in range(num_assets)]
    def objective(weights): 
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        asset_contrib = np.multiply(weights, np.dot(cov_matrix, weights)) / portfolio_variance
        return np.sum(np.square(asset_contrib - 1./num_assets))
    result = minimize(objective, initial_weights, method='SLSQP', constraints=[weight_sum_constraint], bounds=bounds)
    if not result.success:
        return initial_weights
    return result.x

def risk_parity(returns, weights_df=None, target_annual_vol=0.1, column_filters=[]):
    df = pd.read_csv(returns, parse_dates=True, index_col=0)
    df = df[df.columns[df.columns.str.contains('|'.join(column_filters))]]
    df = df.dropna(how='all')
    if weights_df is None:
        weights_df = pd.DataFrame(1, index=df.index, columns=df.columns)    
    rp_returns = pd.Series(index=df.index, dtype='float64')
    rp_weights = pd.DataFrame(index=df.index, columns=df.columns)
    for end_date in df.index[36:]:
        start_date = end_date - pd.DateOffset(years=3)
        rolling_returns = df.loc[start_date:end_date]
        rolling_returns = rolling_returns.dropna(axis=1)
        cov_matrix = rolling_returns.cov()
        optimal_weights = risk_parity_weights(cov_matrix)
        current_weights = weights_df.loc[end_date, rolling_returns.columns]
        combined_weights = optimal_weights * current_weights        
        portfolio_vol = np.sqrt(np.dot(combined_weights.T, np.dot(cov_matrix, combined_weights)))
        if portfolio_vol != 0:
            scaling_factor = target_annual_vol / (portfolio_vol * np.sqrt(12))
        else:
            scaling_factor = 1
        scaled_weights = combined_weights * scaling_factor
        rp_weights.loc[end_date, rolling_returns.columns] = scaled_weights
        rp_returns[end_date] = np.dot(scaled_weights, rolling_returns.loc[end_date])
    return rp_returns, rp_weights

def compute_metrics(df):
    df.dropna(inplace=True)
    cumulative_return = (1 + df).cumprod()
    annual_return = (cumulative_return.iloc[-1] ** (12 / len(df))) - 1
    annual_volatility = df.std() * (12 ** 0.5)
    sharpe_ratio = annual_return / annual_volatility
    kurt = kurtosis(df)
    skw = skew(df)
    metrics = {
        'Annual Return': annual_return,
        'Annual Volatility': annual_volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Kurtosis': kurt,
        'Skew': skw
    }
    return metrics

def generate_trend_following_weights(returns, column_filters=[]):
    df = pd.read_csv(returns, parse_dates=True, index_col=0)
    df = df[df.columns[df.columns.str.contains('|'.join(column_filters))]]
    rolling_volatility = df.rolling(window=36).std().shift(1)
    past_1_month_return = df.shift(1)
    past_3_month_return = df.rolling(window=3).apply(lambda x: np.prod(1 + x) - 1, raw=True).shift(1)
    past_12_month_return = df.rolling(window=12).apply(lambda x: np.prod(1 + x) - 1, raw=True).shift(1)
    risk_adjusted_signal_1 = past_1_month_return / (rolling_volatility * np.sqrt(1))
    risk_adjusted_signal_3 = past_3_month_return / (rolling_volatility * np.sqrt(3))
    risk_adjusted_signal_12 = past_12_month_return / (rolling_volatility * np.sqrt(12))
    risk_adjusted_signal_1 = np.clip(risk_adjusted_signal_1, -2, 2)
    risk_adjusted_signal_3 = np.clip(risk_adjusted_signal_3, -2, 2)
    risk_adjusted_signal_12 = np.clip(risk_adjusted_signal_12, -2, 2)
    combined_signal = (risk_adjusted_signal_1 + risk_adjusted_signal_3 + risk_adjusted_signal_12) / 3
    combined_signal.fillna(0, inplace=True)
    combined_signal.to_csv('trend_following_signal.csv')
    return combined_signal

def compute_and_plot_sharpe_ratios(file_path, asset_classes, target_annual_vol):
    df = pd.read_csv(file_path)
    assets = df.columns[df.columns.str.contains('|'.join(asset_classes))]
    sharpe_ratios = {}
    for asset in assets:
        tf_weights = generate_trend_following_weights(file_path, [asset])
        tf_returns, _ = risk_parity(file_path, weights_df=tf_weights, target_annual_vol=target_annual_vol, column_filters=[asset])
        metrics = compute_metrics(tf_returns)
        sharpe_ratios[asset] = metrics['Sharpe Ratio']
    colors = ['blue', 'green', 'red', 'yellow']
    color_map = dict(zip(asset_classes, colors))
    fig, ax = plt.subplots()
    bar_width = 0.5
    bar_positions = np.arange(len(sharpe_ratios))
    for i, (asset, sharpe) in enumerate(sharpe_ratios.items()):
        asset_class = next((ac for ac in asset_classes if ac in asset), None)
        ax.bar(bar_positions[i], sharpe, width=bar_width, color=color_map[asset_class])
    ax.set_xlabel('Assets')
    ax.set_ylabel('Sharpe Ratio')
    ax.set_title('Sharpe Ratios of Assets')
    plt.xticks(bar_positions, sharpe_ratios.keys(), rotation=90, fontsize=5)
    plt.tight_layout()
    plt.savefig('Individual Trend Sharpe Ratios.png')
    plt.show()

import matplotlib.pyplot as plt
import pandas as pd

def plot_cumulative_returns(data, filename):
    plt.figure(figsize=(10,6))
    
    # Convert indices to datetime and find the earliest date where all portfolios have data
    start_date = max(pd.to_datetime(series.index[0]) for series in data.values())
    
    for name, series in data.items():
        series.index = pd.to_datetime(series.index)
        series = series[series.index >= start_date]  # Only keep data from the start_date onwards
        cum_returns = (1 + series).cumprod() * 100
        plt.plot(cum_returns, label=name)
    
    plt.title('Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.yscale('log')
    plt.legend(loc='best')
    plt.savefig(filename)
    plt.show()

def generate_global_macro_weights(data, asset_type, indicator_type, sign, lag):
    df = pd.read_csv(data, parse_dates=True, index_col=0)
    df = df[df.columns[df.columns.str.contains('|'.join([asset_type, indicator_type]))]]
    common_countries = rg.find_common_countries(indicator_type, asset_type, df)
    common_cols = [col for col in df.columns if any(country in col for country in common_countries)]
    df = df[common_cols]
    for col in df.columns:
        if indicator_type in col:
            if indicator_type == 'exchange_rate':
                df[col] = (df[col] - df[col].shift(12)) / df[col].shift(12)
            else:
                df[col] = df[col] - df[col].shift(12)
    asset_types = ['equity_index', 'currency', 'government_bond']
    weights = pd.read_csv(data, parse_dates=True, index_col=0)
    weights = weights[weights.columns[weights.columns.str.contains('|'.join(asset_types))]]
    weights[:] = 0
    for country in common_countries:
        indicator_col = f'{indicator_type}_{country}'
        asset_col = f'{asset_type}_{country}'
        pair_df = df[[indicator_col, asset_col]].dropna()
        pair_df[indicator_col] = pair_df[indicator_col].shift(lag)
        pair_df = pair_df.dropna()
        for i in pair_df.index:
            if len(pair_df) < 36:
                continue
            signal = 1 if sign == "positive" else -1
            if pair_df.loc[i, indicator_col] > 0:
                weights.loc[i, asset_col] = signal
            else:
                weights.loc[i, asset_col] = signal * -1
    weights.fillna(0, inplace=True)
    return weights


def gm_compute_and_plot_sharpe_ratios():
    combinations = [
        ('equity_index', 'gdp', 'positive', 6),
        ('government_bond', 'gdp', 'negative', 6),
        ('currency', 'gdp', 'positive', 6),
        ('equity_index', 'cpi', 'negative', 3),
        ('government_bond', 'cpi', 'negative', 3),
        ('currency', 'cpi', 'positive', 3),
        ('equity_index', 'interest_rate', 'negative', 1),
        ('government_bond', 'interest_rate', 'negative', 1),
        ('currency', 'interest_rate', 'positive', 1),
        ('equity_index', 'exchange_rate', 'negative', 1),
        ('government_bond', 'exchange_rate', 'positive', 1),
        ('currency', 'exchange_rate', 'positive', 1),
    ]
    asset_types = ['equity_index', 'government_bond', 'currency']
    indicator_types = ['gdp', 'cpi', 'interest_rate', 'exchange_rate']
    returns_df = pd.DataFrame()
    sharpe_ratios = {}
    first_combination = combinations[0]
    total_weights = generate_global_macro_weights("all_data.csv", *first_combination)
    total_weights[:] = 0
    for asset in asset_types:
        asset_combinations = [comb for comb in combinations if comb[0] == asset]
        asset_weights = None
        for asset_type, indicator_type, sign, lag in asset_combinations:
            weights = generate_global_macro_weights("all_data.csv", asset_type, indicator_type, sign, lag)
            if asset_weights is None:
                asset_weights = weights
            else:
                asset_weights += weights
            total_weights += weights
            returns, _ = risk_parity('all_data.csv', weights_df=weights, target_annual_vol=0.1, column_filters=[asset_type])
            returns_df[f'{asset_type}_{indicator_type}'] = returns
            metrics = compute_metrics(returns)
            sharpe_ratios[f'{asset_type}_{indicator_type}'] = metrics['Sharpe Ratio']
            print(asset_type, indicator_type)
        intermediate_returns, _ = risk_parity('all_data.csv', weights_df=asset_weights, target_annual_vol=0.1, column_filters=[asset_type])
        returns_df[f'integrated_{asset}'] = intermediate_returns
        metrics = compute_metrics(intermediate_returns)
        sharpe_ratios[f'integrated_{asset}'] = metrics['Sharpe Ratio']
        print(f'Integrated {asset} Portfolio')
    for indicator in indicator_types:
        indicator_combinations = [comb for comb in combinations if comb[1] == indicator]
        indicator_weights = None
        for asset_type, indicator_type, sign, lag in indicator_combinations:
            weights = generate_global_macro_weights("all_data.csv", asset_type, indicator_type, sign, lag)
            if indicator_weights is None:
                indicator_weights = weights
            else:
                indicator_weights += weights
            total_weights += weights  # Update total_weights
        intermediate_returns, _ = risk_parity('all_data.csv', weights_df=indicator_weights, target_annual_vol=0.1, column_filters=asset_types)
        returns_df[f'integrated_{indicator}'] = intermediate_returns
        metrics = compute_metrics(intermediate_returns)
        sharpe_ratios[f'integrated_{indicator}'] = metrics['Sharpe Ratio']
        print(f'Integrated {indicator} Portfolio')
    integrated_returns, _ = risk_parity('all_data.csv', weights_df=total_weights, target_annual_vol=0.1, column_filters=asset_types)
    print(integrated_returns.head(10))
    integrated_returns.to_csv('integrated_returns.csv')
    returns_df['integrated_global_macro'] = integrated_returns
    metrics = compute_metrics(integrated_returns)
    sharpe_ratios['integrated_global_macro'] = metrics['Sharpe Ratio']
    print('Integrated Global Macro Portfolio')
    returns_df.to_csv('individual_gm_returns.csv')

    # Order sharpe_ratios and define colors
    individual_sharpe_ratios = {k: v for k, v in sharpe_ratios.items() if '_' in k and 'integrated' not in k}
    intermediate_sharpe_ratios = {k: v for k, v in sharpe_ratios.items() if 'integrated' in k and 'global_macro' not in k}
    integrated_sharpe_ratios = {k: v for k, v in sharpe_ratios.items() if 'global_macro' in k}
    ordered_sharpe_ratios = {**individual_sharpe_ratios, **intermediate_sharpe_ratios, **integrated_sharpe_ratios}

    # Define colors for the different groups
    colors = ['blue'] * len(individual_sharpe_ratios) + ['green'] * len(intermediate_sharpe_ratios) + ['red']
    total_weights.to_csv('integrated_global_macro_weights.csv')
    
    plt.bar(range(len(ordered_sharpe_ratios)), list(ordered_sharpe_ratios.values()), align='center', color=colors)
    plt.xticks(range(len(ordered_sharpe_ratios)), list(ordered_sharpe_ratios.keys()), rotation=90, fontsize=6)
    plt.xlabel('Portfolio')
    plt.ylabel('Sharpe Ratio')
    plt.title('Sharpe Ratios of Portfolios')
    plt.tight_layout()
    plt.savefig('Global Macro Portfolios Sharpe Ratios.png')

def compare_portfolios():
    integrated_global_macro_weights = pd.read_csv('integrated_global_macro_weights.csv', index_col=0)
    non_zero_columns = integrated_global_macro_weights.loc[:, (integrated_global_macro_weights != 0).any(axis=0)].columns
    rp_returns, _ = risk_parity('all_data.csv', weights_df=None, target_annual_vol=0.1, column_filters=non_zero_columns)
    rp_returns.to_csv('rp_strict_returns.csv')
    integrated_returns = pd.read_csv('igm_returns.csv', index_col=0)
    integrated_returns = integrated_returns.squeeze()
    # print(integrated_returns.head(10))
    rp_returns = rp_returns.squeeze()
    # print(rp_returns.tail(10))
    print(compute_metrics(integrated_returns))
    print(compute_metrics(rp_returns))
    plot_cumulative_returns(integrated_returns, rp_returns, 'Integrated Global Macro', 'Risk-Parity', 'Global Macro Cumulative Returns Graph.png')
    print("Metrics for Integrated Global Macro Portfolio:")
    print(compute_metrics(integrated_returns))
    print("Metrics for Risk-Parity Portfolio:")
    print(compute_metrics(rp_returns))

# def generate_integrated_weights(igm_csv):

#     integrated_global_macro_weights = pd.read_csv(igm_csv)
    
#     non_zero_columns = integrated_global_macro_weights.loc[:, (integrated_global_macro_weights != 0).any(axis=0)].columns
    
#     rp_returns, _ = risk_parity('all_data.csv', weights_df=None, target_annual_vol=0.1, column_filters=non_zero_columns)

#     trend_following_weights = generate_trend_following_weights('all_data.csv', non_zero_columns)
#     tf_returns, _ = risk_parity('all_data.csv', weights_df=trend_following_weights, target_annual_vol=0.1, column_filters=non_zero_columns)

#     integrated_global_macro_weights = pd.read_csv('integrated_global_macro_weights.csv', index_col=0)
#     igm_returns = pd.read_csv('integrated_returns.csv', index_col=0)

#     # Convert the index to DateTimeIndex
#     integrated_global_macro_weights.index = pd.to_datetime(integrated_global_macro_weights.index)

#     # Now convert the DateTime index to just date
#     integrated_global_macro_weights.index = integrated_global_macro_weights.index.date    
#     # Convert the index to DateTimeIndex
#     trend_following_weights.index = pd.to_datetime(trend_following_weights.index)

#     # Now convert the DateTime index to just date
#     trend_following_weights.index = trend_following_weights.index.date
#     # Now align the DataFrames
#     aligned_igm, aligned_tf = integrated_global_macro_weights.align(trend_following_weights, join='inner')

#     # Now perform the addition
#     average_weights = (aligned_igm + aligned_tf * 19) / 2    
#     average_weights.index = pd.to_datetime(average_weights.index)

#     print(average_weights)
#     avg_returns, _ = risk_parity('all_data.csv', weights_df=average_weights, target_annual_vol=0.1, column_filters=non_zero_columns)

#     metrics_rp = compute_metrics(rp_returns)
#     print("Metrics for Risk-Parity Portfolio:")
#     print(metrics_rp)

#     metrics_tf = compute_metrics(tf_returns)
#     print("Metrics for Trend-Following Portfolio:")
#     print(metrics_tf)

#     # Ensure igm_returns is a single series of returns
#     igm_returns = igm_returns.squeeze()
#     metrics_igm = compute_metrics(igm_returns)
#     print("Metrics for Integrated Global Macro Portfolio:")
#     print(metrics_igm)

#     metrics_avg = compute_metrics(avg_returns)
#     print("Metrics for Average Weights Portfolio:")
#     print(metrics_avg)