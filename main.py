import data_processing as dp
import regressions as rg
import results as rs
import time

start_time = time.time()

# dp.process_price_data('raw_equities', 'processed_equities', dp.process_equity_data)
# dp.process_price_data('raw_commodities', 'processed_commodities', dp.process_commodity_data)
# dp.process_price_data('raw_currencies', 'processed_currencies', dp.process_fx_data)
# dp.process_price_data('raw_government_bonds', 'processed_government_bonds', dp.process_government_bond_data, 'raw_currencies')
# dp.process_price_data('raw_interest_rate_swaps', 'processed_interest_rate_swaps', dp.process_interest_rate_swap_data, 'raw_currencies')
# dp.compute_excess_returns_and_combine_price_data()

# dp.process_fundamental_data('raw_fundamental_data/GDP.csv', 'processed_gdp_data', dp.process_gdp_data)
# dp.process_fundamental_data('raw_fundamental_data/CPI.csv', 'processed_cpi_data', dp.process_cpi_data)
# dp.process_fundamental_data('raw_fundamental_data/Interest_Rates.csv', 'processed_interest_rate_data', dp.process_interest_rate_data)
# dp.process_fundamental_data('raw_fundamental_data/Exchange_Rates.csv', 'processed_exchange_rate_data', dp.process_exchange_rate_data)
# dp.combine_all_data()

time_periods=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120]
assets = ['equity_index', 'commodity', 'currency', 'government_bond']
# rg.run_sign_regression_price('all_data.csv', time_periods, assets)
# rg.run_magnitude_regression_price('all_data.csv', time_periods, assets)

# rp_returns, rp_weights = rs.risk_parity('all_data.csv', weights_df= None, target_annual_vol=0.1, column_filters=assets)
# print(rs.compute_metrics(rp_returns))

# tf_weights = rs.generate_trend_following_weights('all_data.csv', assets)
# tf_returns, tf_rp_weights = rs.risk_parity('all_data.csv', weights_df= tf_weights, target_annual_vol=0.1, column_filters=assets)
# tf_returns.to_csv('tf_returns.csv')
# print(rs.compute_metrics(tf_returns))

# rs.compute_and_plot_sharpe_ratios('all_data.csv', ['equity_index', 'commodity', 'currency', 'government_bond'], 0.1)

# rs.plot_cumulative_returns(rp_returns, tf_returns, 'Risk-Parity', 'Trend-Following', 'Trend Following Cumulative Returns Graph.png')

# rg.generate_t_stat_charts('all_data.csv')

# rs.gm_compute_and_plot_sharpe_ratios()
# rs.compare_portfolios()

import pandas as pd

# Read the CSV files into pandas DataFrames
rp_returns = pd.read_csv('rp_returns.csv', index_col=0, squeeze=True)
tf_returns = pd.read_csv('tf_returns.csv', index_col=0, squeeze=True)
igm_returns = pd.read_csv('igm_returns.csv', index_col=0, squeeze=True)

# Drop rows where at least one element is missing in either DataFrame
rp_returns_clean = rp_returns.dropna()
tf_returns_clean = tf_returns.dropna()
igm_returns_clean = igm_returns.dropna()

# Only keep rows where the index is present in both DataFrames
common_index = tf_returns_clean.index.intersection(igm_returns_clean.index).intersection(rp_returns_clean.index)
tf_returns_common = tf_returns_clean.loc[common_index]
igm_returns_common = igm_returns_clean.loc[common_index]
rp_returns_common = rp_returns_clean.loc[common_index]

# Calculate and print the correlation
correlation = tf_returns_common.corr(igm_returns_common)
print("Correlation between Trend-Following and Integrated Global Macro returns:")
print(correlation)

# Combine tf_returns and igm_returns into a DataFrame and save to a CSV file
combined_returns_df = pd.concat([tf_returns_common, igm_returns_common], axis=1)
combined_returns_df.columns = ['tf_returns', 'igm_returns']
combined_returns_df.to_csv('combined_returns.csv')

# Convert the index to datetime
combined_returns_df.index = pd.to_datetime(combined_returns_df.index)

# Create a date range that covers all the dates in combined_returns_df and 3 years before the first date
date_range = pd.date_range(start=combined_returns_df.index.min() - pd.DateOffset(years=3), end=combined_returns_df.index.max())

# Create initial weights DataFrame with the new date range
weights_df = pd.DataFrame(index=date_range, columns=combined_returns_df.columns)
weights_df['tf_returns'] = 0.9
weights_df['igm_returns'] = 0.1

# Use the risk_parity function with the initial weights
avg_returns, avg_weights = rs.risk_parity('combined_returns.csv', weights_df=weights_df, target_annual_vol=0.1)
avg_returns.to_csv('avg_returns.csv')

# Compute metrics for tf_returns, igm_returns, and rp_returns
tf_metrics = rs.compute_metrics(tf_returns_common)
igm_metrics = rs.compute_metrics(igm_returns_common)
rp_metrics = rs.compute_metrics(rp_returns_common)
avg_metrics = rs.compute_metrics(avg_returns.dropna())

print("Metrics for Risk-Parity Portfolio:")
print(rp_metrics)
print("Metrics for Trend-Following Portfolio:")
print(tf_metrics)
print("Metrics for Integrated Global Macro Portfolio:")
print(igm_metrics)
print("Metrics for Combined Portfolio:")
print(avg_metrics)

data = {
    'Risk-Parity Portfolio': rp_returns_common,
    'Trend-Following Portfolio': tf_returns_common,
    'Integrated Global Macro Portfolio': igm_returns_common,
    'Combined Portfolio': avg_returns.dropna()
}

# rs.plot_cumulative_returns(data, 'Combined Portfolio Cumulative Returns.png')

import pandas as pd
# Read the data
combined_returns_df = pd.read_csv('avg_returns.csv', index_col=0, parse_dates=True)
all_data_df = pd.read_csv('all_data.csv', index_col=0, parse_dates=True)

# Merge the dataframes on the date index
merged_df = pd.merge(combined_returns_df, all_data_df[['equity_index_us']], left_index=True, right_index=True, how='inner')
merged_df = merged_df.dropna()

# Calculate the monthly correlation between 'equity_index_us' and '0'
correlation = merged_df['equity_index_us'].corr(merged_df['0'])

print("Monthly correlation:", correlation)

end_time = time.time()
print("Total Time: ", end_time - start_time)