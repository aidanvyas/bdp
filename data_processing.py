import os
import pandas as pd
import numpy as np
from dateutil.parser import parse
from datetime import datetime

def process_price_data(input_directory: str, output_directory: str, processing_function, fx_directory: str = None):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    for file in os.listdir(input_directory):
        input_file = os.path.join(input_directory, file)
        output_file = os.path.join(output_directory, file)
        processing_function(input_file, output_file, fx_directory) if fx_directory else processing_function(input_file, output_file)
            
def process_equity_data(input_file: str, output_file: str):
    df = pd.read_csv(input_file)
    asset_name = os.path.splitext(os.path.basename(input_file))[0]
    valid_rows = []
    for index, row in df.iterrows():
        if str(row[6]).replace('--', '').strip():
            try:
                date = parse(str(row[0]))
                value = float(row[6].replace('--', '').strip().replace('%', '')) / 100
                valid_row = [date.strftime('%Y-%m'), value]
                valid_rows.append(valid_row)
            except ValueError:
                continue
    sorted_df = pd.DataFrame(valid_rows, columns=['Date', asset_name]).sort_values(by=['Date'])
    sorted_df.to_csv(output_file, index=False)

def process_commodity_data(input_file: str, output_file: str):
    process_equity_data(input_file, output_file)

def process_fx_data(input_file: str, output_file: str):
    df = pd.read_csv(input_file, skiprows=6)
    asset_name = os.path.basename(input_file).split('.')[0]
    df = df[['Date', 'PX_LAST']]
    df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True).dt.strftime('%Y-%m')
    df = df.sort_values(by='Date')
    df.reset_index(drop=True, inplace=True)
    df['PX_LAST'] = 1 / df['PX_LAST']
    df['Return'] = df['PX_LAST'].pct_change()
    df.replace("", np.nan, inplace=True)
    df.dropna(inplace=True)
    df.drop(columns=['PX_LAST'], inplace=True)
    df.columns = ['Date', asset_name]
    df.to_csv(output_file, index=False)

def process_government_bond_data(input_file: str, output_file: str, fx_directory: str):
    df = pd.read_csv(input_file)
    df = df.iloc[:, :2]
    df.to_csv(output_file, index=False)
    process_interest_rate_swap_data(output_file, output_file, fx_directory)

def process_interest_rate_swap_data(input_file: str, output_file: str, fx_directory: str):
    df = pd.read_csv(input_file)
    asset_name = os.path.basename(input_file).split('.')[0]
    currency = df.iloc[3, 1]
    df = pd.read_csv(input_file, skiprows=6)
    df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True).dt.strftime('%Y-%m')
    df.sort_values(by='Date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = df.rename(columns={'PX_LAST': currency})
    if currency != "USD":
        currency_file = f'USD{currency}.csv'
        currency_df = pd.read_csv(os.path.join(fx_directory, currency_file), skiprows=6)
        currency_df['PX_LAST'] = pd.to_numeric(currency_df['PX_LAST'], errors='coerce')
        currency_df['Date'] = pd.to_datetime(currency_df['Date'], infer_datetime_format=True).dt.strftime('%Y-%m')
        df = df.merge(currency_df, on='Date', how='left')
        df['USD_VALUE'] = pd.to_numeric(df[currency]) / df['PX_LAST']
    else:
        df['USD_VALUE'] = pd.to_numeric(df[currency])
    df['Return'] = df['USD_VALUE'].pct_change()
    df.replace("", np.nan, inplace=True)
    df.dropna(inplace=True)
    output_df = df[['Date', 'Return']].rename(columns={'Return': asset_name})
    output_df.to_csv(output_file, index=False)

def compute_excess_returns_and_combine_price_data():
    rf_file_path = 'ff.csv'
    data_folders = ['processed_equities', 'processed_currencies', 'processed_commodities', 'processed_government_bonds']
    rf_data = pd.read_csv(rf_file_path, skiprows=3, usecols=[0, 4])
    rf_data = rf_data[['Unnamed: 0', 'RF']].rename(columns={'Unnamed: 0': 'Date'})
    rf_data['Date'] = pd.to_datetime(rf_data['Date'])
    rf_data['Date'] = rf_data['Date'].dt.to_period('M')    
    rf_data.set_index('Date', inplace=True)
    all_dataframes = []
    for folder in data_folders:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            asset_data = pd.read_csv(file_path)
            asset_data['Date'] = pd.to_datetime(asset_data['Date']).dt.to_period('M')
            asset_data.set_index('Date', inplace=True)
            all_dataframes.append(asset_data)
    combined_data = pd.concat(all_dataframes, axis=1)
    combined_data = combined_data.join(rf_data, how='left')
    for col in combined_data.columns:
        if 'RF' in col:
            combined_data[col] = combined_data[col] - combined_data["RF"]
    combined_data.drop(columns="RF", inplace=True)
    combined_data = combined_data.sort_index()
    combined_data = combined_data.loc[:pd.Timestamp('2022-12').to_period('M')]
    combined_data.drop_duplicates(inplace=True)
    combined_data.to_csv("output_file.csv")

def process_fundamental_data(input_directory: str, output_directory: str, process_function):
    data = pd.read_csv(input_directory, skiprows=1)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    for i in range(len(data)):
        country = data.iloc[i, 0]
        dates = []
        values = []
        for j in range(1, len(data.columns)):
            column_name = process_function(data, dates, values, i, j)
        df = pd.DataFrame({
            "Date": dates,
            column_name: values
        }).dropna()
        df[column_name] = df[column_name].astype(str).str.replace(',', '')
        df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
        if column_name == 'GDP Growth':
            df[column_name] = ((df[column_name] - df[column_name].shift(12)) / df[column_name].shift(12))
        file_name = f"{country}_{column_name}.csv".replace(",", "").replace(" ", "_")
        file_path = f"{output_directory}/{file_name}"
        df.dropna(inplace=True)
        df.rename(columns={column_name: file_name.replace(".csv", "")}, inplace=True)
        df[['Date', file_name.replace(".csv", "")]].to_csv(file_path, index=False)

def process_gdp_data(data, dates, values, i, j):
    quarter, year = data.columns[j].split(' ')
    if quarter[1].isdigit():
        quarter_number = int(quarter[1])
        for month in range((quarter_number - 1) * 3 + 1, quarter_number * 3 + 1):
            dates.append(f"{year}-{month:02d}")
            value = data.iloc[i, j]
            if pd.to_numeric(value, errors='coerce'):
                values.append(value)
    return 'GDP Growth'

def process_monthly_data(data, dates, values, i, j):
    date_str = data.columns[j]
    date = datetime.strptime(date_str, "%b %Y").strftime("%Y-%m")
    value = data.iloc[i, j]
    if pd.to_numeric(value, errors='coerce'):
        dates.append(date)
        values.append(value)

def process_cpi_data(data, dates, values, i, j):
    process_monthly_data(data, dates, values, i, j)
    return 'CPI Growth'

def process_interest_rate_data(data, dates, values, i, j):
    process_monthly_data(data, dates, values, i, j)
    return 'Interest Rate'

def process_exchange_rate_data(data, dates, values, i, j):
    process_monthly_data(data, dates, values, i, j)
    return 'Exchange Rate'

def combine_all_data():
    data_folders = {
        'gdp': 'processed_gdp_data',
        'cpi': 'processed_cpi_data',
        'interest_rate': 'processed_interest_rate_data',
        'exchange_rate': 'processed_exchange_rate_data'
    }
    asset_data = pd.read_csv("output_file.csv")
    for data_type, folder_path in data_folders.items():
        for data_file in os.listdir(folder_path):
            data_path = os.path.join(folder_path, data_file)
            data = pd.read_csv(data_path)
            asset_data = pd.merge(asset_data, data, on='Date', how='left')
            asset_data.drop_duplicates(inplace=True)
    rename_columns = {
        'MXAU': 'equity_index_australia',
        'MXAR': 'equity_index_argentina',
        'MXBR': 'equity_index_brazil',
        'MXCA': 'equity_index_canada',
        'MXCN': 'equity_index_china',
        'MXFR': 'equity_index_france',
        'MXDE': 'equity_index_germany',
        'MXIN': 'equity_index_india',
        'MXID': 'equity_index_indonesia',
        'MXIT': 'equity_index_italy',
        'MXJP': 'equity_index_japan',
        'MXMX': 'equity_index_mexico',
        'MXKR': 'equity_index_south_korea',
        'MXRU': 'equity_index_russia',
        'MXSA': 'equity_index_saudi_arabia',
        'MXZA': 'equity_index_south_africa',
        'MXTR': 'equity_index_turkey',
        'MXGB': 'equity_index_uk',
        'MXUS': 'equity_index_us',
        'USDARS': 'currency_argentina',
        'USDAUD': 'currency_australia',
        'USDBRL': 'currency_brazil',
        'USDCAD': 'currency_canada',
        'USDCNY': 'currency_china',
        'USDINR': 'currency_india',
        'USDIDR': 'currency_indonesia',
        'USDJPY': 'currency_japan',
        'USDMXN': 'currency_mexico',
        'USDKRW': 'currency_south_korea',
        'USDRUB': 'currency_russia',
        'USDSAR': 'currency_saudi_arabia',
        'USDZAR': 'currency_south_africa',
        'USDTRY': 'currency_turkey',
        'USDGBP': 'currency_uk',
        'USDEUR': 'currency_eurozone',
        'America_G1': 'government_bond_us',
        'France_G1': 'government_bond_france',
        'China_G1': 'government_bond_china',
        'Australia_G1': 'government_bond_australia',
        'Japan_G1': 'government_bond_japan',
        'England_G1': 'government_bond_uk',
        'Germany_G1': 'government_bond_germany',
        'Mexico_G1': 'government_bond_mexico',
        'Italy_G1': 'government_bond_italy',
        'Canada_G1': 'government_bond_canada',
        'Argentina_GDP_Growth': 'gdp_argentina',
        'Australia_GDP_Growth': 'gdp_australia',
        'Brazil_GDP_Growth': 'gdp_brazil',
        'Canada_GDP_Growth': 'gdp_canada',
        'France_GDP_Growth': 'gdp_france',
        'Germany_GDP_Growth': 'gdp_germany',
        'India_GDP_Growth': 'gdp_india',
        'Indonesia_GDP_Growth': 'gdp_indonesia',
        'Italy_GDP_Growth': 'gdp_italy',
        'Japan_GDP_Growth': 'gdp_japan',
        'Korea_Rep._of_GDP_Growth': 'gdp_south_korea',
        'Mexico_GDP_Growth': 'gdp_mexico',
        'Russian_Federation_GDP_Growth': 'gdp_russia',
        'Saudi_Arabia_GDP_Growth': 'gdp_saudi_arabia',
        'South_Africa_GDP_Growth': 'gdp_south_africa',
        'Türkiye_Rep_of_GDP_Growth': 'gdp_turkey',
        'United_Kingdom_GDP_Growth': 'gdp_uk',
        'United_States_GDP_Growth': 'gdp_us',
        'Brazil_CPI_Growth': 'cpi_brazil',
        'Canada_CPI_Growth': 'cpi_canada',
        'China_P.R.:_Mainland_CPI_Growth': 'cpi_china',
        'France_CPI_Growth': 'cpi_france',
        'Germany_CPI_Growth': 'cpi_germany',
        'India_CPI_Growth': 'cpi_india',
        'Indonesia_CPI_Growth': 'cpi_indonesia',
        'Italy_CPI_Growth': 'cpi_italy',
        'Japan_CPI_Growth': 'cpi_japan',
        'Korea_Rep._of_CPI_Growth': 'cpi_south_korea',
        'Mexico_CPI_Growth': 'cpi_mexico',
        'Russian_Federation_CPI_Growth': 'cpi_russia',
        'Saudi_Arabia_CPI_Growth': 'cpi_saudi_arabia',
        'South_Africa_CPI_Growth': 'cpi_south_africa',
        'Türkiye_Rep_of_CPI_Growth': 'cpi_turkey',
        'United_Kingdom_CPI_Growth': 'cpi_uk',
        'United_States_CPI_Growth': 'cpi_us',
        'Argentina_Interest_Rate': 'interest_rate_argentina',
        'Australia_Interest_Rate': 'interest_rate_australia',
        'Brazil_Interest_Rate': 'interest_rate_brazil',
        'Canada_Interest_Rate': 'interest_rate_canada',
        'China_P.R.:_Mainland_Interest_Rate': 'interest_rate_china',
        'India_Interest_Rate': 'interest_rate_india',
        'Indonesia_Interest_Rate': 'interest_rate_indonesia',
        'Japan_Interest_Rate': 'interest_rate_japan',
        'Korea_Rep._of_Interest_Rate': 'interest_rate_south_korea',
        'Mexico_Interest_Rate': 'interest_rate_mexico',
        'Russian_Federation_Interest_Rate': 'interest_rate_russia',
        'Saudi_Arabia_Interest_Rate': 'interest_rate_saudi_arabia',
        'South_Africa_Interest_Rate': 'interest_rate_south_africa',
        'Türkiye_Rep_of_Interest_Rate': 'interest_rate_turkey',
        'United_Kingdom_Interest_Rate': 'interest_rate_uk',
        'United_States_Interest_Rate': 'interest_rate_us',
        'Australia_Exchange_Rate': 'exchange_rate_australia',
        'Brazil_Exchange_Rate': 'exchange_rate_brazil',
        'Canada_Exchange_Rate': 'exchange_rate_canada',
        'China_P.R.:_Mainland_Exchange_Rate': 'exchange_rate_china',
        'France_Exchange_Rate': 'exchange_rate_france',
        'Germany_Exchange_Rate': 'exchange_rate_germany',
        'Italy_Exchange_Rate': 'exchange_rate_italy',
        'Japan_Exchange_Rate': 'exchange_rate_japan',
        'Mexico_Exchange_Rate': 'exchange_rate_mexico',
        'Russian_Federation_Exchange_Rate': 'exchange_rate_russia',
        'Saudi_Arabia_Exchange_Rate': 'exchange_rate_saudi_arabia',
        'South_Africa_Exchange_Rate': 'exchange_rate_south_africa',
        'United_Kingdom_Exchange_Rate': 'exchange_rate_uk',
        'United_States_Exchange_Rate': 'exchange_rate_us',
        'Aluminum': 'commodity_aluminum',
        'Brent Crude': 'commodity_brent_crude',
        'Cocoa': 'commodity_cocoa',
        'Coffee': 'commodity_coffee',
        'Copper': 'commodity_copper',
        'Corn': 'commodity_corn',
        'Cotton': 'commodity_cotton',
        'Feeder Cattle': 'commodity_feeder_cattle',
        'Gas Oil': 'commodity_gas_oil',
        'Gold': 'commodity_gold',
        'ULS Diesel': 'commodity_uls_diesel',
        'Lead': 'commodity_lead',
        'Lean Hogs': 'commodity_lean_hogs',
        'Live Cattle': 'commodity_live_cattle',
        'Natural Gas': 'commodity_natural_gas',
        'Nickel': 'commodity_nickel',
        'Orange Juice': 'commodity_orange_juice',
        'Platinum': 'commodity_platinum',
        'Silver': 'commodity_silver',
        'Soybean Meal': 'commodity_soybean_meal',
        'Soybean Oil': 'commodity_soybean_oil',
        'Soybeans': 'commodity_soybeans',
        'Sugar': 'commodity_sugar',
        'Tin': 'commodity_tin',
        'Unleaded Gas': 'commodity_unleaded_gas',
        'Wheat': 'commodity_wheat',
        'Kansas Wheat': 'commodity_kansas_wheat',
        'WTI Crude Oil': 'commodity_wti_crude_oil',
        'Zinc': 'commodity_zinc'
    }
    asset_data.rename(columns=rename_columns, inplace=True)
    asset_data.to_csv('all_data.csv', index=False)