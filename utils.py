import pandas as pd
import numpy as np

def process_asset_data(ticker, data):
    df = pd.DataFrame(data['Close'])
    df.reset_index(inplace=True)
    df.rename(columns={ticker:'Close'},inplace=True)
    df['Change in asset returns'] = df['Close']/df['Close'].shift(1)

    return df


def process_treasury_data(treasury_data):
    df_treasury = pd.DataFrame(treasury_data['Close'])
    df_treasury.columns = ['3M', '5Y', '10Y', '30Y']
    df_treasury.reset_index(inplace=True)

    return df_treasury

def calculate_rolling_volatility(prices, window=21, annualize=True, trading_days=252):
    """
    Calculate rolling volatility from a series of stock prices.
    
    Parameters:
    -----------
    prices : array-like or pandas Series
        Time series of stock prices
    window : int, default 21
        Rolling window size (typically 21 for monthly)
    method : str, default 'returns'
        Method to calculate volatility:
        - 'returns': var of returns
    annualize : bool, default True
        Whether to annualize the volatility
    trading_days : int, default 252
        Number of trading days in a year, used for annualization
        
    Returns:
    --------
    pandas.Series
        Rolling volatility series
    """
    # Convert to pandas Series if not already
    prices = pd.Series(prices)
    
    # Calculate simple returns
    returns = prices.pct_change()
    # Calculate rolling standard deviation
    rolling_std = returns.rolling(window=window).var()

    
    # Annualize if requested
    if annualize:
        rolling_vol = rolling_std * np.sqrt(trading_days)
    else:
        rolling_vol = rolling_std
        
    return rolling_vol