import yfinance as yf
import pandas as pd
import numpy as np

def get_oil_data(symbol='CL=F', period='1y', interval='1d'):
    oil = yf.Ticker(symbol)
    df = oil.history(period=period, interval=interval)
    return df[['Close']].rename(columns={'Close': 'Oil Price'})
if __name__ == "__main__":
    print(get_oil_data().tail())

def calculate_annualized_volatility(df, window=30):
    # Calculate log returns
    df['Log Return'] = np.log(df['Oil Price'] / df['Oil Price'].shift(1))
    
    # Drop NaNs
    df = df.dropna()

    # Daily volatility (std dev of returns)
    daily_vol = df['Log Return'].rolling(window=window).std()
    
    # Annualized volatility = daily volatility * sqrt(252 trading days)
    annualized_vol = daily_vol.iloc[-1] * np.sqrt(252)
    return round(annualized_vol, 4)