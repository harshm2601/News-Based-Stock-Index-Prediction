import pandas as pd
import yfinance as yf

def download_stock_data(ticker, start, end):
    """
    download stock price data from Yahoo Finance
    """
    stock_data = yf.download(ticker, start, end , auto_adjust=False)
    df = pd.DataFrame(stock_data)
    df.to_csv("stock_price1.csv")


download_stock_data("NDX", "2024-03-27", "2025-03-27")