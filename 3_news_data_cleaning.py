import pandas as pd 
import numpy as np

news_df = pd.read_csv("news1.csv")
stock_df = pd.read_csv("stock_price1.csv")

for i in range(len(stock_df)):
    date = stock_df['Date'][i][:10]
    # stock_df['Date'][i] = date
    stock_df.loc[i, 'Date'] = date


#change date format from 2024-03-27 to 27-03-2024
for i in range(len(news_df)):
    date = news_df['Date'][i]
    year, month, day = date.split("-")
    news_df.loc[i, 'Date'] = f"{day}-{month}-{year}"

# print(news_df.head())
# print()
# print(stock_df['Date'].tolist())

news_df = news_df[news_df['Date'].isin(stock_df['Date'].tolist())]

news_df.to_csv("news_data1.csv", index=False)