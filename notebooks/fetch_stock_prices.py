#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import pandas as pd
import yfinance as yf

# Load your dataset containing stock symbols and dates from a CSV file into a pandas DataFrame
df = pd.read_csv('raw_analyst_ratings.csv')

# Convert the 'date' column to datetime objects
df['date'] = pd.to_datetime(df['date'], utc=True)

# Group by 'stock' column and find the earliest and latest date for each stock
date_ranges = df.groupby('stock')['date'].agg(['min', 'max'])

# Fetch stock data for each stock within its respective date range
for stock, dates in date_ranges.iterrows():
    start_date = dates['min']
    end_date = dates['max']
    
    # Fetch stock data from Yahoo Finance using yfinance
    stock_data = yf.download(stock, start=start_date, end=end_date)
    
    # Display the fetched stock data
    print(f"\nStock: {stock}")
    print(stock_data.head())  # Display the first few rows of the fetched stock data


# In[ ]:




