import os
import requests
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import plotly.express as px
from MCForecastTools import MCSimulation
import panel as pn
import seaborn as sns
import streamlit as st

import pypfopt
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import EfficientFrontier
from pypfopt import CLA, plotting
from pypfopt import objective_functions

tickers = ["AMC","AMD","BABA","BB","BBBY","GME","MVIS","NVDA","TSLA","BTC","BCH","ETH","ETC","LTC","XRP","DOGE"]

#containers
header = st.beta_container()
RetAndCorr = st.beta_container()
Markowitz = st.beta_container()
BlackLitterman = st.beta_container()

#data set
crypto_prices = pd.read_csv('../Resources/crypto_prices.csv',parse_dates=True,index_col='Date',infer_datetime_format=True)
#st.write(crypto_prices.head())
stock_prices = pd.read_csv('../Resources/stock_prices.csv',parse_dates=True,index_col='Date',infer_datetime_format=True)
#st.write(stock_prices.head())
all_prices = crypto_prices.join(stock_prices)
all_prices = all_prices.drop(columns="SPY")
all_prices_clean = all_prices.dropna()
st.write(all_prices_clean.head())


#cache and reading data
@st.cache
def get_data(filename):
    data = pd.read_csv(filename)
    return data

with header:
    st.title("#Welcome to YOLO advisors Inc")


with RetAndCorr:
    st.header("Investment Universe")
    st.text("Choose the stocks and cryptos you want to invest in")
    assets = tickers
    select_asset = st.multiselect('select investments', assets, key="1")
    
    df_select_asset = all_prices_clean[select_asset]
    
    st.write('prices_asset'+str(df_select_asset.shape[0]))
    st.write(df_select_asset)
    
    #S = risk_models.CovarianceShrinkage(df_select_stock).ledoit_wolf()
    #plotting.plot_covariance(S, plot_correlation=True)
    
    
with Markowitz:
    st.header("Markowitz")
    st.text("description....")
    
    choose = tickers
    selected_asset = st.multiselect('select investments', choose, key="2")
    
    df_selected_asset = all_prices_clean[selected_asset]
    df_total_min_max = st.selectbox('Max or not',options=["Maxmize","Minimize"])
    
    if df_total_min_max == "Minimize":
        percentage = st.number_input('percentage',min_value=0.00,value=0.027,step=0.01,key="7")
    else:
        pass
    
    
with BlackLitterman:
    st.header("Black Litterman")
    col1, col2, col3, col4, col5 = st.beta_columns ([2,2,2,2.8,2.8])
    B1=col1.selectbox('select',options=tickers,key="1")
    B2=col2.selectbox('your view',options=["will outperform","will underperform"],key="2")
    B3=col3.selectbox('select',options=tickers,key="3")
    B4=col4.number_input('percentage',min_value=0.00,value=0.027,step=0.01,key="1")
    B5=col5.number_input('confidence level',min_value=1.0, max_value=100.0, value=20.0,step=4.0,key="2")
    
    col1, col2, col3, col4, col5 = st.beta_columns ([2,2,2,2.8,2.8])
    B1=col1.selectbox('select',options=tickeres,key="2")
    B2=col2.selectbox('your view',options=["will outperform","will underperform"],key="3")
    B3=col3.selectbox('select',options=tickers,key="4")
    B4=col4.number_input('percentage',min_value=0.00,value=0.027,step=0.01,key="2")
    B5=col5.number_input('confidence level',min_value=1.0, max_value=100.0, value=20.0,step=4.0,key="3")
    
    col1, col2, col3, col4, col5 = st.beta_columns ([2,2,2,2.8,2.8])
    B1=col1.selectbox('select',options=tickers,key="3")
    B2=col2.selectbox('your view',options=["will outperform","will underperform"],key="4")
    B3=col3.selectbox('select',options=tickers,key="5")
    B4=col4.number_input('percentage',min_value=0.00,value=0.027,step=0.01,key="3")
    B5=col5.number_input('confidence level',min_value=1.0, max_value=100.0, value=20.0,step=4.0,key="4")
    
    col1, col2, col3, col4, col5 = st.beta_columns ([2,2,2,2.8,2.8])
    B1=col1.selectbox('select',options=tickers,key="4")
    B2=col2.selectbox('your view',options=["will outperform","will underperform"],key="5")
    B3=col3.selectbox('select',options=tickers,key="6")
    B4=col4.number_input('percentage',min_value=0.00,value=0.027,step=0.01,key="4")
    B5=col5.number_input('confidence level',min_value=1.0, max_value=100.0, value=20.0,step=4.0,key="5")
    
    col1, col2, col3, col4, col5 = st.beta_columns ([2,2,2,2.8,2.8])
    B1=col1.selectbox('select',options=tickers,key="5")
    B2=col2.selectbox('your view',options=["will outperform","will underperform"],key="6")
    B3=col3.selectbox('select',options=tickers,key="7")
    B4=col4.number_input('percentage',min_value=0.00,value=0.027,step=0.01,key="5")
    B5=col5.number_input('confidence level',min_value=1.0, max_value=100.0, value=20.0,step=4.0,key="6")

    

 