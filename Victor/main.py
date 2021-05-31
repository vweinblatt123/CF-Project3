# import os
# import requests
# import yfinance as yf
# import numpy as np
# import pandas as pd
# #import matplotlib.pyplot as plt
# #import plotly.express as px
# from MCForecastTools import MCSimulation
# #import panel as pn
# #import seaborn as sns
# import streamlit as st
# import altair as alt

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt

import pypfopt
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import EfficientFrontier
from pypfopt import CLA, plotting
from pypfopt import objective_functions

from asset_allocation import mean_variance

tickers = ["AMC","AMD","BABA","BB","BBBY","GME","MVIS","NVDA","TSLA","BTC","BCH","ETH","ETC","LTC","XRP","DOGE"]

bl_column_width = [3,3,3,3,3]
df_select_assets = []

st.set_page_config(layout="wide")

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
    st.title("#Welcome to YOLO Advisors Inc")


with RetAndCorr:
    st.header("Investment Universe")
    st.text("Choose the stocks and cryptos you want to invest in")
    assets = tickers
    select_assets = st.multiselect('select investments', assets, key="1")
    
    df_select_assets = all_prices_clean[select_assets]
    
    #st.write('prices_asset'+str(df_select_asset.shape[0]))
    #st.write(df_select_asset)
    
    #S = risk_models.CovarianceShrinkage(df_select_asset).ledoit_wolf()
    #plotting.plot_covariance(S, plot_correlation=True)
    chart = st.line_chart(df_select_assets)    
    
with Markowitz:
    st.header("Markowitz Mean-Variance Optimization")
    
    col1, col2 = st.beta_columns([20, 5])
    objective = col1.selectbox('Objective',options=["Maximize Sharpe Ratio","Maximize Return for given level of Risk","Minimize Risk for given level of Return"])
    percentage = col2.number_input('% (not applicable for maximizing Sharpe)', min_value = 0.0, value = 0.0, step = 1.0)
    
    weights, port_perf, plt = mean_variance(df_select_assets, "obj")
    
    graph_col, pie_col = st.beta_columns(2)
    fig = px.pie(weights, values = weights["weight"]*100, names = weights.index)
    pie_col.plotly_chart(fig)
    graph_col.pyplot(plt);
    
    
with BlackLitterman:
    st.header("Black Litterman")
    col1, col2, col3, col4, col5 = st.beta_columns (bl_column_width)
    B1=col1.selectbox('select',options=tickers,key = "4")
    B2=col2.selectbox('your view',options=["will outperform","will underperform"], key = "5")
    B3=col3.selectbox('select',options=tickers, key = "6")
    B4=col4.number_input('percentage',min_value=0.00,value=0.0,step=1.0, key = "7")
    B5=col5.number_input('confidence level',min_value=1.0, max_value=100.0, value=50.0,step=5.0, key = "8")
    
    col1, col2, col3, col4, col5 = st.beta_columns (bl_column_width)
    B1=col1.selectbox('select',options=tickers, key = "9")
    B2=col2.selectbox('your view',options=["will outperform","will underperform"], key = "10")
    B3=col3.selectbox('select',options=tickers, key = "11")
    B4=col4.number_input('percentage',min_value=0.00,value=0.0,step=1.0, key = "12")
    B5=col5.number_input('confidence level',min_value=1.0, max_value=100.0, value=50.0,step=5.0, key = "13")
    
    col1, col2, col3, col4, col5 = st.beta_columns (bl_column_width)
    B1=col1.selectbox('select',options=tickers, key = "14")
    B2=col2.selectbox('your view',options=["will outperform","will underperform"], key = "15")
    B3=col3.selectbox('select',options=tickers, key = "16")
    B4=col4.number_input('percentage',min_value=0.00,value=0.0,step=1.0, key = "17")
    B5=col5.number_input('confidence level',min_value=1.0, max_value=100.0, value=50.0,step=5.0, key = "18")
    
    col1, col2, col3, col4, col5 = st.beta_columns (bl_column_width)
    B1=col1.selectbox('select',options=tickers, key = "19")
    B2=col2.selectbox('your view',options=["will outperform","will underperform"], key = "20")
    B3=col3.selectbox('select',options=tickers, key = "21")
    B4=col4.number_input('percentage',min_value=0.00,value=0.0,step=1.0, key = "22")
    B5=col5.number_input('confidence level',min_value=1.0, max_value=100.0, value=50.0,step=5.0, key = "23")
    
    col1, col2, col3, col4, col5 = st.beta_columns (bl_column_width)
    B1=col1.selectbox('select',options=tickers, key = "24")
    B2=col2.selectbox('your view',options=["will outperform","will underperform"], key = "25")
    B3=col3.selectbox('select',options=tickers, key = "26")
    B4=col4.number_input('percentage',min_value=0.00,value=0.0,step=1.0, key = "27")
    B5=col5.number_input('confidence level',min_value=1.0, max_value=100.0, value=50.0,step=5.0, key = "28")

    

 