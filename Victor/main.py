import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from asset_allocation import mean_variance, black_litterman_func, monte_carlo

tickers = ["AMC","AMD","BABA","BB","BBBY","GME","MVIS","NVDA","TSLA","BTC","BCH","ETH","ETC","LTC","XRP","DOGE"]

bl_column_width = [3,3,3,3,3]
select_assets = []
df_select_assets = []

st.set_page_config(layout="wide")

#containers
header = st.beta_container()
RetAndCorr = st.beta_container()
Markowitz = st.beta_container()
BlackLitterman = st.beta_container()

#data set
crypto_prices = pd.read_csv('Resources/crypto_prices.csv',parse_dates=True,index_col='Date',infer_datetime_format=True)
stock_prices = pd.read_csv('Resources/stock_prices.csv',parse_dates=True,index_col='Date',infer_datetime_format=True)
all_prices = crypto_prices.join(stock_prices)
market_prices = all_prices['SPY']
all_prices = all_prices.drop(columns="SPY")
all_prices_clean = all_prices.dropna()
#st.write(all_prices_clean.head())

mcaps = pd.read_csv('Resources/market_caps.csv')
mcaps = dict(mcaps.values)
del mcaps['SPY']

@st.cache
def build_views(B1, B2, B3, B4, B5):
    
    if B1 == " ":
        return np.array([]), np.array([]), np.array([])
    
    q_list = np.array([B4/100])
    c_list = np.array([B5/100])
    p_list = np.zeros((1, len(select_assets)))
    
    if(B2 == "will outperform"):
        p_list[0, select_assets.index(B1)] = 1
        if(B3 != " "):
            p_list[0, select_assets.index(B3)] = -1
    else:
        p_list[0, select_assets.index(B1)] = -1
        if(B3 != " "):
            p_list[0, select_assets.index(B3)] = 1
    
    return q_list, p_list, c_list


with header:
    st.title("#Welcome to YOLO Advisors Inc - your robo-advisor to avoid #FOMO")


with RetAndCorr:
    st.header("Investment Universe")
    st.text("Choose the stocks and cryptos you want to invest in")
    assets = tickers
    select_assets = st.multiselect('select investments', assets, key="1")
    df_select_assets = all_prices_clean[select_assets]
    
    if st.button('Display Historical Prices', key = "30"):
        st.line_chart(df_select_assets)    
    
with Markowitz:
    st.header("Markowitz Mean-Variance Optimization")
    
    col1, col2 = st.beta_columns([20, 5])
    objective = col1.selectbox('Objective',options=["Maximize Sharpe Ratio","Maximize Return for given level of Risk","Minimize Risk for given level of Return"])
    percentage = col2.number_input('% (not applicable for maximizing Sharpe)', min_value = 0.0, value = 70.0, step = 5.0)
    
    if st.button('Compute Optimal Portfolio', key = "31"):
    
        weights, port_perf, plt = mean_variance(df_select_assets, objective, percentage)
    
        st.write("Expected annual return: " + str(round(port_perf[0]*100,2)) + "%")
        st.write("Annual volatility: " + str(round(port_perf[1]*100,2)) + "%")
        st.write("Sharpe Ratio: " + str(round(port_perf[2],2)))

        pie_col, graph_col = st.beta_columns(2)
        fig = px.pie(weights, values = weights["weight"]*100, names = weights.index)
        pie_col.plotly_chart(fig)
        graph_col.pyplot(plt)
    
    #if st.button('Show Monte Carlo Simulation'):
#         plot_col, data_col = st.beta_columns(2)
#         portfolio_data = df_select_assets
#         input_tickers = df_select_assets.columns
#         column_names = [(x,"close") for x in input_tickers]
#         portfolio_data.columns = pd.MultiIndex.from_tuples(column_names)
#         mc_plt, mc_tbl = monte_carlo(portfolio_data, weights["weight"].values)
#         plot_col.pyplot(mc_plt)
#         data_col.write(tbl)
    
    
with BlackLitterman:
    st.header("Black-Litterman with Views")
    
    col1, col2, col3, col4, col5 = st.beta_columns (bl_column_width)
    B11=col1.selectbox('select',options=[" "] + select_assets,key = "4")
    B21=col2.selectbox('your view',options=["will outperform","will underperform"], key = "5")
    B31=col3.selectbox('select (empty for absolute view)',options=[" "] + select_assets, key = "6")
    B41=col4.number_input('percentage',min_value=0.00,value=0.0,step=1.0, key = "7")
    B51=col5.number_input('confidence level',min_value=1.0, max_value=100.0, value=50.0,step=5.0, key = "8")
    
    col1, col2, col3, col4, col5 = st.beta_columns (bl_column_width)
    B12=col1.selectbox('select',options=[" "] + select_assets, key = "9")
    B22=col2.selectbox('your view',options=["will outperform","will underperform"], key = "10")
    B32=col3.selectbox('select (empty for absolute view)',options=[" "] + select_assets, key = "11")
    B42=col4.number_input('percentage',min_value=0.00,value=0.0,step=1.0, key = "12")
    B52=col5.number_input('confidence level',min_value=1.0, max_value=100.0, value=50.0,step=5.0, key = "13")
    
    col1, col2, col3, col4, col5 = st.beta_columns (bl_column_width)
    B13=col1.selectbox('select',options=[" "] + select_assets, key = "14")
    B23=col2.selectbox('your view',options=["will outperform","will underperform"], key = "15")
    B33=col3.selectbox('select (empty for absolute view)',options=[" "] + select_assets, key = "16")
    B43=col4.number_input('percentage',min_value=0.00,value=0.0,step=1.0, key = "17")
    B53=col5.number_input('confidence level',min_value=1.0, max_value=100.0, value=50.0,step=5.0, key = "18")
    
    col1, col2, col3, col4, col5 = st.beta_columns (bl_column_width)
    B14=col1.selectbox('select',options=[" "] + select_assets, key = "19")
    B24=col2.selectbox('your view',options=["will outperform","will underperform"], key = "20")
    B34=col3.selectbox('select (empty for absolute view)',options=[" "] + select_assets, key = "21")
    B44=col4.number_input('percentage',min_value=0.00,value=0.0,step=1.0, key = "22")
    B54=col5.number_input('confidence level',min_value=1.0, max_value=100.0, value=50.0,step=5.0, key = "23")
    
    col1, col2, col3, col4, col5 = st.beta_columns (bl_column_width)
    B15=col1.selectbox('select',options=[" "] + select_assets, key = "24")
    B25=col2.selectbox('your view',options=["will outperform","will underperform"], key = "25")
    B35=col3.selectbox('select (empty for absolute view)',options=[" "] + select_assets, key = "26")
    B45=col4.number_input('percentage',min_value=0.00,value=0.0,step=1.0, key = "27")
    B55=col5.number_input('confidence level',min_value=1.0, max_value=100.0, value=50.0,step=5.0, key = "28")
    
    col1, col2 = st.beta_columns([20, 5])
    objective_bl = col1.selectbox('Objective',options=["Maximize Sharpe Ratio","Maximize Return for given level of Risk","Minimize Risk for given level of Return"], key = "29")
    percentage_bl = col2.number_input('% (not applicable for maximizing Sharpe)', min_value = 0.0, value = 70.0, step = 5.0, key = "30")
    
    if st.button ('Compute Optimal Portfolio', key = "32"):
    
        q_list1, p_list1, c_list1 = build_views(B11,B21,B31,B41,B51)
        q_list2, p_list2, c_list2 = build_views(B12,B22,B32,B42,B52)
        q_list3, p_list3, c_list3 = build_views(B13,B23,B33,B43,B53)
        q_list4, p_list4, c_list4 = build_views(B14,B24,B34,B44,B54)
        q_list5, p_list5, c_list5 = build_views(B15,B25,B35,B45,B55)

        Q = np.concatenate([x for x in [q_list1, q_list2, q_list3, q_list4, q_list5] if x.size > 0])
        P = np.concatenate([x for x in [p_list1, p_list2, p_list3, p_list4, p_list5] if x.size > 0])
        confidence = np.concatenate([x for x in [c_list1, c_list2, c_list3, c_list4, c_list5] if x.size > 0])    

        weights, port_perf, plt, rets_df = black_litterman_func(df_select_assets, market_prices, mcaps, select_assets, Q, P, confidence, objective_bl, percentage_bl)

        st.write("Expected annual return: " + str(round(port_perf[0]*100,2)) + "%")
        st.write("Annual volatility: " + str(round(port_perf[1]*100,2)) + "%")
        st.write("Sharpe Ratio: " + str(round(port_perf[2],2)))

        pie_col, graph_col = st.beta_columns(2)
        fig = px.pie(weights, values = weights["weight"]*100, names = weights.index)
        pie_col.plotly_chart(fig)
        graph_col.pyplot(plt) 

        return_bar = px.bar(rets_df, barmode = "group", labels = {"index":"", "value" : "Returns (%)"}, title = "Prior vs Posterior Expected Returns")
        st.plotly_chart(return_bar)
    
    #print(select_assets)
    #print(Q)
    #print(P)
    #print(confidence)

 