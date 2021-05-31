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

#@st.cache
def mean_variance(prices, objective):
    S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
    mu = expected_returns.mean_historical_return(prices)
    
    ef = EfficientFrontier(mu, S) 
    ef.max_sharpe()
    weights_maxsharpe = ef.clean_weights()
    
    port_perf = ef.portfolio_performance(verbose=True);
    
    #pd.Series(weights_maxsharpe).plot.pie(figsize=(10,10));
    weights_df = pd.DataFrame(weights_maxsharpe.values(), weights_maxsharpe.keys())
    weights_df.rename(columns = {0:"weight"}, inplace = True)
    
    n_samples = 10000
    w = np.random.dirichlet(np.ones(len(mu)), n_samples)
    rets = w.dot(mu)
    stds = np.sqrt((w.T * (S @ w.T)).sum(axis=0))
    sharpes = rets / stds
    
    # Draw Efficient frontier
    ef = EfficientFrontier(mu, S)

    fig, ax = plt.subplots()
    plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False)

    # Find and plot the tangency portfolio
    ef.max_sharpe()
    ret_tangent, std_tangent, _ = ef.portfolio_performance()
    ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Max Sharpe")

    # Plot random portfolios
    ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")

    # Format
    ax.set_title("Efficient Frontier with random portfolios")
    ax.legend()
    plt.tight_layout()
    #plt.show()
    
    return weights_df, port_perf, plt