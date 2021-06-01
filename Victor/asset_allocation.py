import os
import requests
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

@st.cache
def mean_variance(prices, objective, percentage):
    S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
    mu = expected_returns.mean_historical_return(prices)
    
    ef = EfficientFrontier(mu, S) 
    ef.add_objective(objective_functions.L2_reg, gamma=0.1) 
    
    if(objective == "Maximize Sharpe Ratio"):
        ef.max_sharpe()
    elif(objective == "Maximize Return for given level of Risk"):    
        ef.efficient_risk(percentage/100)
    else:
        ef.efficient_return(percentage/100)
        
    weights = ef.clean_weights()
    
    port_perf = ef.portfolio_performance(verbose=True);
    
    #pd.Series(weights_maxsharpe).plot.pie(figsize=(10,10));
    weights_df = pd.DataFrame(weights.values(), weights.keys())
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
    if(objective == "Maximize Sharpe Ratio"):
        ef.max_sharpe()
    elif(objective == "Maximize Return for given level of Risk"):    
        ef.efficient_risk(percentage/100)
    else:
        ef.efficient_return(percentage/100)
        
    ret_tangent, std_tangent, _ = ef.portfolio_performance()
    ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label=objective)

    # Plot random portfolios
    ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")

    # Format
    ax.set_title("Efficient Frontier")
    ax.legend()
    plt.tight_layout()
    #plt.show()
    
    return weights_df, port_perf, plt

@st.cache
def monte_carlo(portfolio_data, weights):

    simulation_data = MCSimulation(
        portfolio_data = portfolio_data,
        weights = weights,
        num_simulation = 100,
        num_trading_days = 252
    )
    simulation_data.calc_cumulative_return()
    line_plot = simulation_data.plot_simulation()
    tbl = simulation_data.summarize_cumulative_return()

    return line_plot, tbl
