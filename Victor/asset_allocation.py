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

from pypfopt import risk_models, expected_returns, EfficientFrontier, CLA, plotting, objective_functions, black_litterman, BlackLittermanModel

#@st.cache
def build_efficient_frontier(S, mu, objective, percentage):
    ef = EfficientFrontier(mu, S, weight_bounds=(0,1)) 
    ef.add_objective(objective_functions.L2_reg, gamma=0.1) 
    
    if(objective == "Maximize Sharpe Ratio"):
        ef.max_sharpe()
    elif(objective == "Maximize Return for given level of Risk"):    
        ef.efficient_risk(percentage/100)
    else:
        ef.efficient_return(percentage/100)
        
    weights = ef.clean_weights()
    
    port_perf = ef.portfolio_performance(verbose=True);
    
    weights_df = pd.DataFrame(weights.values(), weights.keys())
    weights_df.rename(columns = {0:"weight"}, inplace = True)
    
    n_samples = 10000
    w = np.random.dirichlet(np.ones(len(mu)), n_samples)
    rets = w.dot(mu)
    stds = np.sqrt((w.T * (S @ w.T)).sum(axis=0))
    sharpes = rets / stds
    
    # Draw Efficient frontier
    ef = EfficientFrontier(mu, S, weight_bounds = (0,1))
    ef.add_objective(objective_functions.L2_reg, gamma=0.1) 

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

#@st.cache
def mean_variance(prices, objective, percentage):
    S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
    mu = expected_returns.mean_historical_return(prices)
    
    weights_df, port_perf, plt = build_efficient_frontier(S, mu, objective, percentage)
    return weights_df, port_perf, plt
    

#@st.cache
def black_litterman_func(prices, market_prices, mcaps, select_assets, Q, P, confidences, objective, percentage):
    
    mcap_subset = {key: mcaps[key] for key in select_assets}
    
    S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
    delta = black_litterman.market_implied_risk_aversion(market_prices)
    market_prior = black_litterman.market_implied_prior_returns(mcap_subset, delta, S)
    
    bl = BlackLittermanModel(S, pi=market_prior, omega="idzorek", view_confidences=confidences, Q = Q, P = P)
    ret_bl = bl.bl_returns()
    rets_df = pd.DataFrame([market_prior*100, ret_bl*100], index=["Prior", "Posterior"]).T
    S_bl = bl.bl_cov()
    
    weights_df, port_perf, plt = build_efficient_frontier(S_bl, ret_bl, objective, percentage)
    
    return weights_df, port_perf, plt, rets_df
    

#@st.cache
def monte_carlo(portfolio_data, weights):

    simulation_data = MCSimulation(
        portfolio_data = portfolio_data,
        weights = weights,
        num_simulation = 100,
        num_trading_days = 252
    )
    simulation_data.calc_cumulative_return()
    fig, ax = plt.subplots() 
    fig = simulation_data.plot_simulation()
    tbl = simulation_data.summarize_cumulative_return()

    return fig, tbl
