import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

import scipy.stats as ss
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression

import sys

sys.path.insert(0, "../load_data")
from load import load_stats

pd.options.mode.chained_assignment = None


# ----- Load Data -----
players = load_stats()
players_train = pd.read_csv("./players_training_set.csv")
history_train = pd.read_csv("./history_training_set.csv")

tj = players_train[players_train["Top Junior"] == True]
ntj = players_train[players_train["Top Junior"] == False]


# ----- Logarithmic Regression Analysis -----

"""
method log_reg
    logarithmic regression of player rating graph, returns a in aln(bx)
    
    parameters:
        player - player to run regression on
        n_tours - run regression on first n tournaments
        plot - to plot the regression
        axes - axes in which plot is drawn

    returns:
        a - logarithmic coefficient
"""


def log_reg(player, n_tours=-1, plot=False, axes=None):
    rhist = history_train[
        history_train["Database#"] == player["Database#"]
    ].reset_index(drop=True)
    data = rhist[["days played", "Initial Rating"]]
    if n_tours >= 0:
        data = data[-n_tours:]
    data["log days played"] = np.log1p(data["days played"])

    reg = ss.linregress(data["log days played"], data["Initial Rating"])
    a = reg.slope

    if plot:
        b = np.exp(reg.intercept / reg.slope)

        # plot log correlation graph
        ax = axes[1]
        sns.scatterplot(
            data=data,
            x="log days played",
            y="Initial Rating",
            ax=axes[1],
            label="Rating History (log)",
            color="#35b779",
        )
        sns.regplot(
            data=data,
            x="log days played",
            y="Initial Rating",
            label=f"r: {round(reg.rvalue, 4)}",
            ax=ax,
            scatter=False,
        )
        ax.set_xlabel("Days Played (log)")
        ax.set_title("Strength of Correlation between Data and Model")
        ax.legend()

        # plot log regression model
        x = np.linspace(data["days played"].min(), data["days played"].max(), 60)
        y = a * np.log1p(b * x)

        ax = axes[0]
        data.plot(
            kind="line",
            x="days played",
            y="Initial Rating",
            ax=ax,
            label="USATT Rating",
            color="#35b779",
        )
        sns.lineplot(x=x, y=y, ax=ax, label=f"Logarithmic Coefficient: {round(a, 4)}")
        ax.set_xlabel("Days Played")
        ax.set_title("Logarithmic Progression Model")
        ax.set_ylim(0, None)
        ax.legend()

    return a


"""
method log_coef_func
    a normal distribution curve that defines the distribution of log coefficients
    
    parameters:
        x - input into function (log coef)
        scale - scale of normal distribution
        mu - mean of normal distribution
        sigma - standard deviation of normal distribution

    returns:
        the estimated number of players with the given log coefficient
"""


def log_coef_func(x, scale, mu, sigma):
    return scale * np.exp(-0.5 * ((x / 100 - mu) / sigma) ** 2)


"""
method log_coef_dist
    fit equation of normal distribution over log coef data for first n tournaments of every player

    parameters:
        data - population to fit equation for
        bins - how many bins to cut population into for distribution
        n_tours - fit equation on first n tournaments
        plot - to plot the distribution
        ax - axes in which to plot distribution
        label - label of distribution on graph

    returns:
        parameters - parameters of normal distribution equation
        data - population data with added features
"""


# fit equation of normal distribution over log coef data for first n tournaments of every player
def log_coef_dist(data, bins, n_tours=-1, plot=False, ax=None, label=""):
    data["log coef"] = data.apply(log_reg, axis=1, args=(n_tours, False))
    data["log coef bin"] = pd.cut(data["log coef"], bins)
    data_coefs = (
        data[["log coef", "log coef bin"]]
        .groupby("log coef bin", observed=True)
        .count()
    )
    data_coefs.index = pd.IntervalIndex(data_coefs.index)

    parameters, covariance = curve_fit(
        log_coef_func, data_coefs.index.mid, data_coefs["log coef"]
    )

    if plot:
        sns.histplot(data=data, x="log coef", bins=bins, alpha=0.5, label=label, ax=ax)
        x = np.linspace(data["log coef"].min(), data["log coef"].max(), 100)
        scale, mu, sigma = parameters
        y = log_coef_func(x, scale, mu, sigma)
        sns.lineplot(x=x, y=y, ax=ax)
        ax.set_xlabel("Logarithmic Coefficient")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Logarithmic Coefficients by Top Junior Class")

    return parameters, data


"""
method top_junior_prob_func
    probability function of becoming top junior given log coef

    parameters:
        log_coef - input logarithmic coefficient
        tj_params - parameters for top junior normal distribution
        ntj_params - parameters for not top junior normal distribution
    
    returns:
        probability that player with given logarithmic coefficient will become a top junior
"""


def top_junior_prob_func(log_coef, tj_params, ntj_params):
    tj_scale, tj_mu, tj_sigma = tj_params
    ntj_scale, ntj_mu, ntj_sigma = ntj_params

    tj_prob = log_coef_func(log_coef, tj_scale, tj_mu, tj_sigma)
    ntj_prob = log_coef_func(log_coef, ntj_scale, ntj_mu, ntj_sigma)

    return tj_prob / (tj_prob + ntj_prob)


"""
method log_reg_analysis
    graph previous visualizations about a player's regression model and calculate probability that player will become a top junior

    parameters:
        player - player on which to run logarithmic regression analysis
    
    returns:
        player_prob - probability that given player will become a top junior
"""


def log_reg_analysis(player):
    rhist = history_train[history_train["Database#"] == player["Database#"]]
    name = f"{player['First Name']} {player['Last Name']}"
    n_tours = len(rhist)

    # make entire figure
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # plot player logarithmic regression model
    player_log_coef = log_reg(player, plot=True, axes=axes[1])

    # plot distribution of log coefs for first n tournaments
    ax = axes[0][0]
    ntj_params, ntj_n_tours = log_coef_dist(
        data=ntj, bins=10, label="Not Top Juniors", n_tours=n_tours, plot=True, ax=ax
    )
    tj_params, tj_n_tours = log_coef_dist(
        data=tj, bins=10, label="Top Juniors", n_tours=n_tours, plot=True, ax=ax
    )
    ax.axvline(
        x=player_log_coef,
        label=name,
        linestyle="--",
        color="#35b779",
        alpha=0.75,
    )
    ax.legend()

    # plot the top junior probability function with player point drawn
    min_coef = min(ntj_n_tours["log coef"].min(), tj_n_tours["log coef"].min())
    max_coef = max(ntj_n_tours["log coef"].max(), tj_n_tours["log coef"].max())

    x = np.linspace(min_coef, max_coef, 100)
    y = top_junior_prob_func(x, tj_params, ntj_params)

    player_prob = top_junior_prob_func(player_log_coef, tj_params, ntj_params)
    fig.suptitle(
        f"{name}'s Progression Analysis\nTop Junior Probability: {round(player_prob * 100, 2)}%"
    )

    ax = axes[0][1]
    sns.lineplot(x=x, y=y, label="Probability Function", ax=ax)
    sns.scatterplot(
        x=[player_log_coef],
        y=[player_prob],
        marker="o",
        s=100,
        label=name,
        color="#35b779",
        alpha=0.75,
        ax=ax,
    )
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_xlabel("Logarithmic Coefficient")
    ax.set_ylabel("Probability")
    ax.set_title("Probability of Becoming a Top Junior by Logarithmic Coefficient")
    ax.legend()

    plt.tight_layout()
    plt.show()

    return player_prob


me = players[players["USATT#"] == 280573].iloc[0]
log_reg_analysis(me)
