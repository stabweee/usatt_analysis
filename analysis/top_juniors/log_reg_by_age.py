import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

import scipy.stats as ss
from scipy.optimize import least_squares
from sklearn.linear_model import LinearRegression

from load_data.load import load_stats

pd.options.mode.chained_assignment = None


# ----- Load Data -----
players = load_stats()
players_train = pd.read_csv("top_juniors/players_training_set.csv")
history_train = pd.read_csv("top_juniors/history_training_set.csv")

history = history_train

tj = players_train[players_train["Top Junior"] == True]
ntj = players_train[players_train["Top Junior"] == False]


# ----- Logarithmic Regression Analysis -----

"""
method log_reg
    logarithmic regression of player rating graph, returns a in aln(bx)
    
    parameters:
        player - player to run regression on
        age - run regression on tournaments before age
        plot - to plot the regression
        axes - axes in which plot is drawn

    returns:
        a - logarithmic coefficient
"""


def log_reg(player, history, age=-1, plot=False, axes=None):
    rhist = history[history["Database#"] == player["Database#"]].reset_index(drop=True)
    data = rhist[["days played", "tournament age", "Initial Rating"]]
    if age >= 0:
        data = data[data["tournament age"] < age]
    if len(data) < 5:
        return pd.NA
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
            color="#CC6677",
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
        ax.set_ylabel("Rating")
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
            label="USATT Rating Progression",
            color="#CC6677",
            marker="o",
            markersize=3,
        )
        sns.lineplot(
            x=x,
            y=y,
            ax=ax,
            label=f"Logarithmic Coefficient: {round(a, 4)}",
        )
        ax.set_xlabel("Days Played")
        ax.set_ylabel("Rating")
        ax.set_title("Logarithmic Progression Model")
        ax.set_ylim(0, None)
        ax.legend()

    return a


# logarithmic regression of 90 day window moving average of player rating graph, returns a, b in aln(bx)
def rolling_log_reg(player, history, window=90, age=-1, plot=False, axes=None):
    rhist = history[history["Database#"] == player["Database#"]].reset_index(drop=True)
    data = rhist[["days played", "tournament age", "Initial Rating"]]
    if age >= 0:
        data = data[data["tournament age"] < age]
    if len(data) < 5:
        return pd.NA

    data = data.drop("tournament age", axis=1)
    data = data.set_index("days played")
    data.index = pd.to_timedelta(data.index, unit="days")
    rolldata = data.rolling(f"{window}d").mean().reset_index(names="days played")
    rolldata["days played"] = rolldata["days played"].dt.days
    rolldata["log days played"] = np.log1p(rolldata["days played"])

    reg = ss.linregress(rolldata["log days played"], rolldata["Initial Rating"])
    a = reg.slope

    if plot:
        b = np.exp(reg.intercept / reg.slope)

        # plot log correlation graph
        ax = axes[1]
        sns.scatterplot(
            data=rolldata,
            x="log days played",
            y="Initial Rating",
            ax=axes[1],
            label="Rating History (rolling log)",
            color="#CC6677",
        )
        sns.regplot(
            data=rolldata,
            x="log days played",
            y="Initial Rating",
            label=f"r: {round(reg.rvalue, 4)}",
            ax=ax,
            scatter=False,
        )
        ax.set_xlabel("Days Played (log)")
        ax.set_ylabel("Rating")
        ax.set_title("Strength of Correlation between Data and Model")
        ax.legend()

        # plot log regression model
        x = np.linspace(
            rolldata["days played"].min(), rolldata["days played"].max(), 60
        )
        y = a * np.log1p(b * x)

        ax = axes[0]
        rolldata.plot(
            kind="line",
            x="days played",
            y="Initial Rating",
            ax=ax,
            label="USATT Rating Progression",
            color="#CC6677",
            marker="o",
            markersize=3,
        )
        sns.lineplot(
            x=x,
            y=y,
            ax=ax,
            label=f"Rolling Logarithmic Coefficient: {round(a, 4)}",
        )
        ax.set_xlabel("Days Played")
        ax.set_ylabel("Rating")
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


# Wrap it to compute residuals
def residuals(params, x, y):
    return log_coef_func(x, *params) - y


"""
method log_coef_dist
    fit equation of normal distribution over log coef data for first n tournaments of every player

    parameters:
        data - population to fit equation for
        bins - how many bins to cut population into for distribution
        age - fit equation on tournaments before age
        plot - to plot the distribution
        ax - axes in which to plot distribution
        label - label of distribution on graph

    returns:
        parameters - parameters of normal distribution equation
        data - population data with added features
"""


# fit equation of normal distribution over log coef data for all tournaments before age of every player
def log_coef_dist(data, bins, age=-1, rolling=False, plot=False, ax=None, label=""):
    if rolling:
        data["log coef"] = data.apply(
            rolling_log_reg, axis=1, args=(history_train, 90, age, False)
        )
    else:
        data["log coef"] = data.apply(log_reg, axis=1, args=(history_train, age, False))

    data = data.dropna(subset=["log coef"])

    # ensure enough data points in sample
    if len(data) < 15:
        print("insufficient sample size for analysis")
        return False, False

    data["log coef bin"] = pd.cut(data["log coef"], bins)
    data_coefs = (
        data[["log coef", "log coef bin"]]
        .groupby("log coef bin", observed=True)
        .count()
    )
    data_coefs.index = pd.IntervalIndex(data_coefs.index)

    results = least_squares(
        residuals,
        x0=[1, 1, 1],
        args=(data_coefs.index.mid, data_coefs["log coef"]),
        max_nfev=1000,
    )
    parameters = results.x

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


def log_reg_analysis(player, history, rolling=False, plot=False, bar=None):
    rhist = history[history["Database#"] == player["Database#"]].reset_index(drop=True)
    if len(rhist) < 5:
        print("player has not played enough tournaments to perform an analysis")
        return -1

    name = f"{player['First Name']} {player['Last Name']}"
    age = rhist.loc[0, "tournament age"]

    # make entire figure
    if plot:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    else:
        axes = np.zeros((2, 2))

    # plot player logarithmic regression model
    if rolling:
        player_log_coef = rolling_log_reg(
            player, history, window=90, plot=plot, axes=axes[1]
        )
    else:
        player_log_coef = log_reg(player, history, plot=plot, axes=axes[1])

    # plot distribution of log coefs for first n tournaments
    ax = axes[0][0]
    ntj_params, ntj_age = log_coef_dist(
        data=ntj,
        bins=10,
        rolling=rolling,
        label="Not Top Juniors",
        age=age,
        plot=plot,
        ax=ax,
    )
    if type(ntj_params) == bool:
        if bar is not None:
            bar()
        return -1

    tj_params, tj_age = log_coef_dist(
        data=tj,
        bins=10,
        rolling=rolling,
        label="Top Juniors",
        age=age,
        plot=plot,
        ax=ax,
    )
    if type(tj_params) == bool:
        if bar is not None:
            bar()
        return -1

    # plot the top junior probability function with player point drawn
    min_coef = min(ntj_age["log coef"].min(), tj_age["log coef"].min())
    max_coef = max(ntj_age["log coef"].max(), tj_age["log coef"].max())

    x = np.linspace(min_coef, max_coef, 100)
    y = top_junior_prob_func(x, tj_params, ntj_params)

    player_prob = top_junior_prob_func(player_log_coef, tj_params, ntj_params)

    if plot:
        ax.axvline(
            x=player_log_coef,
            label=name,
            linestyle="--",
            color="#CC6677",
            alpha=0.75,
        )
        ax.legend()

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
            color="#CC6677",
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

    if bar is not None:
        bar()

    return player_prob


# ----- Test -----
from alive_progress import alive_bar
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

history_test = history_train[history_train["tournament age"] < 10]
tournament_counts = history_test["USATT#"].value_counts(ascending=True)
valid_players = tournament_counts[tournament_counts >= 5]

history_test = history_test[history_test["USATT#"].isin(valid_players.index)]
players_test = players_train[players_train["USATT#"].isin(history_test["USATT#"])]

with alive_bar(len(players_test)) as bar:
    test_probs = players_test.apply(
        log_reg_analysis, axis=1, args=(history_test, False, False, bar)
    )

y_pred = np.where(test_probs < 0.5, 0, 1)
y_true = players_test["Top Junior"]
y_test = pd.DataFrame({"true": y_true, "pred": y_pred})
y_test = y_test[y_test["pred"] != -1]

print(f"accuracy: {accuracy_score(y_test["true"], y_test["pred"])}")
print(f"precision: {precision_score(y_test["true"], y_test["pred"])}")
print(f"recall: {recall_score(y_test["true"], y_test["pred"])}")
print("confusion matrix:")
print(confusion_matrix(y_test["true"], y_test["pred"]))

# for i in range(len(players_test)):
#     player = players_test.iloc[i]
#     name = f"{player['First Name']} {player['Last Name']}"
#     print(f"{name}: {round(test_probs.iloc[i], 4) * 100}%, {player['Top Junior']}")


# try:
#     me = players[players["USATT#"] == 217865].iloc[0]
#     my_prob = log_reg_analysis(me, history=history_test, rolling=False, plot=True)
#     print(my_prob)
# except IndexError:
#     print("player not found")
