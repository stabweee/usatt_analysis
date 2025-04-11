import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

import scipy.stats as ss
from sklearn.linear_model import LinearRegression

from load_data.load import load_stats, load_history

pd.options.mode.chained_assignment = None

"""
----- Load Data -----
"""
players = load_stats()
history = load_history()


"""
----- Clean Data -----
"""
# find age at each tournament for every player
age_history = history.merge(players[["Database#", "Date of Birth"]], how="left")
history["tournament age"] = (
    history["Start Date"] - age_history["Date of Birth"]
).dt.days / 365


# find number of days since first tournament for every tournament for every player
def start_date(player):
    rhist = history[history["Database#"] == player["Database#"]]
    return rhist["Start Date"].iloc[-1]


# calculate number of days played by each tournament
start_dates = players.apply(start_date, axis=1)
start_dates.index = players["Database#"]
history["days played"] = (
    history["Start Date"].reset_index(drop=True)
    - start_dates[history["Database#"]].reset_index(drop=True)
).dt.days

# filter out people who don't have initial ratings, start dates, or ages
history = history.dropna(subset=["Initial Rating", "Start Date", "days played"])
players = players[players["Database#"].isin(history["Database#"])]


# valid player defined as players who didn't gain more than 1500 points in their first tournament, have played more than 5
# tournaments before they turned 19, and have a correlation coerffiient of over 0.75
def valid_player(player):
    max_first_tournament = 1500
    min_tournament_count = 5
    min_correlation = 0.75

    rhist = history[history["Database#"] == player["Database#"]]
    if len(rhist) == 0:
        return False

    first_tournament = rhist.iloc[-1]
    if not (
        (first_tournament["Initial Rating"] == 0)
        and (first_tournament["Final Rating"] <= max_first_tournament)
        and (
            len(
                rhist[
                    (rhist["Start Date"] - player["Date of Birth"]).dt.days <= 19 * 365
                ]
            )
            >= min_tournament_count
        )
    ):
        return False

    data = rhist[["days played", "Initial Rating"]]
    data["log days played"] = np.log1p(data["days played"])
    reg = ss.linregress(data["log days played"], data["Initial Rating"])

    return reg.rvalue > min_correlation


# filter out all invalid players
players["Valid Player"] = players.apply(valid_player, axis=1)
valid = players[["Database#", "Valid Player"]]
history = history.merge(valid, how="inner", on="Database#")
players = players[players["Valid Player"] == True].reset_index(drop=True)
history = history[history["Valid Player"] == True].reset_index(drop=True)


# check if tournament is played while player is U19
def junior_tournaments(tournament):
    player = players[players["Database#"] == tournament["Database#"]].reset_index(
        drop=True
    )
    if len(player) == 0:
        return False
    return (tournament["Start Date"] - player.loc[0, "Date of Birth"]).days <= 19 * 365


history = history[history.apply(junior_tournaments, axis=1)].reset_index(drop=True)

"""
----- Feature Engineering -----
"""


# age at x rating
def age_by_x(player, r):
    r_hist = history[history["USATT#"] == player["USATT#"]]
    scope = r_hist[r_hist["Final Rating"] > r]
    if len(scope) == 0:
        return -1
    return (scope.loc[scope.index[-1], "Start Date"] - player["Date of Birth"]).days


# hit the target rating by u19
def is_top_junior(player, male_rating, female_rating):
    if player["Gender"] == "M":
        age_by_rating = age_by_x(player, male_rating)
    elif player["Gender"] == "F":
        age_by_rating = age_by_x(player, female_rating)
    else:
        return pd.NA

    if age_by_rating == -1:
        return False
    return age_by_rating < 19 * 365


players["Top Junior"] = players.apply(is_top_junior, args=(2400, 2200), axis=1)

# scope training to only include people over 19 years old or people who are currently U19 and a top junior
training = players[
    (players["Top Junior"] == True) | (players["Age"].dt.days > 19 * 365)
]

training.to_csv("./top_juniors/players_training_set.csv", index=False)
history.to_csv("./top_juniors/history_training_set.csv", index=False)
