import pandas as pd
from datetime import date

players = pd.read_csv("./data/player-list.csv")
td = pd.read_csv("./data/td-ratings.csv")
history = pd.read_csv("./data/player-history.csv")

td.rename({"Member ID": "USATT#"}, axis=1, inplace=True)
td = td[["USATT#", "State", "Zip", "Gender", "Date of Birth", "Expiration Date"]]
players = players.merge(td, how="left")


def max_rating(player):
    r_hist = history[history["USATT#"] == player["USATT#"]]
    return r_hist["Final Rating"].max()


def tournament_count(player):
    rhist = history[history["USATT#"] == player["USATT#"]]
    return len(rhist)


history["Final Rating"] = pd.to_numeric(history["Final Rating"], errors="coerce")
players["Max Rating"] = players.apply(max_rating, axis=1)

players["Date of Birth"] = pd.to_datetime(players["Date of Birth"], errors="coerce")
tdy = date(2024, 9, 23)
players["Age"] = pd.to_timedelta(pd.to_datetime(tdy) - players["Date of Birth"])  # type: ignore

players["Tournaments Played"] = players.apply(tournament_count, axis=1)

players = players[players["Tournament Rating"] > 0]

players.to_csv("./data/player-stats.csv", index=False)
print("player-stats.csv created")
