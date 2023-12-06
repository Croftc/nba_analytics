import pandas as pd
from IPython.display import HTML, display
import numpy as np
import pandas as pd
import warnings
from analytics.MoneylineModel import MoneylineModel
from analytics.Dataset import Dataset
warnings.filterwarnings('ignore')

data = Dataset()
model = MoneylineModel(do_ensemble=True)

team_logos = {
    "Atlanta": "https://content.sportslogos.net/logos/6/220/thumbs/22081902021.gif",
    "Boston": "https://content.sportslogos.net/logos/6/213/thumbs/slhg02hbef3j1ov4lsnwyol5o.gif",
    "Brooklyn": "https://content.sportslogos.net/logos/6/3786/thumbs/hsuff5m3dgiv20kovde422r1f.gif",
    "Charlotte": "https://content.sportslogos.net/logos/6/5120/thumbs/512019262015.gif",
    "Chicago": "https://content.sportslogos.net/logos/6/221/thumbs/hj3gmh82w9hffmeh3fjm5h874.gif",
    "Cleveland": "https://content.sportslogos.net/logos/6/222/thumbs/22269212018.gif",
    "Dallas": "https://content.sportslogos.net/logos/6/228/thumbs/22834632018.gif",
    "Denver": "https://content.sportslogos.net/logos/6/229/thumbs/22989262019.gif",
    "Detroit": "https://content.sportslogos.net/logos/6/223/thumbs/22321642018.gif",
    "Golden State": "https://content.sportslogos.net/logos/6/235/thumbs/23531522020.gif",
    "Houston": "https://content.sportslogos.net/logos/6/230/thumbs/23068302020.gif",
    "Indiana": "https://content.sportslogos.net/logos/6/224/thumbs/22448122018.gif",
    "LA Clippers": "https://content.sportslogos.net/logos/6/236/thumbs/23654622016.gif",
    "LA Lakers": "https://content.sportslogos.net/logos/6/237/thumbs/23787452016.gif",
    "Memphis": "https://content.sportslogos.net/logos/6/231/thumbs/23143732019.gif",
    "Miami": "https://content.sportslogos.net/logos/6/214/thumbs/burm5gh2wvjti3xhei5h16k8e.gif",
    "Milwaukee": "https://content.sportslogos.net/logos/6/225/thumbs/22582752016.gif",
    "Minnesota": "https://content.sportslogos.net/logos/6/232/thumbs/23296692018.gif",
    "New Orleans": "https://content.sportslogos.net/logos/6/4962/thumbs/496226812014.gif",
    "New York": "https://content.sportslogos.net/logos/6/216/thumbs/21671702024.gif",
    "Oklahoma City": "https://content.sportslogos.net/logos/6/2687/thumbs/2687722018.gif",
    "Orlando": "https://content.sportslogos.net/logos/6/217/thumbs/wd9ic7qafgfb0yxs7tem7n5g4.gif",
    "Philadelphia": "https://content.sportslogos.net/logos/6/218/thumbs/21870342016.gif",
    "Phoenix": "https://content.sportslogos.net/logos/6/238/thumbs/23843702014.gif",
    "Portland": "https://content.sportslogos.net/logos/6/239/thumbs/23997252018.gif",
    "Sacramento": "https://content.sportslogos.net/logos/6/240/thumbs/24040432017.gif",
    "San Antonio": "https://content.sportslogos.net/logos/6/233/thumbs/23325472018.gif",
    "Toronto": "https://content.sportslogos.net/logos/6/227/thumbs/22745782016.gif",
    "Utah": "https://content.sportslogos.net/logos/6/234/thumbs/23467492017.gif",
    "Washington": "https://content.sportslogos.net/logos/6/219/thumbs/21956712016.gif"
}

def american_odds_to_probability(odds):
    if odds > 0:
        probability = 100 / (odds + 100)
    else:
        probability = -odds / (-odds + 100)
    return probability

def calculate_profit(odds, size):
    if odds > 0:
        profit = (odds / 100) * size
    else:
        profit = (100 / -(odds + 0.0000001)) * size
    return profit

def kelly_criterion(bankroll, probability, odds, temper=1):
    """
    Calculate the optimal bet size using the Kelly Criterion.

    :param bankroll: Total amount of money you have to bet with.
    :param probability: The probability of the bet winning (from 0 to 1).
    :param odds: The odds being offered on the bet (in decimal format).
    :return: The recommended bet size according to the Kelly Criterion.
    """
    # Convert American odds to decimal if necessary
    if odds > 0:
        odds = (odds / 100) + 1
    elif odds < 0:
        odds = (100 / -odds) + 1

    # Calculate the Kelly bet fraction
    b = odds - 1  # Decimal odds minus 1
    q = 1 - probability  # Probability of losing
    kelly_fraction = (b * probability - q) / b

    # Calculate the recommended bet
    recommended_bet = (temper * kelly_fraction) * bankroll

    return recommended_bet

def combine_parlay_odds(odds_list):
    total_multiplier = 1
    for odds in odds_list:
        if odds > 0:  # Positive odds
            total_multiplier *= (odds / 100) + 1
        else:  # Negative odds
            total_multiplier *= 1 - (100 / (odds + 0.0000001))

    # Calculate parlay odds
    if total_multiplier >= 2:
        parlay_odds = (total_multiplier - 1) * 100
    else:
        parlay_odds = -100 / ((total_multiplier - 1) + 0.00000001)

    return round(parlay_odds)
def print_wrapper(func):
    ansi_reset = '\033[0m'
    ansi_black = '\033[90m'
    ansi_red = '\033[91m'
    ansi_green = '\033[92m'
    ansi_yellow = '\033[93m'
    ansi_blue = '\033[94m'
    ansi_pink = '\033[95m'
    ansi_teal = '\033[96m'
    ansi_gray = '\033[97m'
    ansi_warning = '\033[31;1;4m'
    ansi_error = '\033[31;100m'
    def wrapped_func(*args,**kwargs):
        new_args = args + tuple()
        new_kwargs = kwargs.copy()
        for kwarg, kwvalue in kwargs.items(): # Loop through the keyword arguments
            if kwarg == "color":
                if kwvalue == "black":
                    color = ansi_black
                elif kwvalue == "red":
                    color = ansi_red
                elif kwvalue == "green":
                    color = ansi_green
                elif kwvalue == "yellow":
                    color = ansi_yellow
                elif kwvalue == "blue":
                    color = ansi_blue
                elif kwvalue == "pink":
                    color = ansi_pink
                elif kwvalue == "teal":
                    color = ansi_teal
                elif kwvalue == "gray":
                    color = ansi_gray
                elif kwvalue == "warning":
                    color = ansi_warning
                elif kwvalue == "error":
                    color = ansi_error
                new_kwargs = kwargs.copy() # Make a copy of the keyword arguments dict
                del new_kwargs["color"] # Remove color from the keyword arguments dict
        try: # Is the variable color defined?
            color
        except NameError:
            pass
            # no color was specified
        else:
            new_args = ()
            for arg in args:
                new_args += (f"{color}{arg}{ansi_reset}",) # Apply the ANSI escape codes to each non-keyword argument
        return func(*new_args,**new_kwargs)
    return wrapped_func

print = print_wrapper(print) # Apply the wrapper to the print() function

def probability_to_american_odds(probability):
    if probability < 0 or probability > 1:
        raise ValueError("Probability must be between 0 and 1")

    if probability == 0.5:
        return 100  # Even odds

    if probability > 0.5:
        return int(-100 * (probability / (1 - probability)))
    else:
        return int(100 * ((1 - probability) / probability))

def odds_to_str(odds):
  if odds <= 0:
    return odds
  else:
    return f'+{odds}'


    thresh = 0.5

def process_data_frame(df):
    df['DATE'] = pd.to_datetime(df['DATE'])
    return df.sort_values('DATE')

def update_bankroll(bankroll, profit):
    bankroll += profit
    return max(bankroll, 0)  # Prevents negative bankroll

def print_bet_results(date, wins, losses, total, bankroll, start, hit_all, all_odds, hit_all_all, all_all_odds):
    win_rate = wins / total if total > 0 else 0

    if hit_all and total > 1:
      print(f'\tBANGGG!!! Hit a {total} leg parlay at +{combine_parlay_odds(all_odds)} - pays {round(calculate_profit(combine_parlay_odds(all_odds), bankroll*0.1), 2)}')
    if hit_all_all and total > 1:
      print(f'\t HOLY SHIT WE CLEARED A {total} LEG SLATE AT +{combine_parlay_odds(all_all_odds)} PAID {round(calculate_profit(combine_parlay_odds(all_all_odds), 10), 2)}')
    print(f'Results: bankroll start: {round(start,2)} end: {round(bankroll,2)} for profit of: {round(bankroll - start, 2)}, win rate = {win_rate:.2f}\n')


def predict_today():

    best_model = model
    TODAY_MAP = data.get_today_data()
    t_teams = list(TODAY_MAP.keys())
    pre_tdf = data.df[(data.df['Season'] == 2024)]
    raw_tdf = data.get_ydf()
    raw_tdf.columns = data.t_cleaned_cols

    
    tdf = raw_tdf
    tdf['DATE'] = tdf['DATE'].astype('datetime64[ns]')
    tdf = tdf.sort_values('DATE')
    tdf['Season'] = raw_tdf['BIGDATABALL_DATASET'].map(data.SEASON_MAP)


    # Step 1: Result of the Game
    def assign_results(group):
        group['MAIN REF'] = [group['MAIN REF'].iloc[0]]*2
        if group.iloc[0]['PTS'] > group.iloc[1]['PTS']:
            group['Result'] = [1, 0]
        else:
            group['Result'] = [0, 1]

        group['Opp_Avg_3_game_PTS'] = [group.iloc[1]['Avg_3_game_PTS'], group.iloc[0]['Avg_3_game_PTS']]
        group['Opp_Avg_2_game_PTS'] = [group.iloc[1]['Avg_5_game_PTS'], group.iloc[0]['Avg_5_game_PTS']]
        group['Opp_Season_Avg_PTS'] = [group.iloc[1]['Season_Avg_PTS'], group.iloc[0]['Season_Avg_PTS']]

        group['Opp_Avg_3_game_POSS'] = [group.iloc[1]['Avg_3_game_POSS'], group.iloc[0]['Avg_3_game_POSS']]
        group['Opp_Avg_5_game_POSS'] = [group.iloc[1]['Avg_5_game_POSS'], group.iloc[0]['Avg_5_game_POSS']]
        group['Opp_Season_Avg_POSS'] = [group.iloc[1]['Season_Avg_POSS'], group.iloc[0]['Season_Avg_POSS']]

        group['Opp_Avg_3_game_PACE'] = [group.iloc[1]['Avg_3_game_PACE'], group.iloc[0]['Avg_3_game_PACE']]
        group['Opp_Avg_5_game_PACE'] = [group.iloc[1]['Avg_5_game_PACE'], group.iloc[0]['Avg_5_game_PACE']]
        group['Opp_Season_Avg_PACE'] = [group.iloc[1]['Season_Avg_PACE'], group.iloc[0]['Season_Avg_PACE']]

        group['Opp_Avg_3_game_DEFF'] = [group.iloc[1]['Avg_3_game_DEFF'], group.iloc[0]['Avg_3_game_DEFF']]
        group['Opp_Avg_5_game_DEFF'] = [group.iloc[1]['Avg_5_game_DEFF'], group.iloc[0]['Avg_5_game_DEFF']]
        group['Opp_Season_Avg_DEFF'] = [group.iloc[1]['Season_Avg_DEFF'], group.iloc[0]['Season_Avg_DEFF']]

        group['Opp_Avg_3_game_OEFF'] = [group.iloc[1]['Avg_3_game_OEFF'], group.iloc[0]['Avg_3_game_OEFF']]
        group['Opp_Avg_5_game_OEFF'] = [group.iloc[1]['Avg_5_game_OEFF'], group.iloc[0]['Avg_5_game_OEFF']]
        group['Opp_Season_Avg_OEFF'] = [group.iloc[1]['Season_Avg_OEFF'], group.iloc[0]['Season_Avg_OEFF']]

        return group


    # Step 2: Average Points
    tdf['Avg_3_game_PTS'] = tdf.groupby(['TEAM', 'Season'])['PTS'].transform(lambda x: x.shift(1).rolling(3).mean())
    tdf['Avg_5_game_PTS'] = tdf.groupby(['TEAM', 'Season'])['PTS'].transform(lambda x: x.shift(1).rolling(5).mean())
    tdf['Season_Avg_PTS'] = tdf.groupby(['TEAM', 'Season'])['PTS'].transform('mean')

    tdf['Avg_3_game_POSS'] = tdf.groupby(['TEAM', 'Season'])['POSS'].transform(lambda x: x.shift(1).rolling(3).mean())
    tdf['Avg_5_game_POSS'] = tdf.groupby(['TEAM', 'Season'])['POSS'].transform(lambda x: x.shift(1).rolling(5).mean())
    tdf['Season_Avg_POSS'] = tdf.groupby(['TEAM', 'Season'])['POSS'].transform('mean')

    tdf['Avg_3_game_PACE'] = tdf.groupby(['TEAM', 'Season'])['PACE'].transform(lambda x: x.shift(1).rolling(3).mean())
    tdf['Avg_5_game_PACE'] = tdf.groupby(['TEAM', 'Season'])['PACE'].transform(lambda x: x.shift(1).rolling(5).mean())
    tdf['Season_Avg_PACE'] = tdf.groupby(['TEAM', 'Season'])['PACE'].transform('mean')

    tdf['Avg_3_game_OEFF'] = tdf.groupby(['TEAM', 'Season'])['OEFF'].transform(lambda x: x.shift(1).rolling(3).mean())
    tdf['Avg_5_game_OEFF'] = tdf.groupby(['TEAM', 'Season'])['OEFF'].transform(lambda x: x.shift(1).rolling(5).mean())
    tdf['Season_Avg_OEFF'] = tdf.groupby(['TEAM', 'Season'])['OEFF'].transform('mean')

    tdf['Avg_3_game_DEFF'] = tdf.groupby(['TEAM', 'Season'])['DEFF'].transform(lambda x: x.shift(1).rolling(3).mean())
    tdf['Avg_5_game_DEFF'] = tdf.groupby(['TEAM', 'Season'])['DEFF'].transform(lambda x: x.shift(1).rolling(5).mean())
    tdf['Season_Avg_DEFF'] = tdf.groupby(['TEAM', 'Season'])['DEFF'].transform('mean')


    # Apply the function to each game group
    tdf = tdf.groupby('GAME-ID').apply(assign_results)
    # Reset index if needed
    tdf.reset_index(drop=True, inplace=True)

    # Shift the Result column for streak calculation
    tdf['Prev_Result'] = tdf.groupby(['TEAM', 'Season'])['Result'].shift()

    # Step 3: Win/Loss Streak
    def calculate_streak(group):
        streak = group['Prev_Result'].diff().ne(0).cumsum()
        group['Streak'] = streak.groupby(streak).cumcount()
        group['Streak'] *= group['Prev_Result'].map({1: 1, 0: -1})
        return group

    tdf = tdf.groupby(['TEAM', 'Season']).apply(calculate_streak)

    # Step 4: Last 3 Games Moneylines
    tdf['Last_ML_1'] = tdf.groupby(['TEAM'])['MONEYLINE'].shift(1)
    tdf['Last_ML_2'] = tdf.groupby(['TEAM'])['MONEYLINE'].shift(2)
    tdf['Last_ML_3'] = tdf.groupby(['TEAM'])['MONEYLINE'].shift(3)

    # Step 5: Current Number of Wins - Losses
    tdf['Wins'] = tdf.groupby(['TEAM'])['Result'].cumsum()
    tdf['Losses'] = tdf.groupby(['TEAM'])['Result'].transform('count') - tdf['Wins']
    tdf['Win_Loss_Diff'] = tdf['Wins'] - tdf['Losses']

    # Step 6: Current Opponent
    tdf['Opponent'] = tdf.groupby('GAME-ID')['TEAM'].shift(-1).fillna(tdf.groupby('GAME-ID')['TEAM'].shift())

    # Clean up and remove the temporary 'Prev_Result' column
    tdf.drop('Prev_Result', axis=1, inplace=True)

    ref_map = {team: details[0] for team, details in TODAY_MAP.items()}
    moneyline_map = {team: details[1] for team, details in TODAY_MAP.items()}
    venue_map = {team: details[2] for team, details in TODAY_MAP.items()}
    opp_map = {team: details[3] for team, details in TODAY_MAP.items()}
    spread_map = {team: details[4] for team, details in TODAY_MAP.items()}
    total_map = {team: details[5] for team, details in TODAY_MAP.items()}


    tdf['DATE'] = tdf['DATE'].astype('datetime64[ns]')
    tdf['MAIN REF'] = tdf['TEAM'].map(ref_map)
    tdf['MONEYLINE'] = tdf['TEAM'].map(moneyline_map)
    tdf['MONEYLINE'] = tdf['TEAM'].map(moneyline_map)
    tdf['CLOSING_SPREAD'] = tdf['TEAM'].map(spread_map)
    tdf['CLOSING_TOTAL'] = tdf['TEAM'].map(total_map)
    tdf['Opponent'] = tdf['TEAM'].map(opp_map)
    refs = data.get_refs_data()
    tdf = tdf.merge(refs.groupby('REFEREE').mean(), how='left', left_on='MAIN REF', right_on='REFEREE')
    tdf = tdf.sort_values('DATE')

    temp = pre_tdf.sort_values(by=['TEAM', 'DATE'], ascending=[True, False])

    # Drop duplicates, keep the first (latest) entry for each 'name'
    temp = temp.drop_duplicates(subset='TEAM')

    # Merge df1 with the processed df2
    temp = pd.merge(tdf, temp[['TEAM', 'Elo_Rating', 'Momentum']], on='TEAM', how='left', suffixes=('_x', '_y'))

    # look at the latest
    temp = temp.sort_values(by=['TEAM', 'DATE'], ascending=[True, False])
    temp = temp.drop_duplicates(subset='TEAM')[data.t_train_cols]

    temp = temp[temp['TEAM'].isin(t_teams)]
    temp.columns = data.train_cols_final

    X = temp.copy()
    X['MONEYLINE'] = X['TEAM'].map(moneyline_map)
    X[['MONEYLINE', 'Last_ML_1', 'Last_ML_2', 'Last_ML_3', 'CLOSING_SPREAD', 'CLOSING_TOTAL']] = X[['MONEYLINE', 'Last_ML_1', 'Last_ML_2', 'Last_ML_3', 'CLOSING_SPREAD', 'CLOSING_TOTAL']].replace('Even', '-100', regex=True)
    X[['MONEYLINE', 'Last_ML_1', 'Last_ML_2', 'Last_ML_3', 'CLOSING_SPREAD', 'CLOSING_TOTAL']] = X[['MONEYLINE', 'Last_ML_1', 'Last_ML_2', 'Last_ML_3', 'CLOSING_SPREAD', 'CLOSING_TOTAL']].fillna(0).astype(float)
    X['MAIN REF'] = X['MAIN REF'].astype('category')
    X['CREW'] = X['CREW'].astype('category')
    X['TEAM'] = X['TEAM'].astype('category')
    X['Opponent'] = X['Opponent'].astype('category')
    X['TEAM_REST_DAYS'] = X['TEAM_REST_DAYS'].astype('category')
    X['VENUE'] = (X['VENUE'] == 'H')*1

    # make predictions
    probs = best_model.predict_proba(X)
    team_probs_map = {team: prob for team, prob in zip(X['TEAM'].values, probs[:, 1])}
    odds = X['MONEYLINE'].values
    #booster = best_model.get_booster()
    normed_odds = {team: team_probs_map[team]/(team_probs_map[team] + team_probs_map[opp]) for team, opp in zip(X['TEAM'], X['Opponent'])}
    do_bet = {team: normed_odds[team] > normed_odds[opp] for team, opp in zip(X['TEAM'], X['Opponent'])}

    #pred_contribs = booster.predict(DMatrix(X, enable_categorical=True), pred_contribs=True)
    pred_contribs = np.ones(shape=(len(X), len(X)))

    output_html = ''
    matchups = []
    for team, prob, opp, contribs, elo, mom in zip(X['TEAM'].values, probs[:, 1], X['Opponent'].values, pred_contribs[:, :-1], X['Elo_Rating'].values, X['Momentum'].values):
        home, away = {}, {}
        # get the most important features
        helpers = np.array(data.TRAIN_COLS)[np.argpartition(contribs, -3)[-3:]]
        detractions = (np.array(data.TRAIN_COLS)[np.argpartition(contribs, -3)[:3]])

        # get this team odds
        o = -100 if moneyline_map[team] == 'Even' else int(moneyline_map[team])
        odd = str(o) if o < 0 else f'+{o}'

        # get opp odds
        o2 = -100 if moneyline_map[opp] == 'Even' else int(moneyline_map[opp])
        odd2 = str(o2) if o2 < 0 else f'+{o2}'

        # get out odds
        our_line = probability_to_american_odds(normed_odds[team])
        our_line = str(our_line) if our_line < 0 else f'+{our_line}'

        # get our opp odds
        our_opp_line = probability_to_american_odds(normed_odds[opp])
        our_opp_line = str(our_opp_line) if our_opp_line < 0 else f'+{our_opp_line}'

        # get the bet sizing
        bet = kelly_criterion(100, normed_odds[team], o, temper=0.13)

        # tab character for spacing the prints
        tab = '&nbsp;&nbsp;&nbsp;&nbsp;'
        win_color, lose_color = 'black', 'black'
        do_save = False
        # make picks
        if (bet >= 0) and do_bet[team]:
            win_color = 'green'
            lose_color = 'red'
            do_save = True
            #b = f'Stright bet {round(bet, 2)}u to win {round(calculate_profit(o, round(bet, 2)),2)}u' if round(bet, 2) > 0 else 'Don\'t bet this straight - parlay only'

        elif do_bet[team] and (bet < 0):
            win_color = '#E4CD05'
            lose_color = 'orange'
            do_save = True
            #b = f'Stright bet {round(bet, 2)}u to win {round(calculate_profit(o, round(bet, 2)),2)}u' if round(bet, 2) > 0 else 'Don\'t bet this straight - parlay only'

            # home['team'] = team
            # home['win_probability'] = round(normed_odds[team][0]*100, 2)
            # home['bet'] = round(bet, 2)
            # home['team_rating'] = int(elo)
            # home['momentum'] = int(mom)
            # home['best_features'] = helpers[:3]
            # home['logo'] = team_logos[team]
            # home['vegas'] = odd
            # home['our_line'] = our_line
            # home['color'] = 'yellow'

            # away['team'] = opp
            # away['win_probability'] = round(normed_odds[opp][0]*100, 2)
            # away['bet'] = -1
            # away['team_rating'] = int(X[X["TEAM"] == opp]["Elo_Rating"])
            # away['momentum'] = int(X[X["TEAM"] == opp]["Momentum"])
            # away['best_features'] = detractions[:3]
            # away['logo'] = team_logos[opp]
            # away['vegas'] = odd2
            # away['our_line'] = our_opp_line
            # away['color'] = 'orange'
        
        home['team'] = team
        home['win_probability'] = round(normed_odds[team]*100, 2)
        home['bet'] = round(bet, 2)
        home['team_rating'] = int(elo)
        home['momentum'] = int(mom)
        home['best_features'] = helpers[:3]
        home['logo'] = team_logos[team]
        home['vegas'] = odd
        home['our_line'] = our_line
        home['color'] = win_color

        away['team'] = opp
        away['win_probability'] = round(normed_odds[opp]*100, 2)
        away['bet'] = -1
        away['team_rating'] = int(X[X["TEAM"] == opp]["Elo_Rating"])
        away['momentum'] = int(X[X["TEAM"] == opp]["Momentum"])
        away['best_features'] = detractions[:3]
        away['logo'] = team_logos[opp]
        away['vegas'] = odd2
        away['our_line'] = our_opp_line
        away['color'] = lose_color

        if do_save:
            matchups.append([home, away])

    with open('Output.html', 'w', encoding='utf-8') as f:
        f.write(output_html)

    return {'matchups': matchups}  