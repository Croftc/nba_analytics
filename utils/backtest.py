import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from IPython.display import display, HTML
import torch.nn.functional as F
import joblib
from utils import nba_inference_utils as niu
import matplotlib.pyplot as plt

# Constants
THRESHOLD = 0.5

def process_data_frame(df):
    """Convert DATE to datetime and sort the DataFrame."""
    df['DATE'] = pd.to_datetime(df['DATE'])
    return df.sort_values('DATE')

def update_bankroll(bankroll, profit):
    """Update bankroll, ensuring it doesn't go negative."""
    return max(bankroll + profit, 0)

def print_bet_results(date, wins, losses, total, bankroll, start, made_a_bet, all_odds):
    """Print results of betting for the day."""
    win_rate = wins / total if total > 0 else 0
    #print(f'Results for {date}: Start Bankroll: {round(start, 2)}, End Bankroll: {round(bankroll, 2)}, '
    #      f'Profit: {round(bankroll - start, 2)}, Win Rate: {win_rate:.2f}\n')

def prepare_features(day_data, feature_cols):
    # remove GAME-ID from feature_cols
    feature_cols = [col for col in feature_cols if col != 'GAME-ID']
    """Prepare features for prediction."""
    X = day_data[feature_cols].copy()
    # Prepare features
    # X[["MONEYLINE", "Last_ML_1", "Last_ML_2", "Last_ML_3"]] = (
    #     X[["MONEYLINE", "Last_ML_1", "Last_ML_2", "Last_ML_3"]]
    #     .replace('even', '-100', regex=True)
    #     .fillna(0)
    #     .astype(int)
    # )
    # Convert categorical columns
    for col in ['MAIN REF', 'TEAM', 'CREW', 'Opponent']:
        X[col] = X[col].astype('category')
    #X['VENUE'] = (X['VENUE'] == 'H').astype(int)
    return X

def calculate_bets(X, model, y, odds_col, threshold=THRESHOLD):
    """Calculate betting decisions based on predictions."""
    probabilities = model.predict_proba(X)[:, 1]
    
    # Initial probabilities mapping: team to its predicted probability
    ps = {team: prob for team, prob in zip(X['TEAM'], probabilities)}
    
    if odds_col == 'CLOSING_TOTAL':
        # Adjust probabilities for total bets by averaging the probabilities of both teams
        real_probabilities = {}
        processed_games = set()
        
        for team, opp in zip(X['TEAM'], X['Opponent']):
            # Ensure we process each game only once
            game = tuple(sorted([team, opp]))
            if game not in processed_games:
                processed_games.add(game)
                # Sum probabilities of both teams and divide by 2
                prob_team = ps[team]
                prob_opp = ps[opp]
                average_prob = (prob_team + prob_opp) / 2
                # Assign the average probability to both teams
                real_probabilities[team] = average_prob
                real_probabilities[opp] = average_prob
            else:
                # If the game is already processed, assign the existing average probability
                average_prob = real_probabilities[team]
                
        # Update the probabilities mapping with the averaged probabilities
        ps = real_probabilities
    predictions = probabilities > threshold
    
    # Calculate normalized odds and betting decisions
    normed_odds = {team: ps[team] / (ps[team] + ps[opp]) for team, opp in zip(X['TEAM'], X['Opponent'])}

    if odds_col != 'CLOSING_TOTAL':
        do_bet = {team: normed_odds[team] > normed_odds[opp] for team, opp in zip(X['TEAM'], X['Opponent'])}
    else:
        do_bet = {team: ps[team] > 0.5 for team in X['TEAM']}
        normed_odds = ps

    return predictions, probabilities, do_bet, normed_odds


def process_daily_bets(day_data, model, bankroll, bet_size, y, feature_cols, odds_col, end_date, threshold=THRESHOLD, kf=0.5):
    """Process bets for a single day."""
    X = prepare_features(day_data, feature_cols)
    predictions, probabilities, do_bet, normed_odds = calculate_bets(X, model, y, odds_col, threshold)

    wins, losses, day_profit, num_bets = 0, 0, 0, 0
    all_odds = []
    already_bet = []
    end = pd.to_datetime(end_date)
    end_1 = pd.to_datetime(end_date) - pd.DateOffset(1)
    end_2 = pd.to_datetime(end_date) - pd.DateOffset(2)
    if (day_data['DATE'].iloc[0] == end) or (day_data['DATE'].iloc[0] == end_1) or (day_data['DATE'].iloc[0] == end_2):
        print(day_data['DATE'].iloc[0])
    for pred, actual, odds, team, opp in zip(predictions, y, X[odds_col], X['TEAM'], X['Opponent']):
        
        if (team in already_bet or opp in already_bet) and odds_col == 'CLOSING_TOTAL':
            continue

        already_bet.append(team)
        already_bet.append(opp)
        
        if do_bet[team]:
            if odds_col == 'Momentum':
                odds = 300
            elif odds_col != 'MONEYLINE':
                odds = -110
            unit = bankroll * 0.1
            # Calculate bet size using Kelly Criterion
            bet_size = round(niu.kelly_criterion(unit, normed_odds[team], odds, kf), 2)

            if (unit - bet_size) >= 0 and bet_size > 0:
                to_win = round(niu.calculate_profit(odds, bet_size), 2)
                bankroll -= bet_size
                num_bets += 1
                
                
                if actual:
                    if (day_data['DATE'].iloc[0] == end) or (day_data['DATE'].iloc[0] == end_1) or (day_data['DATE'].iloc[0] == end_2):
                        display(HTML(f'<span style="color: green;">Won {to_win} betting {bet_size} on {team} vs {opp} at {odds}</span>'))
                    day_profit += (to_win + bet_size)
                    wins += 1
                else:
                    if (day_data['DATE'].iloc[0] == end) or (day_data['DATE'].iloc[0] == end_1) or (day_data['DATE'].iloc[0] == end_2):
                        display(HTML(f'<span style="color: red;">Lost {bet_size} betting on {team} vs {opp} at {odds}</span>'))
                    losses += 1
                all_odds.append(odds)
    if (day_data['DATE'].iloc[0] == end) or (day_data['DATE'].iloc[0] == end_1) or (day_data['DATE'].iloc[0] == end_2):
        print()
    return bankroll + day_profit, wins, losses, num_bets, all_odds

def backtest_model(df, model, feature_cols, label_col, odds_col, start_date, end_date, initial_bankroll, bet_size, threshold=THRESHOLD, kf=0.5):
    """Backtest the betting model over a specified date range and plot bankroll and daily ROI over time."""
    df = process_data_frame(df)
    bankroll = initial_bankroll
    dates = []
    bankrolls = []
    daily_rois = []
    current_date = pd.to_datetime(start_date)

    while current_date <= pd.to_datetime(end_date):
        #print(f'Processing {current_date.date()}...')
        day_data = df[df['DATE'] == current_date]
        start_bankroll = bankroll

        if not day_data.empty and bankroll > 0:
            y = day_data[label_col]
            bankroll, wins, losses, num_bets, all_odds = process_daily_bets(day_data, model, bankroll, bet_size, y, feature_cols, odds_col, end_date, threshold, kf=kf)
            print_bet_results(current_date.date(), wins, losses, num_bets, bankroll, start_bankroll, len(all_odds) > 0, all_odds)
        else:
            # No bets made today
            wins, losses, num_bets = 0, 0, 0
            #print(f'No bets made on {current_date.date()}.')

        # Compute Daily ROI
        if start_bankroll > 0:
            daily_roi = (bankroll - start_bankroll) / start_bankroll
        else:
            daily_roi = 0  # Avoid division by zero

        # Append the date, bankroll, and daily ROI
        dates.append(current_date)
        bankrolls.append(bankroll)
        daily_rois.append(daily_roi)

        current_date += pd.Timedelta(days=1)

    # Convert dates to pandas datetime for plotting
    dates = pd.to_datetime(dates)

    # Plot bankroll over time
    plt.figure(figsize=(12, 6))
    plt.plot(dates, bankrolls, label='Bankroll', marker='o')
    plt.xlabel('Date')
    plt.ylabel('Bankroll ($)')
    plt.title('Bankroll Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot Daily ROI over time
    plt.figure(figsize=(12, 6))
    plt.plot(dates, daily_rois, label='Daily ROI', color='orange', marker='o')
    plt.xlabel('Date')
    plt.ylabel('Daily ROI')
    plt.title('Daily ROI Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

    return bankroll, bankroll - initial_bankroll, bankrolls, daily_rois
