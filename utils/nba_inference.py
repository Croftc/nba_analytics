#!/usr/bin/env python3

import warnings
warnings.simplefilter(action='ignore')  # Ignore all warnings

import numpy as np
import time
import pandas as pd
import joblib
import nba_inference_utils as niu
from ccb_model import BootstrapCalibratedClassifier
from datetime import datetime

def main():
    print("### Loading Models")
    spread_model = joblib.load('calibrated_spread_model.pkl')
    ml_model = joblib.load('calibrated_ml_model.pkl')
    total_model = joblib.load('calibrated_total_model_2.pkl')

    train_cols = [
        'Opp_Elo', 'Opp_Momentum', 'SPREAD_LINE_MOVEMENT_1', 
        'SPREAD_LINE_MOVEMENT_2', 'SPREAD_LINE_MOVEMENT_3',
        'TOTAL_LINE_MOVEMENT_1', 'TOTAL_LINE_MOVEMENT_2',
        'TOTAL_LINE_MOVEMENT_3', 'CREW', 'Opp_Avg_3_game_DEFF',
        'Opp_Avg_5_game_DEFF', 'Opp_Season_Avg_DEFF',
        'Opp_Avg_3_game_OEFF', 'Opp_Avg_5_game_OEFF',
        'Opp_Season_Avg_OEFF', 'Opp_Avg_3_game_PACE',
        'Opp_Avg_5_game_PACE', 'Opp_Season_Avg_PACE',
        'Opp_Avg_3_game_POSS', 'Opp_Avg_5_game_POSS',
        'Opp_Season_Avg_POSS', 'Avg_3_game_DEFF',
        'Avg_5_game_DEFF', 'Season_Avg_DEFF',
        'Avg_3_game_OEFF', 'Avg_5_game_OEFF',
        'Season_Avg_OEFF', 'Avg_3_game_PACE',
        'Avg_5_game_PACE', 'Season_Avg_PACE',
        'Avg_3_game_POSS', 'Avg_5_game_POSS',
        'Season_Avg_POSS', 'Avg_3_game_OR', 'Avg_5_game_OR',
        'Season_Avg_OR','Avg_3_game_3P', 'Avg_5_game_3P',
        'Season_Avg_3P','Avg_3_game_3PA', 'Avg_5_game_3PA',
        'Season_Avg_3PA','Avg_3_game_TO', 'Avg_5_game_TO',
        'Season_Avg_TO','Avg_3_game_FT', 'Avg_5_game_FT',
        'Season_Avg_FT','CLOSING_SPREAD',
        'CLOSING_TOTAL', 'MONEYLINE', 'Avg_3_game_PTS',
        'Avg_5_game_PTS', 'Season_Avg_PTS', 'Last_ML_1',
        'Last_ML_2', 'Last_ML_3', 'VENUE', 'TEAM', 'Opponent',
        'Win_Loss_Diff', 'HOME TEAM WIN%', 'HOME TEAM POINTS DIFFERENTIAL',
        'TOTAL POINTS PER GAME', 'CALLED FOULS PER GAME',
        'FOUL% AGAINST ROAD TEAMS', 'FOUL% AGAINST HOME TEAMS',
        'FOUL DIFFERENTIAL (Against Road Team) - (Against Home Team)',
        'Elo_Rating', 'MAIN REF', 'TEAM_REST_DAYS',
        'Offensive_Rating', 'Defensive_Rating',
        'Opp_Offensive_Rating', 'Opp_Defensive_Rating', 'two_week_totals'
    ]

    print("### 1. Scraping data / fetching referee data")
    current_odds = niu.scrape_odds()
    ref_data = pd.read_csv('../historical_data/2024-2025.csv')
    driver = niu.init_driver()
    driver.get('https://official.nba.com/referee-assignments/')
    time.sleep(5)
    html_content = driver.page_source
    driver.quit()

    referee_data = niu.parse_referee_data(html_content)
    for city, refs in referee_data.items():
        if city == 'L.A. Lakers':
            city = 'LA Lakers'
        try:
            current_odds[city][0] = ' '.join(refs[0].split(' ')[:-1])
            current_odds[city][-2] = ' '.join(refs[1].split(' ')[:-1])
            current_odds[city][-1] = ' '.join(refs[2].split(' ')[:-1])
        except KeyError:
            pass

    TODAY_MAP = niu.remove_ref_keys(current_odds)
    TODAY_MAP = pd.DataFrame.from_dict(
        TODAY_MAP, orient='index',
        columns=[
            'Referee', 'MONEYLINE', 'Venue', 'Opponent',
            'CLOSING_SPREAD', 'CLOSING_TOTAL',
            'Spread_Movement', 'Total_Movement',
            'CREW', 'CREW2'
        ]
    )
    TODAY_MAP['TEAM'] = TODAY_MAP.index.copy(deep=True)
    TODAY_MAP = TODAY_MAP.reset_index(drop=True)
    TODAY_MAP = TODAY_MAP.merge(
        ref_data.groupby('REFEREE').first(),
        how='left', left_on='Referee', right_on='REFEREE',
        suffixes=['x', '']
    )

    print("### 2. Load historical data")
    today = datetime.now().strftime('%Y-%m-%d')
    yesterday = (datetime.now() - pd.DateOffset(3)).strftime('%Y-%m-%d')
    historical_data = pd.read_csv(f'2024_2025_nba_team_full_{today}.csv')
    today_teams_list = list(niu.team_map.values())
    most_recent_historical = niu.get_most_recent_rows(historical_data, today_teams_list)

    print("### 3. Download current data (yesterday) & update ELO")
    yesterday_data = niu.download_current_data()
    yesterday_df = pd.read_excel(yesterday_data)
    yesterday_df.columns = yesterday_df.columns.str.replace(' ', '_')
    yesterday_df.columns = yesterday_df.columns.str.replace('\\n', '_')
    yesterday_df.columns = yesterday_df.columns.str.replace('__', '_')
    yesterday_df['DATE'] = pd.to_datetime(yesterday_df['DATE'])
    most_recent_historical_date = most_recent_historical['DATE'].max()

    # Opponent fill
    yesterday_df['Opponent'] = (
        yesterday_df.groupby('GAME-ID')['TEAM']
        .shift(-1).fillna(yesterday_df.groupby('GAME-ID')['TEAM'].shift())
    )
    yesterday_df = yesterday_df.groupby('GAME-ID').apply(niu.assign_results)
    yesterday_df = yesterday_df.sort_values('DATE').set_index('DATE')
    yesterday_df['two_week_totals'] = (
        yesterday_df['total_result']
        .rolling('14D')
        .mean()
        .shift(1)
    )
    yesterday_df.reset_index(inplace=True)
    just_yesterday_df = yesterday_df[yesterday_df['DATE'] >= most_recent_historical_date]

    print("### 4. Use most recent historical data to update Elo / Momentum")
    elo_ratings = most_recent_historical.set_index('TEAM')['Elo_Rating'].to_dict()
    variances = most_recent_historical.set_index('TEAM')['Elo_Var'].to_dict()
    momentum_scores = most_recent_historical.set_index('TEAM')['Momentum'].to_dict()

    team_strengths = {
        team: {'mu': elo_ratings[team], 'sigma2': variances[team]}
        for team in elo_ratings
    }

    for _, row in just_yesterday_df.iterrows():
        team_elo, _, team_momentum, opp_elo, _, opp_momentum = (
            niu.update_bayesian_elo_momentum(
                row, just_yesterday_df, team_strengths, momentum_scores
            )
        )
        elo_ratings[row['TEAM']] = team_elo
        momentum_scores[row['TEAM']] = team_momentum
        elo_ratings[row['Opponent']] = opp_elo
        momentum_scores[row['Opponent']] = opp_momentum
        variances[row['TEAM']] = team_strengths[row['TEAM']]['sigma2']
        variances[row['Opponent']] = team_strengths[row['Opponent']]['sigma2']

    print("### 5. Update Offensive/Defensive Ratings")
    off_ratings = most_recent_historical.set_index('TEAM')['Offensive_Rating'].to_dict()
    def_ratings = most_recent_historical.set_index('TEAM')['Defensive_Rating'].to_dict()
    off_variances = most_recent_historical.set_index('TEAM')['Offensive_Var'].to_dict()
    def_variances = most_recent_historical.set_index('TEAM')['Defensive_Var'].to_dict()

    # reinit momentum for O/D rating updates
    momentum_scores = most_recent_historical.set_index('TEAM')['Momentum'].to_dict()

    team_strengths = {
        team: {
            'offense': {'mu': off_ratings[team], 'sigma2': off_variances[team]},
            'defense': {'mu': def_ratings[team], 'sigma2': def_variances[team]}
        }
        for team in elo_ratings
    }

    for _, row in just_yesterday_df.iterrows():
        team_elo, team_elo_d, team_momentum, opp_elo, opp_elo_d, opp_momentum = (
            niu.update_bayesian_off_def(
                row, just_yesterday_df, team_strengths, momentum_scores
            )
        )
        try:
            team_strengths[row['TEAM']]['mu'] = team_elo['mu']
            team_strengths[row['TEAM']]['mu'] = team_elo_d['mu']
            team_strengths[row['TEAM']]['sigma2'] = team_elo['sigma2']
            momentum_scores[row['TEAM']] = team_momentum

            team_strengths[row['Opponent']]['mu'] = opp_elo['mu']
            team_strengths[row['Opponent']]['mu'] = opp_elo_d['mu']
            team_strengths[row['Opponent']]['sigma2'] = opp_elo['sigma2']
            momentum_scores[row['Opponent']] = opp_momentum
        except:
            pass

    print("### 6. Update TODAY_MAP with new Elo / Momentum")
    TODAY_MAP['ELO_Rating'] = TODAY_MAP['TEAM'].map(elo_ratings)
    TODAY_MAP['Offensive_Rating'] = TODAY_MAP['TEAM'].map(team_strengths).apply(
        lambda x: x['offense']['mu'] if isinstance(x, dict) else None
    )
    TODAY_MAP['Defensive_Rating'] = TODAY_MAP['TEAM'].map(team_strengths).apply(
        lambda x: x['defense']['mu'] if isinstance(x, dict) else None
    )
    TODAY_MAP['Momentum'] = TODAY_MAP['TEAM'].map(momentum_scores)
    TODAY_MAP['Opp_Elo'] = TODAY_MAP['Opponent'].map(elo_ratings)
    TODAY_MAP['Opp_Offensive_Rating'] = TODAY_MAP['Opponent'].map(team_strengths).apply(
        lambda x: x['offense']['mu'] if isinstance(x, dict) else None
    )
    TODAY_MAP['Opp_Defensive_Rating'] = TODAY_MAP['Opponent'].map(team_strengths).apply(
        lambda x: x['defense']['mu'] if isinstance(x, dict) else None
    )

    TODAY_MAP[['SPREAD_LINE_MOVEMENT_1',
               'SPREAD_LINE_MOVEMENT_2',
               'SPREAD_LINE_MOVEMENT_3']] = pd.DataFrame(
        TODAY_MAP['Spread_Movement'].to_list(), index=TODAY_MAP.index
    )
    TODAY_MAP[['TOTAL_LINE_MOVEMENT_1',
               'TOTAL_LINE_MOVEMENT_2',
               'TOTAL_LINE_MOVEMENT_3']] = pd.DataFrame(
        TODAY_MAP['Total_Movement'].to_list(), index=TODAY_MAP.index
    )
    TODAY_MAP['Opp_Momentum'] = TODAY_MAP['Opponent'].map(momentum_scores)

    print("### 7. Update rolling stats & collect features for inference")
    today_map_features = [
        'TEAM', 'Opponent', 'MONEYLINE', 'CLOSING_SPREAD', 'CLOSING_TOTAL', 'Venue',
        'Referee', 'ELO_Rating', 'Momentum', 'HOME TEAM WIN%', 'HOME TEAM POINTS DIFFERENTIAL',
        'Opp_Elo', 'Opp_Momentum', 'CREW', 'SPREAD_LINE_MOVEMENT_1', 'SPREAD_LINE_MOVEMENT_2',
        'SPREAD_LINE_MOVEMENT_3', 'TOTAL_LINE_MOVEMENT_1', 'TOTAL_LINE_MOVEMENT_2',
        'TOTAL_LINE_MOVEMENT_3', 'Offensive_Rating', 'Defensive_Rating',
        'Opp_Offensive_Rating', 'Opp_Defensive_Rating'
    ]

    most_recent_tdf = niu.get_rolling_stats(yesterday_df, today_teams_list)
    today_features = TODAY_MAP[today_map_features]
    infer_df = most_recent_tdf.merge(today_features, how='left', on='TEAM')

    infer_df['MAIN REF'] = infer_df['Referee'].astype('category')
    infer_df['TEAM'] = infer_df['TEAM'].astype('category')
    infer_df['CREW'] = infer_df['CREW'].astype('category')
    infer_df['Opponent'] = infer_df['Opponent_y'].astype('category')
    infer_df['TEAM_REST_DAYS'] = infer_df['TEAM_REST_DAYS'].astype('category')

    infer_df['MONEYLINE'] = infer_df['MONEYLINE_y']
    infer_df['VENUE'] = infer_df['Venue']
    infer_df['CLOSING_SPREAD'] = infer_df['CLOSING_SPREAD_y']
    infer_df['CLOSING_TOTAL'] = infer_df['CLOSING_TOTAL_y']
    infer_df['Elo_Rating'] = infer_df['ELO_Rating']
    infer_df['VENUE'] = (infer_df['VENUE'] == 'H') * 1

    infer_df[["MONEYLINE", "Last_ML_1", "Last_ML_2", "Last_ML_3"]] = (
        infer_df[["MONEYLINE", "Last_ML_1", "Last_ML_2", "Last_ML_3"]]
        .replace('even', '-100', regex=True)
        .fillna(0)
        .astype(int)
    )

    temp_df = infer_df.dropna(subset=['TEAM', 'Opponent'])
    infer_df = infer_df[train_cols + ['DATE', 'GAME-ID']]

    print("### 8. Performing spread/ml/total inference")
    total_probabilities = total_model.predict_proba(
        infer_df.drop(['DATE', 'GAME-ID', 'Momentum'], axis=1)
    )[:, 1]
    spread_probabilities = np.zeros(len(total_probabilities))
    ml_probabilities = np.zeros(len(spread_probabilities))

    real_probabilities = {}
    processed_games = set()
    ps = {team: prob for team, prob in zip(temp_df['TEAM'], total_probabilities)}
    for team, opp in zip(temp_df['TEAM'], temp_df['Opponent']):
        game = tuple(sorted([team, opp]))
        if game not in processed_games:
            processed_games.add(game)
            prob_team = ps[team]
            prob_opp = ps[opp]
            average_prob = (prob_team + prob_opp) / 2
            real_probabilities[team] = average_prob
            real_probabilities[opp] = average_prob

    ps = real_probabilities
    spread_predictions = np.array([x > 0.5 for x in spread_probabilities])
    ml_predictions = np.array([x > 0.5 for x in ml_probabilities])
    total_predictions = np.array([x > 0.5 for x in total_probabilities])

    infer_df['spread_prob'] = spread_probabilities
    infer_df['ml_prob'] = ml_probabilities
    infer_df['total_prob'] = total_probabilities

    today_results = infer_df[[
        'TEAM', 'Opponent', 'MONEYLINE', 'CLOSING_SPREAD', 'CLOSING_TOTAL',
        'spread_prob', 'ml_prob', 'total_prob', 'DATE', 'GAME-ID'
    ]].dropna().reset_index(drop=True)

    spread_ps = {
        team: prob for team, prob in zip(today_results['TEAM'].values, spread_probabilities)
    }
    normed_spread_odds = {
        team: spread_ps[team] / (spread_ps[team] + spread_ps[opp])
        for team, opp in zip(today_results['TEAM'], today_results['Opponent'])
    }
    ml_ps = {
        team: prob for team, prob in zip(today_results['TEAM'].values, ml_probabilities)
    }
    normed_ml_odds = {
        team: ml_ps[team] / (ml_ps[team] + ml_ps[opp])
        for team, opp in zip(today_results['TEAM'], today_results['Opponent'])
    }
    total_ps = ps
    normed_total_odds = ps

    today_results['spread_prob_normed'] = today_results['TEAM'].map(normed_spread_odds)
    today_results['ml_prob_normed'] = today_results['TEAM'].map(normed_ml_odds)
    today_results['total_prob_normed'] = today_results['TEAM'].map(normed_total_odds)
    today_results['total_prob'] = today_results['TEAM'].map(normed_total_odds)

    today_results['CLOSING_SPREAD_LINE'] = -110
    today_results['CLOSING_TOTAL_LINE'] = -110
    today_results['spread_implied_prob'] = today_results['CLOSING_SPREAD_LINE'].apply(niu.odds_to_implied_prob)
    today_results['total_implied_prob'] = today_results['CLOSING_TOTAL_LINE'].apply(niu.odds_to_implied_prob)
    today_results['spread_kelly'] = today_results.apply(
        lambda x: niu.kelly_criterion(80, x['spread_prob_normed'], -110, temper=0.5), axis=1
    )
    today_results['total_kelly'] = today_results.apply(
        lambda x: niu.kelly_criterion(80, x['total_prob_normed'], -110, temper=0.5), axis=1
    )

    print("### 9. Print totals predictions (Overs/Unders)")

    print('PREDICTED OVERS')
    over_df = today_results[(today_results['total_prob'] > 0.5)]
    over_df["sorted_pair"] = over_df.apply(lambda x: tuple(sorted([x["TEAM"], x["Opponent"]])), axis=1)
    over_df = over_df.drop_duplicates(subset="sorted_pair", keep="first")

    total_frame = today_results.drop_duplicates(subset=['TEAM'])
    # Filter
    over_display = over_df[(total_frame['total_prob'] > 0.5)].sort_values('total_prob', ascending=False)[
        ['TEAM','Opponent','CLOSING_TOTAL','total_prob','total_kelly','total_prob_normed']
    ]

    try:
        from tabulate import tabulate
        print(tabulate(over_display, headers='keys', tablefmt='psql', showindex=False))
    except ImportError:
        print(over_display.to_string(index=False))

    under_df = total_frame[(total_frame['total_prob'] <= 0.5)]
    under_df['total_prob'] = 1 - under_df['total_prob']
    under_df["sorted_pair"] = under_df.apply(lambda x: tuple(sorted([x["TEAM"], x["Opponent"]])), axis=1)
    under_df = under_df.drop_duplicates(subset="sorted_pair", keep="first")
    print('\nPREDICTED UNDERS\n')

    under_display = under_df.sort_values('total_prob', ascending=False)[
        ['TEAM','Opponent','CLOSING_TOTAL','total_prob','total_prob_normed']
    ]

    try:
        from tabulate import tabulate
        print(tabulate(under_display, headers='keys', tablefmt='psql', showindex=False))
    except ImportError:
        print(under_display.to_string(index=False))

    print("\nDone! The script has run all inference steps.")

if __name__ == "__main__":
    main()
