import pandas as pd
from IPython.display import HTML

def prdict_today(data):
    TODAY_MAP = data.get_today_data()
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

    ref_map = {team: details[0] for team, details in data.TODAY_MAP.items()}
    moneyline_map = {team: details[1] for team, details in data.TODAY_MAP.items()}
    venue_map = {team: details[2] for team, details in data.TODAY_MAP.items()}
    opp_map = {team: details[3] for team, details in data.TODAY_MAP.items()}
    spread_map = {team: details[4] for team, details in data.TODAY_MAP.items()}
    total_map = {team: details[5] for team, details in data.TODAY_MAP.items()}


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
    temp = temp.drop_duplicates(subset='TEAM')[t_train_cols]

    temp = temp[temp['TEAM'].isin(t_teams)]
    temp.columns = train_cols_final

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
    odds = X['MONEYLINE'].values
    booster = best_model.get_booster()
    normed_odds = {team: best_model.predict_proba(X[X['TEAM'] == team])[:, 1]/(best_model.predict_proba(X[X['TEAM'] == team])[:, 1] + best_model.predict_proba(X[X['TEAM'] == opp])[:, 1]) for team, opp in zip(X['TEAM'], X['Opponent'])}
    do_bet = {team: normed_odds[team] > normed_odds[opp] for team, opp in zip(X['TEAM'], X['Opponent'])}

    pred_contribs = booster.predict(DMatrix(X, enable_categorical=True), pred_contribs=True)


    for team, win, prob, opp, contribs, elo, mom in zip(X['TEAM'].values, best_model.predict(X), best_model.predict_proba(X)[:, 1], X['Opponent'].values, pred_contribs[:, :-1], X['Elo_Rating'].values, X['Momentum'].values):
        # get the most important features
        helpers = np.array(TRAIN_COLS)[np.argpartition(contribs, -3)[-3:]]
        detractions = (np.array(TRAIN_COLS)[np.argpartition(contribs, -3)[:3]])

        # get this team odds
        o = -100 if moneyline_map[team] == 'Even' else int(moneyline_map[team])
        odd = str(o) if o < 0 else f'+{o}'

        # get opp odds
        o2 = -100 if moneyline_map[opp] == 'Even' else int(moneyline_map[opp])
        odd2 = str(o2) if o2 < 0 else f'+{o2}'

        # get out odds
        our_line = probability_to_american_odds(normed_odds[team][0])
        our_line = str(our_line) if our_line < 0 else f'+{our_line}'

        # get our opp odds
        our_opp_line = probability_to_american_odds(normed_odds[opp][0])
        our_opp_line = str(our_opp_line) if our_opp_line < 0 else f'+{our_opp_line}'

        # get the bet sizing
        bet = kelly_criterion(100, normed_odds[team][0], o, temper=0.5)

        # tab character for spacing the prints
        tab = '&nbsp;&nbsp;&nbsp;&nbsp;'

        # make picks
        if (bet >= 0) and do_bet[team]:
            win_color = 'green'
            lose_color = 'red'
            b = f'Stright bet {round(bet, 2)}u to win {round(calculate_profit(o, round(bet, 2)),2)}u' if round(bet, 2) > 0 else 'Don\'t bet this straight - parlay only'

            t = HTML(f"""<div style="display:flex"> \
                        <div style="margin-left:3%; width: 400px"> \
                            <h2 style="color:{win_color}">{odd} : {team} : {our_line}</h2> \
                            <h3>{round(normed_odds[team][0]*100, 2)}% win probability</h3> \
                            <h3>{b}</h3>
                            <h3> Team Rating: {int(elo)} </h3>
                            <h3> Momentum: {int(mom)} </h3>
                            <h3> Best Features: </h3>
                            <h4> - {tab + helpers[0]} </h4>
                            <h4> - {tab + helpers[1]} </h4>
                            <h4> - {tab + helpers[2]} </h4>
                        </div>

                        <div style="width: 400px"> \
                            <h2 style="color:{lose_color}">{odd2} : {opp} : {our_opp_line}</h2> \
                            <h3>{round(normed_odds[opp][0]*100, 2)}% win probability</h3> \
                            <h3>Don\'t bet on this</h3> \
                            <h3> Team Rating: {int(X[X["TEAM"] == opp]["Elo_Rating"])} </h3>
                            <h3> Momentum: {int(X[X["TEAM"] == opp]["Momentum"])} </h3>
                            <h3> Opp Mitigations: </h3>
                            <h4> - {tab + detractions[0]} </h4>
                            <h4> - {tab + detractions[1]} </h4>
                            <h4> - {tab + detractions[2]} </h4>
                        </div> \
                    </div>""")
            display(t)
            print('___________________________________________________________________________________________________________')

        elif do_bet[team] and (bet < 0):
            win_color = '#E4CD05'
            lose_color = 'orange'
            b = f'Stright bet {round(bet, 2)}u to win {round(calculate_profit(o, round(bet, 2)),2)}u' if round(bet, 2) > 0 else 'Don\'t bet this straight - parlay only'
            t = HTML(f"""<div style="display:flex"> \
                        <div style="margin-left:3%; width: 400px"> \
                            <h2 style="color:{win_color}">{odd} : {team} : {our_line}</h2> \
                            <h3>{round(normed_odds[team][0]*100, 2)}% win probability</h3> \
                            <h3>{b}</h3>
                            <h3> Team Rating: {int(elo)} </h3>
                            <h3> Momentum: {int(mom)} </h3>
                            <h3> Best Features: </h3>
                            <h4> - {tab + helpers[0]} </h4>
                            <h4> - {tab + helpers[1]} </h4>
                            <h4> - {tab + helpers[2]} </h4>
                        </div>

                        <div style="width: 400px"> \
                            <h2 style="color:{lose_color}">{odd2} : {opp} : {our_opp_line}</h2> \
                            <h3>{round(normed_odds[opp][0]*100, 2)}% win probability</h3> \
                            <h3>Don\'t bet on this</h3> \
                            <h3> Team Rating: {int(X[X["TEAM"] == opp]["Elo_Rating"])} </h3>
                            <h3> Momentum: {int(X[X["TEAM"] == opp]["Momentum"])} </h3>
                            <h3> Opp Mitigations: </h3>
                            <h4> - {tab + detractions[0]} </h4>
                            <h4> - {tab + detractions[1]} </h4>
                            <h4> - {tab + detractions[2]} </h4>
                        </div> \
                    </div>""")
            display(t)
            print('___________________________________________________________________________________________________________')


            print('Done')