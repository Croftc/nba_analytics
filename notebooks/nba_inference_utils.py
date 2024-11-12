import time
import pandas as pd
from datetime import datetime
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium import webdriver
from IPython.display import display, HTML
from bs4 import BeautifulSoup
import requests
import os
import numpy as np
from datetime import datetime, timedelta

#mapping of team names to cities
team_map = {
    'Hawks': 'Atlanta',
    'Nets': 'Brooklyn',
    'Celtics': 'Boston',
    'Hornets': 'Charlotte',
    'Bulls': 'Chicago',
    'Cavaliers': 'Cleveland',
    'Mavericks': 'Dallas',
    'Nuggets': 'Denver',
    'Pistons': 'Detroit',
    'Warriors': 'Golden State',
    'Rockets': 'Houston',
    'Pacers': 'Indiana',
    'Clippers': 'LA Clippers',
    'Lakers': 'LA Lakers',
    'Grizzlies': 'Memphis',
    'Heat': 'Miami',
    'Bucks': 'Milwaukee',
    'Timberwolves': 'Minnesota',
    'Pelicans': 'New Orleans',
    'Knicks': 'New York',
    'Thunder': 'Oklahoma City',
    'Magic': 'Orlando',
    '76ers': 'Philadelphia',
    'Suns': 'Phoenix',
    'Trail Blazers': 'Portland',
    'Kings': 'Sacramento',
    'Spurs': 'San Antonio',
    'Raptors': 'Toronto',
    'Jazz': 'Utah',
    'Wizards': 'Washington'
}

cleaned_cols = ['Dataset', 'GAME-ID', 'DATE', 'TEAM', 'VENUE',
                '1Q', '2Q', '3Q', '4Q', 'OT1', 'OT2', 'OT3',
                'OT4', 'OT5', 'F', 'MIN', 'FG', 'FGA', '3P',
                '3PA', 'FT', 'FTA', 'OR', 'DR', 'TOT', 'A',
                'PF', 'ST', 'TO', 'TO_TO', 'BL', 'PTS', 'POSS',
                'PACE', 'OEFF', 'DEFF', 'TEAM_REST_DAYS',
                'STARTER_1', 'STARTER_2', 'STARTER_3', 'STARTER_4',
                'STARTER_5', 'MAIN REF', 'CREW', 'OPENING ODDS',
                'OPENING SPREAD', 'OPENING TOTAL', 'LINE_MOVEMENT_1',
                'LINE_MOVEMENT_2', 'LINE_MOVEMENT_3', 'CLOSING_ODDS',
                'CLOSING_SPREAD', 'CLOSING_TOTAL', 'MONEYLINE', 'HALFTIME',
                'BOX_SCORE_URL', 'ODDS_URL']

def download_current_data(date=None):
    # Define the base URL and parameters
    base_url = "https://www.bigdataball.com/wp-admin/admin-ajax.php?action=outofthebox-download"
    account_id = "dbid:AADL0JM6TbjOPoH-7_QmtAYk4iT4-vis0Tk"
    listtoken = "5a58bb7418a59d0ec0a5558a510e959d"

    # Get current date in the required format
    current_date = datetime.now()
    yesterday = current_date - timedelta(1)
    current_date = yesterday.strftime("%m-%d-%Y") if date == None else date
    filename = f"{current_date}-nba-season-team-feed.xlsx"
    outofthebox_path = f"%2F{filename}"

    # Construct the full URL
    full_url = f"{base_url}&OutoftheBoxpath={outofthebox_path}&lastpath=%2F&account_id={account_id}&listtoken={listtoken}&dl=1"

    # Directory to save the file
    save_dir = "./"
    save_path = os.path.join(save_dir, filename)
    print(save_path)

    # don't redownload if we already have it
    if os.path.exists(save_path):
        return filename

    # Use curl to download the file
    response = requests.get(full_url, stream=True)
    print(response.status_code)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
    return filename

# get team averages
def get_rolling_stats(tdf, today_teams_list):
    tdf['Avg_3_game_PTS'] = tdf.groupby(['TEAM',])['PTS'].transform(lambda x: x.rolling(3).mean())
    tdf['Avg_5_game_PTS'] = tdf.groupby(['TEAM',])['PTS'].transform(lambda x: x.rolling(5).mean())
    tdf['Season_Avg_PTS'] = tdf.groupby(['TEAM',])['PTS'].transform('mean')

    tdf['Avg_3_game_POSS'] = tdf.groupby(['TEAM',])['POSS'].transform(lambda x: x.rolling(3).mean())
    tdf['Avg_5_game_POSS'] = tdf.groupby(['TEAM',])['POSS'].transform(lambda x: x.rolling(5).mean())
    tdf['Season_Avg_POSS'] = tdf.groupby(['TEAM',])['POSS'].transform('mean')

    tdf['Avg_3_game_PACE'] = tdf.groupby(['TEAM',])['PACE'].transform(lambda x: x.rolling(3).mean())
    tdf['Avg_5_game_PACE'] = tdf.groupby(['TEAM',])['PACE'].transform(lambda x: x.rolling(5).mean())
    tdf['Season_Avg_PACE'] = tdf.groupby(['TEAM',])['PACE'].transform('mean')

    tdf['Avg_3_game_OEFF'] = tdf.groupby(['TEAM',])['OEFF'].transform(lambda x: x.rolling(3).mean())
    tdf['Avg_5_game_OEFF'] = tdf.groupby(['TEAM',])['OEFF'].transform(lambda x: x.rolling(5).mean())
    tdf['Season_Avg_OEFF'] = tdf.groupby(['TEAM',])['OEFF'].transform('mean')

    tdf['Avg_3_game_DEFF'] = tdf.groupby(['TEAM',])['DEFF'].transform(lambda x: x.rolling(3).mean())
    tdf['Avg_5_game_DEFF'] = tdf.groupby(['TEAM',])['DEFF'].transform(lambda x: x.rolling(5).mean())
    tdf['Season_Avg_DEFF'] = tdf.groupby(['TEAM',])['DEFF'].transform('mean')

    tdf['Last_ML_1'] = tdf.groupby(['TEAM'])['MONEYLINE'].shift(1)
    tdf['Last_ML_2'] = tdf.groupby(['TEAM'])['MONEYLINE'].shift(2)
    tdf['Last_ML_3'] = tdf.groupby(['TEAM'])['MONEYLINE'].shift(3)

    tdf = tdf.reset_index(drop=True)
    tdf = tdf.groupby('GAME-ID').apply(assign_opp_stats)
    tdf['Wins'] = tdf.groupby(['TEAM'])['Result'].cumsum()
    tdf['Losses'] = tdf.groupby(['TEAM'])['Result'].transform('count') - tdf['Wins']
    tdf['Win_Loss_Diff'] = tdf['Wins'] - tdf['Losses']
    today_date = pd.to_datetime('today').normalize()

    tdf['TEAM_REST_DAYS'] = tdf.apply(lambda row: categorize_rest_days(row['TEAM'], today_date, tdf), axis=1)


    tdf = tdf.sort_values('DATE')

    return get_most_recent_rows(tdf, today_teams_list)

# Assign opponent features
def assign_opp_stats(group):
    group['Result'] = [group.iloc[0]['F'] > group.iloc[1]['F'], group.iloc[1]['F'] > group.iloc[0]['F']]
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

def categorize_rest_days(team, today_date, df):
    # Filter DataFrame for the specific team and sort by date
    team_games = df[df['TEAM'] == team].sort_values(by='DATE')

    # convert DATE to datetime
    team_games['DATE'] = pd.to_datetime(team_games['DATE'])
    
    # Find games up to today's date
    past_games = team_games[team_games['DATE'] < today_date]

    # Check if no games played
    if past_games.empty:
        return 'No games played'

    # Calculate the number of days since the last game
    last_game_date = past_games.iloc[-1]['DATE']
    days_since_last_game = (today_date - last_game_date).days - 1

    # Check for 3 games in 4 days (including B2B scenarios)
    if len(past_games) >= 3 and (today_date - past_games.iloc[-3]['DATE']).days <= 4:
        return '3IN4-B2B' if days_since_last_game == 0 else '3IN4'

    # Check for 4 games in 5 days (including B2B scenarios)
    if len(past_games) >= 4 and (today_date - past_games.iloc[-4]['DATE']).days <= 5:
        return '4IN5-B2B' if days_since_last_game == 0 else '4IN5'

    # Standard cases
    # Check for back-to-back games
    if days_since_last_game == 0:
        return 'B2B'
    if days_since_last_game >= 3:
        return '3+'
    elif days_since_last_game == 2:
        return '2'
    elif days_since_last_game == 1:
        return '1'

    return 'No category'

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

def init_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(options=options)
    return driver

def get_most_recent_rows(df, teams):
    most_recent_rows = []
    for team in teams:
        team_df = df[df['TEAM'] == team]
        if not team_df.empty:
            most_recent_row = team_df.loc[team_df['DATE'].idxmax()]
            most_recent_rows.append(most_recent_row)
    return pd.DataFrame(most_recent_rows)

# Function to update Elo and momentum after each game
def update_elo_momentum(row, elo_ratings, momentum_scores):
    K = 20  # K-factor in Elo rating
    m = 1.2 # m factor in momentum
    momentum_decay = 0.1  # Decay factor for momentum

    team = row['TEAM']
    opponent = row['Opponent']  # Assuming opponent team is passed directly in the row

    team_elo, opponent_elo = elo_ratings[team], elo_ratings[opponent]
    tm, om = momentum_scores[team], momentum_scores[opponent]

    # Calculate expected outcomes
    expected_team = 1 / (1 + 10 ** ((opponent_elo - team_elo) / 400))

    # Actual outcome
    actual_team = row['spread_result']  # 1 for win, 0 for loss

    # Update Elo ratings
    elo_ratings[team] += K * (actual_team - expected_team)

    # Calculate Elo difference
    elo_diff = abs(opponent_elo - team_elo)

    # Update momentum
    momentum_scores[team] = momentum_decay * (momentum_scores[team] + elo_diff) if actual_team == 1 else momentum_decay * (momentum_scores[team] - (elo_diff / m))

    return elo_ratings[team], momentum_scores[team], elo_ratings[opponent], momentum_scores[opponent]

def assign_results(group):

    t1_spread_f = group.iloc[0]['F'] + group.iloc[0]['CLOSING_SPREAD']
    t2_spread_f = group.iloc[1]['F'] + group.iloc[1]['CLOSING_SPREAD']

    min_spread_index = np.argmin([group.iloc[0]['CLOSING_SPREAD'], group.iloc[1]['CLOSING_SPREAD']])
    dog_spread_index = np.argmax([group.iloc[0]['CLOSING_SPREAD'], group.iloc[1]['CLOSING_SPREAD']])

    res = [t1_spread_f > group.iloc[1]['F'], t2_spread_f > group.iloc[0]['F']]

    group['spread_result'] = res
    return group

# assign opponent features
def assign_opps(group):

    t1 = group.iloc[0]['Avg_3_game_DEFF', 'Avg_5_game_DEFF', 'Season_Avg_DEFF', 'Avg_3_game_OEFF', 'Avg_5_game_OEFF', 'Season_Avg_OEFF', 'Avg_3_game_PACE', 'Avg_5_game_PACE', 'Season_Avg_PACE', 'Avg_3_game_POSS', 'Avg_5_game_POSS', 'Season_Avg_POSS', 'Avg_3_game_F', 'Avg_5_game_F', 'Season_Avg_PTS', 'Elo_Rating', 'Momentum']
    t2 = group.iloc[1]['Avg_3_game_DEFF', 'Avg_5_game_DEFF', 'Season_Avg_DEFF', 'Avg_3_game_OEFF', 'Avg_5_game_OEFF', 'Season_Avg_OEFF', 'Avg_3_game_PACE', 'Avg_5_game_PACE', 'Season_Avg_PACE', 'Avg_3_game_POSS', 'Avg_5_game_POSS', 'Season_Avg_POSS', 'Avg_3_game_PTS', 'Avg_5_game_PTS', 'Season_Avg_PTS', 'Elo_Rating', 'Momentum']

    group.iloc[0]['Opp_Avg_3_game_DEFF', 'Opp_Avg_5_game_DEFF', 'Opp_Season_Avg_DEFF', 'Opp_Avg_3_game_OEFF', 'Opp_Avg_5_game_OEFF', 'Opp_Season_Avg_OEFF', 'Opp_Avg_3_game_PACE', 'Opp_Avg_5_game_PACE', 'Opp_Season_Avg_PACE', 'Opp_Avg_3_game_POSS', 'Opp_Avg_5_game_POSS', 'Opp_Season_Avg_POSS', 'Opp_Avg_3_game_PTS', 'Opp_Avg_5_game_PTS', 'Opp_Season_Avg_PTS', 'Opp_Elo', 'Opp_Momentum'] = t2
    group.iloc[1]['Opp_Avg_3_game_DEFF', 'Opp_Avg_5_game_DEFF', 'Opp_Season_Avg_DEFF', 'Opp_Avg_3_game_OEFF', 'Opp_Avg_5_game_OEFF', 'Opp_Season_Avg_OEFF', 'Opp_Avg_3_game_PACE', 'Opp_Avg_5_game_PACE', 'Opp_Season_Avg_PACE', 'Opp_Avg_3_game_POSS', 'Opp_Avg_5_game_POSS', 'Opp_Season_Avg_POSS', 'Opp_Avg_3_game_PTS', 'Opp_Avg_5_game_PTS', 'Opp_Season_Avg_PTS', 'Opp_Elo', 'Opp_Momentum'] = t1

    return group

# Function to scrape the odds and games data
def scrape_odds(today=None):
    driver = init_driver()

    if today is None:
        today = datetime.now().strftime('%Y-%m-%d')
    
    print(f'scraping data for {today}')
    driver.get(f'https://www.scoresandodds.com/nba?date={today}')
    
    # Wait for the page to load
    time.sleep(3)
    
    # Get the table data
    tables = driver.find_elements(By.CLASS_NAME, 'event-card-table')
    
    # Process the tables
    current_odds = {}
    for table in tables:
        rows = table.find_elements(By.CLASS_NAME, 'event-card-row')
        all_moves = [float(''.join(c for c in m.text if (c.isdigit() or c == '.' or c == '-'))) for m in table.find_elements(By.CSS_SELECTOR, '[data-tab*="#line-movements"] .data-value')]
        s_moves, t_moves = [], []
        
        for m in all_moves:
            if abs(m) < 100:
                s_moves.append(m)
            else:
                t_moves.append(m)

        home_row = table.find_element(By.CSS_SELECTOR, '[data-side="home"]')
        away_row = table.find_element(By.CSS_SELECTOR, '[data-side="away"]')

        home_team = home_row.find_element(By.CSS_SELECTOR, '.team-name span').text
        away_team = away_row.find_element(By.CSS_SELECTOR, '.team-name span').text

        # Parsing home team details
        h_ml = parse_moneyline(home_row)
        h_spread = parse_spread(home_row)
        h_total = parse_total(home_row)
        home = ['REF', h_ml, 'H', team_map[away_team], h_spread, h_total, s_moves[-3:], t_moves[-3:], 'CREW', 'UMPIRE']

        # Parsing away team details
        a_ml = parse_moneyline(away_row)
        a_spread = parse_spread(away_row)
        a_total = parse_total(away_row)
        away = ['REF', a_ml, 'R', team_map[home_team], a_spread, a_total, s_moves[-3:], t_moves[-3:], 'CREW', 'UMPIRE']

        # Storing results
        current_odds[team_map[home_team]] = home
        current_odds[team_map[away_team]] = away
    
    driver.quit()
    return current_odds

# Helper functions for parsing the details
def parse_moneyline(row):
    raw_ml = row.find_element(By.CSS_SELECTOR, '[data-field="current-moneyline"] .data-value').text
    return -110 if raw_ml == 'even' else int(raw_ml)

def parse_spread(row):
    return float(''.join(c for c in row.find_element(By.CSS_SELECTOR, '[data-field="current-spread"] .data-value').text if (c.isdigit() or c == '.' or c == '-')))

def parse_total(row):
    return float(''.join(c for c in row.find_element(By.CSS_SELECTOR, '[data-field="current-total"] .data-value').text if (c.isdigit() or c == '.' or c == '-')))

# Parse referee data from HTML content
def parse_referee_data(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    table = soup.find('table', class_='table')
    rows = table.find_all('tr')[1:]  # Skipping the header row

    ref_data = {}
    for row in rows:
        columns = row.find_all('td')
        if len(columns) < 4:
            continue  # Skip rows that don't have enough columns

        game = columns[0].get_text(strip=True)
        crew_chief = columns[1].get_text(strip=True)
        referee = columns[2].get_text(strip=True)
        umpire = columns[3].get_text(strip=True)

        teams = game.split(' @ ')
        if len(teams) != 2:
            continue  # Skip if format is not as expected
        city1 = teams[0]
        city2 = teams[1]

        ref_data[city1] = [crew_chief, referee, umpire]
        ref_data[city2] = [crew_chief, referee, umpire]

    return ref_data

# Helper function to remove "REF" from data if necessary
def remove_ref_keys(data_dict):
    return {k: v for k, v in data_dict.items() if v[0] != 'REF'}


# Function to categorize rest days of a team
def categorize_rest_days(team, today_date, df):
    team_games = df[df['TEAM'] == team].sort_values(by='DATE')
    past_games = team_games[team_games['DATE'] < today_date]
    
    if past_games.empty:
        return 'No games played'

    last_game_date = past_games.iloc[-1]['DATE']
    days_since_last_game = (today_date - last_game_date).days - 1

    if len(past_games) >= 3 and (today_date - past_games.iloc[-3]['DATE']).days <= 4:
        return '3IN4-B2B' if days_since_last_game == 0 else '3IN4'
    if len(past_games) >= 4 and (today_date - past_games.iloc[-4]['DATE']).days <= 5:
        return '4IN5-B2B' if days_since_last_game == 0 else '4IN5'
    
    if days_since_last_game == 0:
        return 'B2B'
    if days_since_last_game >= 3:
        return '3+'
    elif days_since_last_game == 2:
        return '2'
    elif days_since_last_game == 1:
        return '1'

    return 'No category'

# Utility functions for Odds and Bet Calculation
def get_team_odds(team, moneyline_map):
    """Get team odds, handling 'even' and negative/positive conversion."""
    ml = moneyline_map.get(team, 'Even')
    return -100 if ml == 'Even' else int(ml)

def probability_to_american_odds(probability):
    """Convert a probability to American odds format."""
    if probability >= 0.5:
        return round((probability / (1 - probability)) * 100)
    else:
        return round((-100 / (probability)) * -1)

def calculate_profit(odds, bet):
    """Calculate profit given odds and bet size."""
    if odds < 0:
        return round(bet * (100 / abs(odds)), 2)
    else:
        return round(bet * (odds / 100), 2)

# Data preparation functions
def prepare_data(tdf, moneyline_map):
    """Prepare data for prediction, including cleaning and feature mapping."""
    X = tdf.copy()

    #X.columns = cleaned_cols
    #X['Opponent'] = X.groupby('GAME-ID')['TEAM'].shift(-1).fillna(X.groupby('GAME-ID')['TEAM'].shift())
    X['Last_ML_1'] = X.groupby(['TEAM'])['MONEYLINE'].shift(1)
    X['Last_ML_2'] = X.groupby(['TEAM'])['MONEYLINE'].shift(2)
    X['Last_ML_3'] = X.groupby(['TEAM'])['MONEYLINE'].shift(3)
    X['MONEYLINE'] = X['TEAM'].map(moneyline_map)
    X[['MONEYLINE', 'Last_ML_1', 'Last_ML_2', 'Last_ML_3', 'CLOSING_SPREAD', 'CLOSING_TOTAL']] = X[['MONEYLINE', 'Last_ML_1', 'Last_ML_2', 'Last_ML_3', 'CLOSING_SPREAD', 'CLOSING_TOTAL']].replace('Even', '-100', regex=True).replace('--', '-100', regex=True).replace('BLANK_STRING', '-100', regex=True)
    X[['MONEYLINE', 'Last_ML_1', 'Last_ML_2', 'Last_ML_3', 'CLOSING_SPREAD', 'CLOSING_TOTAL']] = X[['MONEYLINE', 'Last_ML_1', 'Last_ML_2', 'Last_ML_3', 'CLOSING_SPREAD', 'CLOSING_TOTAL']].fillna(0).astype(float)

    # Convert categorical columns
    categorical_cols = ['MAIN REF', 'TEAM', 'CREW', 'Opponent', 'TEAM_REST_DAYS']
    for col in categorical_cols:
        X[col] = X[col].astype('category')

    X['VENUE'] = (X['VENUE'] == 'H') * 1  # Convert 'H' to 1 and others to 0

    return X

# Prediction functions
def make_pick(team, opp, prob, opp_prob, bet, moneyline_map, normed_odds, elo, mom):
    """Make a prediction for the team and its opponent."""
    team_odds = get_team_odds(team, moneyline_map)
    opp_odds = get_team_odds(opp, moneyline_map)

    # Calculate our line and opponent's line
    our_line = probability_to_american_odds(normed_odds[team])
    opp_line = probability_to_american_odds(normed_odds[opp])

    # Calculate bet sizing using Kelly Criterion
    bet_size = kelly_criterion(100, normed_odds[team], team_odds, temper=0.2)

    # Formatting and displaying the bet info
    display_bet_info(team, opp, team_odds, opp_odds, bet_size, elo, mom, our_line, opp_line)

def display_bet_info(team, opp, team_odds, opp_odds, bet_size, elo, mom, our_line, opp_line, normed_odds):
    """Display bet information in a formatted HTML layout."""
    win_color = 'green' if bet_size > 0 else 'orange'
    lose_color = 'red' if bet_size > 0 else 'orange'

    bet_info = f"Straight bet {bet_size}u to win {calculate_profit(team_odds, bet_size):.2f}u" if bet_size > 0 else "Don't bet this straight - parlay only"

    bet_html = f"""
        <div style="display:flex">
            <div style="margin-left:3%; width: 400px">
                <h2 style="color:{win_color}">{team_odds} : {team} : {our_line}</h2>
                <h3>{normed_odds[team] * 100:.2f}% win probability</h3>
                <h3>{bet_info}</h3>
                <h3> Team Rating: {int(elo)} </h3>
                <h3> Momentum: {int(mom)} </h3>
            </div>
            <div style="width: 400px">
                <h2 style="color:{lose_color}">{opp_odds} : {opp} : {opp_line}</h2>
                <h3>{normed_odds[opp] * 100:.2f}% win probability</h3>
                <h3>Don't bet on this</h3>
                <h3> Team Rating: {int(elo)} </h3>
                <h3> Momentum: {int(mom)} </h3>
            </div>
        </div>
    """
    display(HTML(bet_html))



