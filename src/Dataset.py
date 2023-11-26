from copy import deepcopy
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
from bs4 import BeautifulSoup
import pandas as pd
import os
import json
from datetime import datetime
import shutil
import requests

chrome_options = Options()
chrome_options.add_argument("--headless=new") 

class Dataset():
    
    def __init__(self, 
                    headless=True, 
                    odds_url='https://espnbet.com/sport/basketball/organization/united-states/competition/nba/featured-page', 
                    refs_url='https://official.nba.com/referee-assignments/',
                    scrape_live_data=False):

        self.data_is_processed = False
        self.odds_config = {'teams_div_class': 'flex p-0'}
        self.odds_url = odds_url
        self.refs_url = refs_url
        self.js_delay = 5
        self.chrome_options = chrome_options if headless else None
        self.today_odds_data = None
        self.today_refs_data = None    
        self.historical_data_dir = './historical_data/'
        self.yesterday_data_dir = './live_data/'
        self.today_data_dir = './live_data/'
        self.TODAY_FILE = self.__download_current_data__('11-25-2023')
        self.today_data_file = f'./{self.today_data_dir}{self.TODAY_FILE}.json'
        self.column_mappings = None
        self.column_mappings_file = './config/column_mappings.json'
        self.filename = './live_data/dataframe.pkl'

        with open(self.column_mappings_file, 'r') as file:
            self.column_mappings = json.load(file)

        # Now, you can access your column lists like this
        self.COLS = self.column_mappings['COLS']
        self.cleaned_cols = self.column_mappings['cleaned_cols']
        self.t_cleaned_cols = self.column_mappings['t_cleaned_cols']
        self.TRAIN_COLS = self.column_mappings['TRAIN_COLS']
        self.today_mappings = self.column_mappings['today_mappings']
        self.TARGET = self.column_mappings['TARGET']
        self.SEASON_MAP = self.column_mappings['SEASON_MAP']
        self.t_train_cols = self.column_mappings['t_train_cols']
        self.train_cols_final = self.column_mappings['train_cols_final']

        # Initialize Elo ratings and momentum scores
        self.elo_ratings = {}
        self.momentum_scores = {}
        self.K = 20
        self.momentum_decay = 0.9
        self.team_city_map = {
                                'ATL Hawks': 'Atlanta',
                                'BKN Nets': 'Brooklyn',
                                'BOS Celtics': 'Boston',
                                'CHA Hornets': 'Charlotte',
                                'CHI Bulls': 'Chicago',
                                'CLE Cavaliers': 'Cleveland',
                                'DAL Mavericks': 'Dallas',
                                'DEN Nuggets': 'Denver',
                                'DET Pistons': 'Detroit',
                                'GS Warriors': 'Golden State',
                                'HOU Rockets': 'Houston',
                                'IND Pacers': 'Indiana',
                                'LA Clippers': 'LA Clippers',
                                'LA Lakers': 'LA Lakers',
                                'L.A. Lakers': 'LA Lakers',
                                'MEM Grizzlies': 'Memphis',
                                'MIA Heat': 'Miami',
                                'MIL Bucks': 'Milwaukee',
                                'MIN Timberwolves': 'Minnesota',
                                'NO Pelicans': 'New Orleans',
                                'NY Knicks': 'New York',
                                'OKC Thunder': 'Oklahoma City',
                                'ORL Magic': 'Orlando',
                                'PHI 76ers': 'Philadelphia',
                                'PHX Suns': 'Phoenix',
                                'POR Trail Blazers': 'Portland',
                                'SAC Kings': 'Sacramento',
                                'SA Spurs': 'San Antonio',
                                'TOR Raptors': 'Toronto',
                                'UTA Jazz': 'Utah',
                                'WSH Wizards': 'Washington'
                            }

        self.rename_map = {
            'OPENING_ODDS': 'OPENING ODDS',
            'FULL_GAME_ODDS_URL': 'ODDS_URL'
        }

        if scrape_live_data:
            self.today_data = self.get_today_data()
        else:
            self.today_data = None
        
        self.historical_data = self.get_historical_data()

    def __download_current_data__(self, date=None):
        # Define the base URL and parameters
        base_url = "https://www.bigdataball.com/wp-admin/admin-ajax.php?action=outofthebox-download"
        account_id = "dbid:AADL0JM6TbjOPoH-7_QmtAYk4iT4-vis0Tk"
        listtoken = "421f46cd8fe7a43b705e438648517e48"
        
        # Get current date in the required format
        current_date = datetime.now().strftime("%m-%d-%Y") if date == None else date
        filename = f"{current_date}-nba-season-team-feed.xlsx"
        outofthebox_path = f"%2F{filename}"

        # Construct the full URL
        full_url = f"{base_url}&OutoftheBoxpath={outofthebox_path}&lastpath=%2F&account_id={account_id}&listtoken={listtoken}&dl=1"

        # Directory to save the file
        save_dir = '.\live_data'
        save_path = os.path.join(save_dir, filename)

        # don't redownload if we already have it
        if os.path.exists(save_path):
            return filename

        # Clear out the save directory
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir)

        # Use curl to download the file
        response = requests.get(full_url, stream=True)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
        return filename

                            

    def __scrape_odds_data__(self):
        # Initialize the WebDriver
        driver = webdriver.Chrome(options=self.chrome_options)

        # Navigate to the URL
        driver.get(self.odds_url)

        # Wait for JavaScript to load
        time.sleep(self.js_delay)

        # Get the HTML content of the page
        html_content = driver.page_source

        # Close the browser
        driver.quit()

        # Parse the HTML using BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')

        # Find all relevant div elements
        elements = soup.find_all('div', class_=self.odds_config['teams_div_class'])

        data = []

        for element in elements:
            try:
                team_info = element.find('button', {'data-testid': 'team-name'})
                if team_info:
                    team_name = team_info.find('div', class_='text-primary text-description text-primary').get_text(strip=True)
                    team_stats = team_info.find('div', class_='text-subdued-primary mt-0.5 text-footnote').get_text(strip=True)

                bets = element.find_all('button', {'data-dd-action-name': 'Add Bet Selections'})
                bet_data = []
                for bet in bets:
                    bet_text = bet.get_text(strip=True).split()
                    bet_type = bet_text[0] if bet_text else None
                    bet_value = bet_text[1] if len(bet_text) > 1 else None
                    bet_data.append({'type': bet_type, 'value': bet_value})

                data.append({'team_name': team_name, 'team_stats': team_stats, 'bets': bet_data})
            except:
                pass
        
        
        for row in data:
            print(f"found team: {row['team_name']} at {row['bets'][-1]['type']}")

        output = {}

        for i, item in enumerate(data):
            city = self.team_city_map.get(item['team_name'], 'Unknown City')
            bets = item['bets']

            opp_ind = i - 1 if i%2 == 1 else i + 1
            venue = 'H' if i % 2 == 1 else 'R'

            # Initialize default values
            moneyline = spread = total = 'BLANK_STRING'

            # Extract the moneyline, spread, and total from the bets
            if bets:
                # Moneyline is the value of 'type' in the last object
                moneyline = bets[-1]['type'] if bets[-1]['value'] is None else bets[-1]['value']

                # Spread is the value before the '-' in the first object
                spread_data = bets[0]['type'][:4] if bets[0]['type'] else ''
                spread = spread_data if spread_data else 'BLANK_STRING'

                # Total is the value before the '-' in the second object
                total_data = bets[1]['value'].split('-')[0] if len(bets) > 1 and bets[1]['value'] else ''
                total = total_data if total_data else 'BLANK_STRING'

            # Construct the output format
            output[city] = ['REF', moneyline, venue, self.team_city_map.get(data[opp_ind]['team_name']), spread, total]

        return output


    def __scrape_refs_data__(self):
        
        # Initialize the WebDriver
        driver = webdriver.Chrome(options=self.chrome_options)

        # Navigate to the URL
        driver.get(self.refs_url)

        # Wait for JavaScript to load
        time.sleep(self.js_delay)

        # Get the HTML content of the page
        html_content = driver.page_source

        # Close the browser
        driver.quit()

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

            # Split the game into two teams
            teams = game.split(' @ ')
            if len(teams) != 2:
                continue  # Skip if format is not as expected
            city1 = teams[0] 
            city2 = teams[1] 

            ref_data[city1] = [crew_chief, referee, umpire]
            ref_data[city2] = [crew_chief, referee, umpire]

        return ref_data


    def __read_historical_data_files__(self):

        # main data
        team_df_2019 = pd.read_excel(f'{self.historical_data_dir}2018-2019_NBA_Box_Score_Team-Stats.xlsx')
        team_df_2020 = pd.read_excel(f'{self.historical_data_dir}2019-2020_NBA_Box_Score_Team-Stats.xlsx')
        team_df_2021 = pd.read_excel(f'{self.historical_data_dir}2020-2021_NBA_Box_Score_Team-Stats.xlsx')
        team_df_2022 = pd.read_excel(f'{self.historical_data_dir}2021-2022_NBA_Box_Score_Team-Stats.xlsx')
        team_df_2023 = pd.read_excel(f'{self.historical_data_dir}2022-2023_NBA_Box_Score_Team-Stats.xlsx')
        self.df = pd.concat([team_df_2019, team_df_2020, team_df_2021, team_df_2022, team_df_2023])

        # yesterday's data
        self.ydf = pd.read_excel(f'{self.yesterday_data_dir}{self.TODAY_FILE}')

        # refs data
        refs_2019 = pd.read_csv(f'{self.historical_data_dir}2018-2019.csv')
        refs_2020 = pd.read_csv(f'{self.historical_data_dir}2019-2020.csv')
        refs_2021 = pd.read_csv(f'{self.historical_data_dir}2020-2021.csv')
        refs_2022 = pd.read_csv(f'{self.historical_data_dir}2021_2022.csv')
        refs_2023 = pd.read_csv(f'{self.historical_data_dir}2022-2023.csv')
        self.refs_df = pd.concat([refs_2019, refs_2020, refs_2021, refs_2022, refs_2023])
        
        # get the columns right
        self.df.columns = self.cleaned_cols
        self.ydf.columns = self.t_cleaned_cols
        self.ydf.rename(columns=self.rename_map, inplace=True)

        # sdd missing columns
        missing_cols = set(self.cleaned_cols) - set(self.ydf.columns)
        for col in missing_cols:
            self.ydf[col] = pd.NA


        # concatenate the DataFrames
        self.df = pd.concat([self.df, self.ydf], ignore_index=True)
        self.df['Season'] = self.df['BIGDATABALL_DATASET'].map(self.SEASON_MAP)
    
        return self.df

    def get_ydf(self):
        self.ydf = pd.read_excel(f'{self.yesterday_data_dir}{self.TODAY_FILE}')
        return self.ydf

    def get_refs_data(self):
        # refs data
        refs_2019 = pd.read_csv(f'{self.historical_data_dir}2018-2019.csv')
        refs_2020 = pd.read_csv(f'{self.historical_data_dir}2019-2020.csv')
        refs_2021 = pd.read_csv(f'{self.historical_data_dir}2020-2021.csv')
        refs_2022 = pd.read_csv(f'{self.historical_data_dir}2021_2022.csv')
        refs_2023 = pd.read_csv(f'{self.historical_data_dir}2022-2023.csv')
        self.refs_df = pd.concat([refs_2019, refs_2020, refs_2021, refs_2022, refs_2023])
        return self.refs_df


    # Features that require looking at the opponent
    def __set_group_features__(self, group):

        # main ref and CREW
        group['MAIN REF'] = [group['MAIN REF'].iloc[0]]*2
        group['CREW'] = [group['CREW'].iloc[0]]*2

        # Result - TODO:
        # for now this is just hardcoded as the result of the game...
        # but should be abstracted to accept an arbitrary function
        # which computes a "Result"... the trick will then be to 
        # dynamically provide the correct odds to the backtesting / inference
        if group.iloc[0]['PTS'] > group.iloc[1]['PTS']:
            group['Result'] = [1, 0]
        else:
            group['Result'] = [0, 1]

        # Opponent averages
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

    # get running averages - TODO:
    # should abstract this to take maybe a list of windows to compute the averages over
    # need to be careful about shifting backwards to not leak future data!
    def __set_running_averages__(self):
        self.df['Avg_3_game_PTS'] = self.df.groupby(['TEAM', 'Season'])['PTS'].transform(lambda x: x.shift(1).rolling(3).mean())
        self.df['Avg_5_game_PTS'] = self.df.groupby(['TEAM', 'Season'])['PTS'].transform(lambda x: x.shift(1).rolling(5).mean())
        self.df['Season_Avg_PTS'] = self.df.groupby(['TEAM', 'Season'])['PTS'].transform('mean')

        self.df['Avg_3_game_POSS'] = self.df.groupby(['TEAM', 'Season'])['POSS'].transform(lambda x: x.shift(1).rolling(3).mean())
        self.df['Avg_5_game_POSS'] = self.df.groupby(['TEAM', 'Season'])['POSS'].transform(lambda x: x.shift(1).rolling(5).mean())
        self.df['Season_Avg_POSS'] = self.df.groupby(['TEAM', 'Season'])['POSS'].transform('mean')

        self.df['Avg_3_game_PACE'] = self.df.groupby(['TEAM', 'Season'])['PACE'].transform(lambda x: x.shift(1).rolling(3).mean())
        self.df['Avg_5_game_PACE'] = self.df.groupby(['TEAM', 'Season'])['PACE'].transform(lambda x: x.shift(1).rolling(5).mean())
        self.df['Season_Avg_PACE'] = self.df.groupby(['TEAM', 'Season'])['PACE'].transform('mean')

        self.df['Avg_3_game_OEFF'] = self.df.groupby(['TEAM', 'Season'])['OEFF'].transform(lambda x: x.shift(1).rolling(3).mean())
        self.df['Avg_5_game_OEFF'] = self.df.groupby(['TEAM', 'Season'])['OEFF'].transform(lambda x: x.shift(1).rolling(5).mean())
        self.df['Season_Avg_OEFF'] = self.df.groupby(['TEAM', 'Season'])['OEFF'].transform('mean')

        self.df['Avg_3_game_DEFF'] = self.df.groupby(['TEAM', 'Season'])['DEFF'].transform(lambda x: x.shift(1).rolling(3).mean())
        self.df['Avg_5_game_DEFF'] = self.df.groupby(['TEAM', 'Season'])['DEFF'].transform(lambda x: x.shift(1).rolling(5).mean())
        self.df['Season_Avg_DEFF'] = self.df.groupby(['TEAM', 'Season'])['DEFF'].transform('mean')

    # Win/Loss Streak
    def __calculate_streak__(self, group):
        streak = group['Prev_Result'].diff().ne(0).cumsum()
        group['Streak'] = streak.groupby(streak).cumcount()
        group['Streak'] *= group['Prev_Result'].map({1: 1, 0: -1})
        return group

    def __update_elo_momentum__(self, row):
        team = row['TEAM']
        opponent = self.df[(self.df['GAME-ID'] == row['GAME-ID']) & (self.df['TEAM'] != team)]['TEAM'].values[0]
        team_elo, opponent_elo = self.elo_ratings[team], self.elo_ratings[opponent]

        # Calculate expected outcomes
        expected_team = 1 / (1 + 10 ** ((opponent_elo - team_elo) / 400))

        # Actual outcome
        actual_team = 1 if row['PTS'] > self.df[(self.df['GAME-ID'] == row['GAME-ID']) & (self.df['TEAM'] == opponent)]['PTS'].values[0] else 0

        # Update Elo ratings
        self.elo_ratings[team] += self.K * (actual_team - expected_team)
        self.elo_ratings[opponent] += self.K * ((1 - actual_team) - (1 - expected_team))

        # Calculate Elo difference
        elo_diff = opponent_elo - team_elo

        # Update momentum
        self.momentum_scores[team] = self.momentum_decay * self.momentum_scores[team] + elo_diff * actual_team

        return self.elo_ratings[team], self.momentum_scores[team]

    def __process_historical_data__(self):

        # get the base historical data (up to yesterday!)
        print('reading historical data...')
        self.__read_historical_data_files__()

        # get last 3, 5 and full season avg:
        # PTS, POSS, PACE, OEFF, DEFF
        print('setting running averages...')
        self.__set_running_averages__()

        # get the grouped features for each game:
        # ref, crew, result, opponent averages
        print('setting group features...')
        self.df = self.df.groupby('GAME-ID').apply(self.__set_group_features__)

        # Reset index if needed
        self.df.reset_index(drop=True, inplace=True)

        # Shift the Result column for streak calculation
        self.df['Prev_Result'] = self.df.groupby(['TEAM', 'Season'])['Result'].shift()

        print('calculating streaks...')
        self.df = self.df.groupby(['TEAM', 'Season']).apply(self.__calculate_streak__)

        # Last 3 Games Moneylines
        print('looking at previous moneylines...')
        self.df['Last_ML_1'] = self.df.groupby(['TEAM', 'Season'])['MONEYLINE'].shift(1)
        self.df['Last_ML_2'] = self.df.groupby(['TEAM', 'Season'])['MONEYLINE'].shift(2)
        self.df['Last_ML_3'] = self.df.groupby(['TEAM', 'Season'])['MONEYLINE'].shift(3)

        # Current Number of Wins - Losses
        print('lokking at records...')
        self.df['Wins'] = self.df.groupby(['TEAM', 'Season'])['Result'].cumsum()
        self.df['Losses'] = self.df.groupby(['TEAM', 'Season'])['Result'].transform('count') - self.df['Wins']
        self.df['Win_Loss_Diff'] = self.df['Wins'] - self.df['Losses']

        # Current Opponent
        print('setting opponents...')
        self.df['Opponent'] = self.df.groupby('GAME-ID')['TEAM'].shift(-1).fillna(self.df.groupby('GAME-ID')['TEAM'].shift())

        # Clean up and remove the temporary 'Prev_Result' column
        print('starting cleanup...')
        self.df.drop('Prev_Result', axis=1, inplace=True)
        print('merging...')
        self.df = self.df.merge(self.refs_df.groupby('REFEREE').mean(), how='left', left_on='MAIN REF', right_on='REFEREE')
        self.df['DATE'] = self.df['DATE'].astype('datetime64[ns]')
        print('sorting...')
        self.df = self.df.sort_values('DATE')
        self.elo_ratings = {team: 1500 for team in self.df['TEAM'].unique()}
        self.momentum_scores = {team: 0 for team in self.df['TEAM'].unique()}
        print('calculating ELO and Momentum...')
        self.df[['Elo_Rating', 'Momentum']] = self.df.apply(self.__update_elo_momentum__, axis=1, result_type='expand')
        self.data_is_processed = True
        print('DONE')
        self.__save_dataframe__()
        return self.df

    def __save_dataframe__(self):
        # Save the DataFrame to a file
        self.df.to_pickle(self.filename)

    def __load_dataframe__(self):
        # Load the DataFrame from a file
        self.df = pd.read_pickle(self.filename)

    def __is_file_current__(self):
        # Check if the file exists and was created today
        if os.path.exists(self.filename):
            file_creation_date = datetime.fromtimestamp(os.path.getctime(self.filename)).date()
            return file_creation_date == datetime.now().date()
        return False

    def get_today_data(self, force_scrape=False):

        # don't scrape if we don't have to 
        if os.path.exists(self.today_data_file) and not force_scrape:
            with open(self.today_data_file,'r') as json_file:
                self.today_data = json.load(json_file)
                return self.today_data

        elif not force_scrape:
            print("Please enable webscraping")
            exit()

        else:
            print(f"scraping data")
            self.today_odds_data = self.__scrape_odds_data__()
            self.today_refs_data = self.__scrape_refs_data__()
            self.today_data = deepcopy(self.today_odds_data)

            for city, refs in self.today_refs_data.items():
                try:
                    if city == 'L.A. Lakers':
                        city = 'LA Lakers'
                    self.today_data[city][0] = ' '.join(refs[0].split(' ')[:-1])
                except:
                    pass

            with open(self.today_data_file,'w') as fp:
                fp.write(json.dumps(self.today_data))

            return self.today_data

    def get_historical_data(self):
        if self.data_is_processed:
            print('Have this in memory - here it is')
            return self.df
        elif self.__is_file_current__():
            print("Loading precomputed DataFrame.")
            return self.__load_dataframe__()
        else:
            print("Computing new DataFrame.")
            return self.__process_historical_data__()

        



