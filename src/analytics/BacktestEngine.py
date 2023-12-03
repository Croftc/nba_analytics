import pandas as pd
import warnings
warnings.filterwarnings('ignore')

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

#BASE_CONFIG = 

class BacktestEngine():

    def __init__(self,
                 model=None, 
                 df=None, 
                 start='2023-10-24', 
                 end='2023-11-25', 
                 bankroll=100, 
                 fixed_bet_size=10, 
                 kelly_fraction=1, 
                 label_col='Result',
                 odds_col='MONEYLINE',
                 feature_cols=None) :
        
        if model == None:
            print('Supply a trained model to backtest')
            exit()
        
        self.model = model
        self.initial_bankroll = bankroll
        self.bankroll = bankroll
        self.fixed_bet_size = fixed_bet_size
        self.kelly_fraction = kelly_fraction 
        self.label_col = label_col
        self.odds_col = odds_col
        self.df = df
        self.start_date = start
        self.end_date = end
        self.feature_cols = feature_cols
        self.thresh = 0.5

        try:
            if df == None:
                print('please provide a dataframe for now')
                exit()
        except:
            pass

    def __handle_bet__(self, prob, thresh, do_bet, team, bet_size, num_bets, actual, to_win, opp, odds, day_profit, wins, losses, hit_all, all_odds, made_a_bet_2, all_all_odds, hit_all_all, pred):
        if (prob > thresh) and do_bet and (bet_size > 0):
            self.bankroll -= bet_size
            num_bets += 1
            if actual:
                print(f'\tWon {to_win} betting {bet_size}\n\t\t{team} to beat {opp} at {odds_to_str(odds)} - our model pinned it at {int(probability_to_american_odds(prob))}', color='green')
                day_profit += (to_win + bet_size)
                wins += 1
            else:
                print(f'\tLost {bet_size} trying to win {to_win}\n\t\t{team} to beat {opp} at {odds_to_str(odds)} - our model pinned it at {int(probability_to_american_odds(prob))}', color='red')
                losses += actual != pred
                hit_all = False
            all_odds.append(odds)
        
        if (prob > thresh) and not do_bet and (bet_size > 0) and actual:
            print(f'\tWanted to win {to_win} betting {bet_size}\n\t\t{team} to beat {opp} at {odds_to_str(odds)} - our model pinned it at {int(probability_to_american_odds(prob))}', color='warning')
        
        if do_bet and ((bet_size < 0) or (prob < 0.5)):
            made_a_bet_2 += 1
            all_all_odds.append(int(odds))
        if not actual:
            hit_all_all = False

        return num_bets, day_profit, wins, losses, hit_all, all_odds, made_a_bet_2, all_all_odds, hit_all_all
        

    def __transform_features__(self, X_in):
        X = X_in[self.feature_cols]
        X[['MONEYLINE', 'Last_ML_1', 'Last_ML_2', 'Last_ML_3']] = X[['MONEYLINE', 'Last_ML_1', 'Last_ML_2', 'Last_ML_3',]].replace('even', '-100', regex=True)
        X[['MONEYLINE', 'Last_ML_1', 'Last_ML_2', 'Last_ML_3',]] = X[['MONEYLINE', 'Last_ML_1', 'Last_ML_2', 'Last_ML_3',]].fillna(0).astype(int)
        X['MAIN REF'] = X['MAIN REF'].astype('category')
        X['CREW'] = X['CREW'].astype('category')
        X['TEAM'] = X['TEAM'].astype('category')
        X['Opponent'] = X['Opponent'].astype('category')
        X['TEAM_REST_DAYS'] = X['TEAM_REST_DAYS'].astype('category')
        X['VENUE'] = (X['VENUE'] == 'H')*1
        y = X_in[self.label_col]

        return X, y


    def backtest_model(self):
        df = self.df
        bankroll = self.initial_bankroll
        self.bet_results = []
        current_date = pd.to_datetime(self.start_date)

        while current_date <= pd.to_datetime(self.end_date):
            print(f'{current_date}...')
            day_data = df[df['DATE'] == current_date]
            start_bankroll = self.bankroll
            day_profit = 0
            if not day_data.empty and self.bankroll > 0:

                X, y = self.__transform_features__(day_data)

                probabilities = self.model.predict_proba(X)
                predictions = probabilities[:, 1] >= self.thresh

                #matchups = [(team, prob1, opp, prob2)]
                do_bet = {team: self.model.predict_proba(X[X['TEAM'] == team])[:, 1] > self.model.predict_proba(X[X['TEAM'] == opp])[:, 1] for team, opp in zip(X['TEAM'], X['Opponent'])}
                normed_odds = {team: self.model.predict_proba(X[X['TEAM'] == team])[:, 1]/(self.model.predict_proba(X[X['TEAM'] == team])[:, 1] + self.model.predict_proba(X[X['TEAM'] == opp])[:, 1]) for team, opp in zip(X['TEAM'], X['Opponent'])}

                wins, losses, profit, num_bets, made_a_bet_2 = 0, 0, 0, 0, 0
                all_odds, all_all_odds = [], []
                hit_all, hit_all_all = True, True
               
                # SUBTRACT ONE FOR THE PARLAY RIGHT AWAY
                made_a_bet = False

                # for all the options today
                for pred, actual, odds, prob, team, opp in zip(predictions, y, day_data[self.odds_col], probabilities[:, 1], X['TEAM'].values, X['Opponent'].values):
                    
                    odds = -100 if odds == 'even' else int(odds)
                    # only look at predicted winners
                    if pred:

                        made_a_bet = True

                        # get optimal bet size
                        bet_size = round(kelly_criterion(start_bankroll, normed_odds[team][0], odds, self.kelly_fraction), 2)

                        # if we can afford to bet on this
                        if (self.bankroll - bet_size) >= 0:

                            # how much would we win
                            to_win = round(calculate_profit(odds, bet_size), 2)

                            # TODO: I know this is ugly but it's worse inline 
                            num_bets, day_profit, wins, \
                            losses, hit_all, all_odds, \
                            made_a_bet_2, all_all_odds, hit_all_all = self.__handle_bet__(prob, 
                                                                                            self.thresh, 
                                                                                            do_bet[team], 
                                                                                            team, 
                                                                                            bet_size, 
                                                                                            num_bets, 
                                                                                            actual, 
                                                                                            to_win, 
                                                                                            opp, 
                                                                                            odds, 
                                                                                            day_profit, 
                                                                                            wins, 
                                                                                            losses, 
                                                                                            hit_all, 
                                                                                            all_odds, 
                                                                                            made_a_bet_2, 
                                                                                            all_all_odds, 
                                                                                            hit_all_all,
                                                                                            pred)

                # if made_a_bet and len(all_odds) > 2:
                #   if hit_all:
                #     total_odds = combine_parlay_odds(all_odds)
                #     bankroll += round(calculate_profit(combine_parlay_odds(all_odds), 10), 2)
                #   else:
                #     bankroll -= 10

                # if made_a_bet_2 > 1:
                #   if hit_all_all:
                #     total_odds = combine_parlay_odds(all_all_odds)
                #     bankroll += round(calculate_profit(combine_parlay_odds(all_all_odds), 10), 2)
                #   else:
                #     bankroll -= 10

                self.bankroll += day_profit
                print_bet_results(current_date, wins, losses, num_bets, self.bankroll, start_bankroll, hit_all, all_odds, hit_all_all, all_all_odds)
                self.bet_results.append(self.bankroll)

            current_date += pd.Timedelta(days=1)

        return self.bankroll, self.bankroll - initial_bankroll, self.bet_results