import argparse
from analytics.MoneylineModel import MoneylineModel
from analytics.Dataset import Dataset
from BacktestEngine import BacktestEngine
import matplotlib.pyplot as plt
import utils

if __name__ == "__main__":
    # Set up argparse
    parser = argparse.ArgumentParser(description="Run backtesting and predict today's games")
    parser.add_argument('--do_backtest', action='store_true', help="Run the backtesting on the date range", required=False)
    parser.add_argument('--get_predictions', action='store_true', help="Get or update today's predictions", required=False)
    parser.add_argument('--start', type=str, required=False, help="Start date for backtesting (format: YYYY-MM-DD)")
    parser.add_argument('--end', type=str, required=False, help="End date for backtesting (format: YYYY-MM-DD)")
    parser.add_argument('--bankroll', type=int, default=100, help="Initial bankroll for backtesting")
    parser.add_argument('--bet_size', type=int, default=10, help="Fixed bet size for backtesting")
    parser.add_argument('--kelly_fraction', type=float, default=0.5, help="Fraction of the Kelly criterion to use for bet sizing")
    parser.add_argument('--do_ensemble', action='store_true', help="Use ensemble approach in the MoneylineModel")

    # Parse the arguments
    args = parser.parse_args()

    # Instantiate the classes with the provided arguments
    model = MoneylineModel(do_ensemble=args.do_ensemble)
    data = Dataset()

    be = BacktestEngine(model=model, 
                        df=data.df, 
                        start=args.start, 
                        end=args.end, 
                        bankroll=args.bankroll, 
                        fixed_bet_size=args.bet_size, 
                        kelly_fraction=args.kelly_fraction, 
                        label_col='Result',
                        odds_col='MONEYLINE', 
                        feature_cols=data.TRAIN_COLS)

    if args.do_backtest:
        final_bankroll, total_profit, bet_results = be.backtest_model()

        # Plotting results
        plt.subplots(figsize=(15,12))
        plt.plot(bet_results)
        plt.grid()
        plt.show()
        print(f"Final Bankroll: {round(final_bankroll, 2)}")
        print(f"Total Profit: {round(total_profit, 2)}")
    
    if args.get_predictions:
        print('get today predictions...')
        utils.predict_today(data, model)

    
