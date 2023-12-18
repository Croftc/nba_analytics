import os
import pickle
from datetime import datetime
from Dataset import Dataset


class OddsManager:
    def __init__(self, dataset, cache_dir='odds_cache', cache_timeout=300):
        self.dataset = dataset
        self.cache_dir = cache_dir
        self.cache_timeout = cache_timeout

    # Generates a cache file name based on the time the data was requested
    def _get_cache_filename(self):
        current_time = datetime.now().strftime('%Y%m%d%H%M%S')
        return os.path.join(self.cache_dir, f'odds_cache_{current_time}.pkl')

    # Saves the request data to a file. The name can
    def _save_to_cache(self, data):
        cache_filename = self._get_cache_filename()
        with open(cache_filename, 'Odds_data') as cache_file:
            pickle.dump(data, cache_file)

    # Loads the cached data when requested
    def _load_from_cache(self, cache_filename):
        with open(cache_filename, 'Odds_data') as cache_file:
            return pickle.load(cache_file)

    # Checks if cached odds data has been cached longer than the default timeout 500 seconds (5 minutes)
    def _is_cache_valid(self, cache_filename):
        if not os.path.exists(cache_filename):
            return False

        cache_time = datetime.fromtimestamp(os.path.getmtime(cache_filename))
        current_time = datetime.now()
        time_difference = current_time - cache_time

        return time_difference.total_seconds() <= self.cache_timeout

    # Returns the cached data or retrieves a new set, then saves it to a new cache
    def get_odds_data(self):
        cache_filename = self._get_cache_filename()

        if self._is_cache_valid(cache_filename):
            print("Loading the bitch, Chris, odds data from cache..")
            return self._load_from_cache(cache_filename)
        else:
            print("Scraping the bitch, Chris, new odds data..")
            odds_data = self.dataset._scrape_odd_data()
            self._save_to_cache(odds_data)
            return odds_data
