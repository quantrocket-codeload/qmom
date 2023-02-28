# Copyright 2020 QuantRocket LLC - All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd
from moonshot import Moonshot
from moonshot.commission import PerShareCommission

class USStockCommission(PerShareCommission):
    BROKER_COMMISSION_PER_SHARE = 0.005

class QuantitativeMomentum(Moonshot):
    """
    Momentum strategy modeled on Alpha Architect's QMOM ETF.

    Strategy rules:

    1. Universe selection
      a. Starting universe: all NYSE stocks
      b. Exclude financials, ADRs, REITs
      c. Liquidity screen: select top N percent of stocks by dollar
         volume (N=60)
    2. Apply momentum screen: calculate 12-month returns, excluding
       most recent month, and select N percent of stocks with best
       return (N=10)
    3. Filter by smoothness of momentum: of the momentum stocks, select
       the N percent with the smoothest momentum, as measured by the number
       of positive days in the last 12 months (N=50)
    4. Apply equal weights
    5. Rebalance portfolio before quarter-end to capture window-dressing seasonality effect
    """

    CODE = "qmom"
    DB = "sharadar-us-stk-1d"
    DB_FIELDS = ["Close", "Volume"]
    DOLLAR_VOLUME_TOP_N_PCT = 60
    DOLLAR_VOLUME_WINDOW = 90
    UNIVERSES = "nyse-stk"
    EXCLUDE_UNIVERSES = ["nyse-financials", "nyse-adrs", "nyse-reits"]
    MOMENTUM_WINDOW = 252
    MOMENTUM_EXCLUDE_MOST_RECENT_WINDOW = 22
    TOP_N_PCT = 10
    SMOOTHEST_TOP_N_PCT = 50
    REBALANCE_INTERVAL = "Q-NOV" #  = end of quarter, fiscal year ends in Nov (= Nov 30, Feb 28, May 31, Aug 31); https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#anchored-offsets
    COMMISSION_CLASS = USStockCommission

    def prices_to_signals(self, prices: pd.DataFrame):

        # Step 1.c: get a mask of stocks with adequate dollar volume
        closes = prices.loc["Close"]
        volumes = prices.loc["Volume"]
        avg_dollar_volumes = (closes * volumes).rolling(self.DOLLAR_VOLUME_WINDOW).mean()
        dollar_volume_ranks = avg_dollar_volumes.rank(axis=1, ascending=False, pct=True)
        have_adequate_dollar_volumes = dollar_volume_ranks <= (self.DOLLAR_VOLUME_TOP_N_PCT/100)

        # Step 2: apply momentum screen
        year_ago_closes = closes.shift(self.MOMENTUM_WINDOW)
        month_ago_closes = closes.shift(self.MOMENTUM_EXCLUDE_MOST_RECENT_WINDOW)
        returns = (month_ago_closes - year_ago_closes) / year_ago_closes.where(year_ago_closes != 0) # avoid DivisionByZero errors
        # Rank only among stocks with adequate dollar volume
        returns_ranks = returns.where(have_adequate_dollar_volumes).rank(axis=1, ascending=False, pct=True)
        have_momentum = returns_ranks <= (self.TOP_N_PCT / 100)

        # Step 3: Filter by smoothness of momentum
        are_positive_days = closes.pct_change() > 0
        positive_days_last_twelve_months = are_positive_days.astype(int).rolling(self.MOMENTUM_WINDOW).sum()
        positive_days_last_twelve_months_ranks = positive_days_last_twelve_months.where(have_momentum).rank(axis=1, ascending=False, pct=True)
        have_smooth_momentum = positive_days_last_twelve_months_ranks <= (self.SMOOTHEST_TOP_N_PCT/100)

        signals = have_smooth_momentum.astype(int)
        return signals

    def signals_to_target_weights(self, signals: pd.DataFrame, prices: pd.DataFrame):
        # Step 4: equal weights
        daily_signal_counts = signals.abs().sum(axis=1)
        weights = signals.div(daily_signal_counts, axis=0).fillna(0)

        # Step 5: Rebalance portfolio before quarter-end to capture window-dressing seasonality effect
        # Resample daily to REBALANCE_INTERVAL, taking the last day's signal
        # For pandas offset aliases, see https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
        weights = weights.resample(self.REBALANCE_INTERVAL).last()
        # Reindex back to daily and fill forward
        weights = weights.reindex(prices.loc["Close"].index, method="ffill")

        return weights

    def target_weights_to_positions(self, weights: pd.DataFrame, prices: pd.DataFrame):
        # Enter the position the day after the signal
        return weights.shift()

    def positions_to_gross_returns(self, positions: pd.DataFrame, prices: pd.DataFrame):
        closes = prices.loc["Close"]
        position_ends = positions.shift()

        # The return is the security's percent change over the period,
        # multiplied by the position.
        gross_returns = closes.pct_change() * position_ends

        return gross_returns

    def order_stubs_to_orders(self, orders: pd.DataFrame, prices: pd.DataFrame):
        orders["Exchange"] = "SMART"
        orders["OrderType"] = "MOC"
        orders["Tif"] = "DAY"
        return orders
