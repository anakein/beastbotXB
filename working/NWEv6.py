# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401

# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
pd.options.mode.chained_assignment = None  # default='warn'
from pandas import DataFrame

from freqtrade.strategy import IStrategy, merge_informative_pair
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter, stoploss_from_open

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.persistence import Trade
import math
from technical.indicators import TKE
import math
import logging
from functools import reduce

from datetime import datetime, timedelta, timezone
from timeit import default_timer as timer
from datetime import timedelta
import time
logger = logging.getLogger(__name__)

def SSLChannels_ATR(dataframe, length=7):
    """
    SSL Channels with ATR: https://www.tradingview.com/script/SKHqWzql-SSL-ATR-channel/
    Credit to @JimmyNixx for python
    """
    df = dataframe.copy()

    df['ATR'] = ta.ATR(df, timeperiod=14)
    df['smaHigh'] = df['high'].rolling(length).mean() + df['ATR']
    df['smaLow'] = df['low'].rolling(length).mean() - df['ATR']
    df['hlv'] = np.where(df['close'] > df['smaHigh'], 1, np.where(df['close'] < df['smaLow'], -1, np.NAN))
    df['hlv'] = df['hlv'].ffill()
    df['sslDown'] = np.where(df['hlv'] < 0, df['smaHigh'], df['smaLow'])
    df['sslUp'] = np.where(df['hlv'] < 0, df['smaLow'], df['smaHigh'])

    return df['sslDown'], df['sslUp']

def funcNadarayaWatsonEnvelope(dtloc, source = 'close', bandwidth = 8, window = 500, mult = 3):
    """
    // This work is licensed under a Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) https://creativecommons.org/licenses/by-nc-sa/4.0/
    // Nadaraya-Watson Envelope [LUX]
      https://www.tradingview.com/script/Iko0E2kL-Nadaraya-Watson-Envelope-LUX/
     :return: up and down   
     translated for freqtrade: viksal1982  viktors.s@gmail.com  


     df.shape[0]
    """ 
    dtNWE = dtloc.copy()
    dtNWE['nwe_up'] = np.nan
    dtNWE['nwe_down'] =  np.nan
    wn = np.zeros((window, window))
    for i in range(window):
        for j in range(window):
                wn[i,j] = math.exp(-(math.pow(i-j,2)/(bandwidth*bandwidth*2)))
    sumSCW = wn.sum(axis = 1)

    def calc_nwa(dfr, init=0):
        global calc_src_value
        if init == 1:
            calc_src_value = list()
            return
        calc_src_value.append(dfr[source])
        mae = 0.0
        y2_val = 0.0
        y2_val_up = np.nan
        y2_val_down = np.nan
        if len(calc_src_value) > window:
            calc_src_value.pop(0)
        if len(calc_src_value) >= window:
            src = np.array(calc_src_value)
            sumSC = src * wn
            sumSCS = sumSC.sum(axis = 1)
            y2 = sumSCS / sumSCW
            sum_e = np.absolute(src - y2)
            mae = sum_e.sum()/window*mult 
            y2_val = y2[-1]
            y2_val_up = y2_val + mae
            y2_val_down = y2_val - mae
        return y2_val_up,y2_val_down
    calc_nwa(None, init=1)
    dtNWE[['nwe_up','nwe_down']] = dtNWE.apply(calc_nwa, axis = 1, result_type='expand')
    return dtNWE[['nwe_up','nwe_down']]

class NWEv6(IStrategy):
   
    INTERFACE_VERSION = 2


    window_buy = IntParameter(60, 1000, default=300, space='buy', optimize=True)
    bandwidth_buy = IntParameter(2, 15, default=9, space='buy', optimize=True)
    mult_buy = DecimalParameter(0.5, 20.0, default=4, space='buy', optimize=True)
    marginselldw = DecimalParameter(1.0049, 1.0200, default=1.0038, space='buy', decimals=4, optimize=True, load=True) 
    # hard stoploss profit
    pHSL = DecimalParameter(-0.100, -0.040, default=-0.05, decimals=3, space='sell', load=True)
    # profit threshold 1, trigger point, SL_1 is used
    pPF_1 = DecimalParameter(0.008, 0.020, default=0.016, decimals=3, space='sell', load=True)
    pSL_1 = DecimalParameter(0.008, 0.020, default=0.011, decimals=3, space='sell', load=True)

    # profit threshold 2, SL_2 is used
    pPF_2 = DecimalParameter(0.040, 0.100, default=0.080, decimals=3, space='sell', load=True)
    pSL_2 = DecimalParameter(0.020, 0.070, default=0.040, decimals=3, space='sell', load=True)
 
    # Optimal timeframe for the strategy.
    timeframe = '5m'

    custom_trade_info = {}
    custom_trendBTC_info = {}

    if not 'trend' in custom_trendBTC_info:
        custom_trendBTC_info['trend'] = {}

    # These values can be overridden in the "ask_strategy" section in the config.
    use_custom_stoploss = True
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 300


    minimal_roi = {
      "0": 0.10,
      "423": 0.03,
      "751": 0.01
    }

    stoploss = -0.99
    trailing_stop = True
    trailing_stop_positive = 0.005
    trailing_stop_positive_offset = 0.019
    trailing_only_offset_is_reached = True
    process_only_new_candles = False

    @property
    def protections(self):
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration": 120
            },
            {
                "method": "StoplossGuard",
                "lookback_period": 90,
                "trade_limit": 2,
                "stop_duration": 120,
                "only_per_pair": False
            },
            {
                "method": "StoplossGuard",
                "lookback_period": 90,
                "trade_limit": 1,
                "stop_duration": 120,
                "only_per_pair": True
            },
        ]
    # Optional order type mapping.
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Optional order time in force.
    order_time_in_force = {
        'entry': 'gtc',
        'exit': 'gtc'
    }
    
    plot_config = {
        # Main plot indicators (Moving averages, ...)
        'main_plot': {
            'nwe_up': {'color': 'red'},
            'nwe_down': {'color': "rgba(155,150,200,2.4)"},
            'ema_100': {'color': 'blue'},
            'entry_line': {'color': 'green'},
        },
        'subplots': {
            "TREND/PCT": {
              'btctrend': {'color': 'green'},
              'highpct': {'color': 'red'},
              'lowpct': {'color': 'blue'}
            }  
          
        }
    }

    ## Custom Trailing stoploss ( credit to Perkmeister for this custom stoploss to help the strategy ride a green candle )
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        # hard stoploss profit
        HSL = self.pHSL.value
        PF_1 = self.pPF_1.value
        SL_1 = self.pSL_1.value
        PF_2 = self.pPF_2.value
        SL_2 = self.pSL_2.value

        # For profits between PF_1 and PF_2 the stoploss (sl_profit) used is linearly interpolated
        # between the values of SL_1 and SL_2. For all profits above PL_2 the sl_profit value
        # rises linearly with current profit, for profits below PF_1 the hard stoploss profit is used.

        if (current_profit > PF_2):
            sl_profit = SL_2 + (current_profit - PF_2)
        elif (current_profit > PF_1):
            sl_profit = SL_1 + ((current_profit - PF_1) * (SL_2 - SL_1) / (PF_2 - PF_1))
        else:
            sl_profit = HSL

        # Only for hyperopt invalid return
        if (sl_profit >= current_profit):
            return -0.99

        return stoploss_from_open(sl_profit, current_profit)


    def info_tf_btc_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # -----------------------------------------------------------------------------------------
        if not 'trend' in self.custom_trendBTC_info:
            self.custom_trendBTC_info['trend'] = {}

        if self.dp:
            inf_tf = '3m'
            informative = self.dp.get_pair_dataframe(pair=f"BTC/USDT",
                                                     timeframe=inf_tf)

        ssldown, sslup = SSLChannels_ATR(informative, 25)
        informative['ssl-dir'] = np.where(sslup > ssldown, 1, -1)

        self.custom_trendBTC_info['trend'] = informative['ssl-dir']

        return True

    def informative_pairs(self):

        return [(f"BTC/USDT", '3m')]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        tik = time.perf_counter()
        #only 1 time for populates, dont work backtesting
        btc_info_pair = "BTC/USDT"
        if metadata['pair'] in btc_info_pair:
            btc_info_tfx = self.info_tf_btc_indicators(dataframe, metadata)
            #logger.info(f"BTC SSL 1 time for pairs, Trend  {self.custom_trendBTC_info['trend']}")
            
        dataframe['btctrend'] = self.custom_trendBTC_info['trend'] 
        
        dataframe['ema_100'] = ta.EMA(dataframe, timeperiod=100)
        dataframe[['nwe_up','nwe_down']] = funcNadarayaWatsonEnvelope(dataframe, source = 'close', bandwidth = self.bandwidth_buy.value, window = self.window_buy.value, mult = self.mult_buy.value)
 
        dataframe["highpct"] = dataframe['nwe_down'] * dataframe['nwe_up'].pct_change(periods=24)
        dataframe["lowpct"] =  dataframe['nwe_down'] * dataframe['nwe_down'].pct_change(periods=6)

        dataframe['entry_line'] = dataframe['nwe_down']/self.marginselldw.value
        dataframe['Newentry_line'] = dataframe['nwe_down'] * dataframe["lowpct"]     

        
        tok = time.perf_counter()
        logger.debug(f"[{metadata['pair']}] informative_1h_indicators took: {tok - tik:0.4f} seconds.")
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        conditions = []

        con2 = (
                (dataframe['btctrend'] == 1) &
                (qtpylib.crossed_below(dataframe['highpct'], dataframe['lowpct'])) &
                (dataframe['ema_100'] > dataframe['close'] ) &
                (dataframe['nwe_up'] <= (abs(dataframe['highpct'])*100)) &
                (dataframe['volume'] > 0)
        )


        con3 = ( 
                (dataframe['close'].shift() < dataframe['entry_line'].shift()) & 
                (dataframe['close'].shift(2) < dataframe['entry_line'].shift(2)) &
                (dataframe['close'] > dataframe['entry_line']) &
                (dataframe['ema_100'] > dataframe['close'] ) &
#                (dataframe['ema_100'] > dataframe['nwe_up'] ) &
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
        )

  
        conditions.append(con2)
        conditions.append(con3)
        
        dataframe.loc[con2, 'buy_tag'] = " con2 "
        dataframe.loc[con3, 'buy_tag'] = " con3 "

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'buy'
            ] = 1

        return dataframe
    

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        conditions = []

        con1 = ( 
                 (dataframe['close'].shift() > dataframe['nwe_up'].shift()) &
                 (dataframe['close'] < dataframe['nwe_up'])
            )

        conditions.append(con1)
        #conditions.append(con2)       

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'sell'
            ] = 1

        return dataframe
