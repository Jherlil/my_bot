"""Technical analysis utilities using TA-Lib."""

import numpy as np
import pandas as pd
import talib
from utils import log

class TechnicalAnalyzer:
    """Compute technical indicators and detect patterns."""

    def __init__(self, ma_fast: int = 20, ma_slow: int = 50, volume_period: int = 20):
        self.ma_fast = ma_fast
        self.ma_slow = ma_slow
        self.volume_period = volume_period

    def calculate_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add moving average columns to *df*."""
        df['MA_fast'] = df['close'].rolling(self.ma_fast).mean()
        df['MA_slow'] = df['close'].rolling(self.ma_slow).mean()
        return df

    def detect_trend(self, df: pd.DataFrame) -> str:
        """Return ``up``, ``down`` or ``flat`` based on MAs."""
        last = df.iloc[-1]
        return "up" if last.MA_fast > last.MA_slow else "down" if last.MA_fast < last.MA_slow else "flat"

    def detect_breakout(self, df: pd.DataFrame):
        """Detect breakouts over support or under resistance."""
        support = df['low'].min()
        resistance = df['high'].max()
        last = df.iloc[-1]
        if last.close > resistance:
            return "breakout_up"
        elif last.close < support:
            return "breakout_down"
        return None

    def detect_candlestick_patterns(self, df: pd.DataFrame):
        """Return list of detected candlestick patterns."""
        o, h, l, c = df.open.values, df.high.values, df.low.values, df.close.values
        patterns = []
        for func_name in talib.get_function_groups()['Pattern Recognition']:
            func = getattr(talib, func_name)
            result = func(o, h, l, c)
            last_signal = result[-1]
            if last_signal != 0:
                patterns.append((func_name, last_signal))
        return patterns

    def support_resistance(self, df: pd.DataFrame, lookback: int = 50):
        """Return recent support and resistance levels."""
        return df['low'].rolling(lookback).min().iloc[-1], df['high'].rolling(lookback).max().iloc[-1]

    def fibonacci_levels(self, df: pd.DataFrame):
        """Return key Fibonacci retracement levels."""
        swing_high = df['high'].max()
        swing_low = df['low'].min()
        diff = swing_high - swing_low
        return {r: swing_low + diff * r for r in [0.236, 0.382, 0.5, 0.618, 0.786]}

    def draw_trendlines(self, df: pd.DataFrame):
        """Return basic trendline points."""
        top_idx = df['high'].idxmax()
        bottom_idx = df['low'].idxmin()
        return {"LTA/LTB": (bottom_idx, top_idx)}

    def validate_candle_pattern(self, candle, pattern_name: str) -> bool:
        """Heuristic validation for simple patterns."""
        body = abs(candle.close - candle.open)
        shadow_down = candle.open - candle.low if candle.open > candle.close else candle.close - candle.low
        shadow_up = candle.high - max(candle.open, candle.close)
        if pattern_name == "hammer":
            return shadow_down > body * 2 and shadow_up < body
        elif pattern_name == "shooting_star":
            return shadow_up > body * 2 and shadow_down < body
        elif pattern_name == "engulfing":
            return body > abs(candle.open - candle.close)
        elif pattern_name == "doji":
            return body < (candle.high - candle.low) * 0.1
        return True
