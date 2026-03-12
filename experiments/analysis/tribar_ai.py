import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta

try:
    import joblib
    from lightgbm import LGBMClassifier  # optional upgrade
    from sklearn.ensemble import (
        GradientBoostingClassifier,
        RandomForestClassifier,
        VotingClassifier,
    )
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import (
        TimeSeriesSplit,
        cross_val_score,
        train_test_split,
    )
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "ML dependencies are not installed. Install with the `ml` extra, e.g. `uv sync --extra ml`."
    ) from exc

# Model storage directory
MODEL_DIR = Path(__file__).parent.parent.parent / "trained"
MODEL_DIR.mkdir(exist_ok=True)


# -----------------------
# Utilities
# -----------------------
def add_tribar_column(df: pd.DataFrame) -> pd.Series:
    """
    Original tribar definition: price breaks previous 2 bars' highs/lows
    Used by profitability models.
    """
    prev_high_1, prev_high_2 = df["custom_high"].shift(1), df["custom_high"].shift(2)
    prev_low_1, prev_low_2 = df["custom_low"].shift(1), df["custom_low"].shift(2)

    tribar_condition = np.where(
        df["is_green"],
        (df["close"] > prev_high_1) & (df["close"] > prev_high_2),
        (df["close"] < prev_low_1) & (df["close"] < prev_low_2),
    )

    return pd.Series(np.where(tribar_condition, 1, 0), index=df.index)


def add_body_dominant_column(df: pd.DataFrame, multiplier: float = 2.0) -> pd.Series:
    """
    Body-dominant bar filter: body > multiplier * max(upper_wick, lower_wick)
    Used by direction prediction models.

    This identifies strong momentum bars where the body dominates the wicks.
    Achieves 66.25% accuracy on GC SHORT direction prediction with multiplier=2.0.

    Args:
        df: DataFrame with OHLC data
        multiplier: How much body must exceed max wick (default 2.0 = body > 2x wicks)
                   Lower values (1.0, 1.5) significantly reduce accuracy (53%, 52%)

    Returns:
        Series with 1 for body-dominant bars, 0 otherwise

    Performance (GC SHORT):
        multiplier=2.0: 66.25% accuracy, F1=0.70, 79% recall ✅ BEST
        multiplier=1.5: 51.92% accuracy, F1=0.40, 35% recall ❌
        multiplier=1.0: 53.38% accuracy, F1=0.35, 27% recall ❌
    """
    # Calculate custom high/low (max/min of open/close)
    custom_high = df[["open", "close"]].max(axis=1)
    custom_low = df[["open", "close"]].min(axis=1)

    # Calculate wicks
    upper_wick = df["high"] - custom_high
    lower_wick = custom_low - df["low"]

    # Calculate body
    body = abs(df["close"] - df["open"])

    # Body-dominant: body > multiplier * max(upper_wick, lower_wick)
    max_wick = pd.concat([upper_wick, lower_wick], axis=1).max(axis=1)
    is_body_dominant = (body > multiplier * max_wick).astype(int)

    return pd.Series(is_body_dominant, index=df.index)


def add_tribar_hl_column(df: pd.DataFrame) -> pd.Series:
    """
    Tribar definition using actual high/low: close breaks previous 2 bars' actual highs/lows

    Similar to add_tribar_column but uses actual high/low instead of custom_high/custom_low.

    Green bar tribar: close > prev_high_1 AND close > prev_high_2
    Red bar tribar: close < prev_low_1 AND close < prev_low_2
    """
    prev_high_1, prev_high_2 = df["high"].shift(1), df["high"].shift(2)
    prev_low_1, prev_low_2 = df["low"].shift(1), df["low"].shift(2)

    tribar_condition = np.where(
        df["is_green"],
        (df["close"] > prev_high_1) & (df["close"] > prev_high_2),
        (df["close"] < prev_low_1) & (df["close"] < prev_low_2),
    )

    return pd.Series(np.where(tribar_condition, 1, 0), index=df.index)


def compute_risk_reward(df: pd.DataFrame) -> pd.Series:
    stoploss = np.where(df["is_green"], df["low"], df["high"])
    entry = df["open"]
    exit_ = df["close"]
    rr = np.where(
        df["is_green"],
        (exit_ - entry) / (entry - stoploss),
        (entry - exit_) / (stoploss - entry),
    )
    return pd.Series(rr, index=df.index)


def compute_sl_hit_target(df: pd.DataFrame) -> pd.Series:
    """
    Predict if 1.5x wick SL will be hit during the bar

    Stop Loss calculation:
    - Green bars: SL = open - (wicks_diff_sma14 * 1.5)
    - Red bars: SL = open + (wicks_diff_sma14 * 1.5)

    Returns 1 if SL was hit, 0 if SL was NOT hit (safe)
    """
    # Handle NaN in wicks_diff_sma14
    wicks_diff = df["wicks_diff_sma14"].fillna(0)

    sl_price = np.where(
        df["is_green"],
        df["open"] - (wicks_diff * 1.5),
        df["open"] + (wicks_diff * 1.5),
    )

    sl_hit = np.where(
        df["is_green"],
        df["low"] <= sl_price,  # Green bar: check if low touched SL
        df["high"] >= sl_price,  # Red bar: check if high touched SL
    )

    # Return 1 if SL hit, 0 if safe
    return pd.Series(sl_hit.astype(int), index=df.index)


def compute_trade_profitability(df: pd.DataFrame) -> pd.Series:
    """
    Predict if open-to-close trade will be profitable

    Strategy:
    - LONG (green bars): Entry=open, SL=open-1.5*wicks_diff_sma14, Exit=close
    - SHORT (red bars): Entry=open, SL=open+1.5*wicks_diff_sma14, Exit=close

    Returns:
    - 1 = PROFITABLE (trade wins without hitting SL)
    - 0 = UNPROFITABLE (trade loses OR SL hit)
    """
    # Handle NaN in wicks_diff_sma14
    wicks_diff = df["wicks_diff_sma14"].fillna(0)

    # Calculate stop loss prices
    sl_price_long = df["open"] - (wicks_diff * 1.5)
    sl_price_short = df["open"] + (wicks_diff * 1.5)

    # Check if SL was hit
    sl_hit_long = df["low"] <= sl_price_long
    sl_hit_short = df["high"] >= sl_price_short

    # For GREEN bars (we take LONG trades)
    # Profitable if: close > open AND SL was NOT hit
    long_profitable = (df["close"] > df["open"]) & (~sl_hit_long)

    # For RED bars (we take SHORT trades)
    # Profitable if: close < open AND SL was NOT hit
    short_profitable = (df["close"] < df["open"]) & (~sl_hit_short)

    # Combine based on bar color (trade direction)
    profitable = np.where(df["is_green"], long_profitable, short_profitable)

    # Return 1 if profitable, 0 if unprofitable or stopped out
    return pd.Series(profitable.astype(int), index=df.index)


# -----------------------
# Data loading / preprocessing (STANDALONE)
# -----------------------
def load_main_tf(csv_path="~/Desktop/es.csv") -> pd.DataFrame:
    """Load raw OHLC data from CSV"""
    csv_path = str(Path(csv_path).expanduser())
    df = pd.read_csv(csv_path)

    # Handle different column name formats
    if "time" in df.columns:
        df["datetime"] = pd.to_datetime(df["time"], utc=True)
    elif "timestamp" in df.columns:
        df["datetime"] = pd.to_datetime(df["timestamp"], utc=True)
    elif "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    elif "date" in df.columns:
        df["datetime"] = pd.to_datetime(df["date"], utc=True)

    # Keep only OHLC columns
    df = df[["datetime", "open", "high", "low", "close"]].set_index("datetime")
    return df


def add_basic_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all indicators without depending on strategy"""
    df = df.copy()
    df = df.sort_index()

    # Basic candle features
    df["custom_high"] = df[["open", "close"]].max(axis=1)
    df["custom_low"] = df[["open", "close"]].min(axis=1)
    df["is_green"] = (df["close"] > df["open"]).astype(int)

    # Moving averages
    df["200sma"] = pandas_ta.sma(df["close"], length=200)
    df["9ema"] = pandas_ta.ema(df["close"], length=9)

    # ADX calculation
    df["adx"] = pandas_ta.adx(df["high"], df["low"], df["close"], length=14, mamode="rma")["ADX_14"]

    # ATR
    df["atr"] = pandas_ta.atr(df["high"], df["low"], df["close"], length=14, mamode="rma")

    # Wicks analysis
    df["wicks_diff"] = np.where(df["is_green"], df["open"] - df["low"], df["high"] - df["open"])
    df["wicks_diff_sma14"] = df["wicks_diff"].rolling(14).mean()

    # Extension analysis
    df["ext"] = np.where(df["is_green"], df["high"] - df["open"], df["open"] - df["low"])
    df["ext_sma14"] = df["ext"].rolling(14).mean()

    # Drop NaN rows from indicators
    df = df.dropna()

    return df


def preprocess_data(csv_path="~/Desktop/es.csv") -> pd.DataFrame:
    """Standalone preprocessing - only needs OHLC data"""
    # Load raw OHLC
    main_tf = load_main_tf(csv_path)

    # Compute all indicators standalone
    df = add_basic_indicators(main_tf)

    print(df.tail())

    # Add tribar column
    df["is_tribar"] = add_tribar_column(df)

    # Compute risk/reward
    # df["risk_reward"] = compute_risk_reward(df)

    # Compute SL hit
    # df["sl_hit"] = compute_sl_hit_target(df)

    # Compute trade profitability (NEW - for profitability model)
    # df["trade_profitable"] = compute_trade_profitability(df)

    # Reset index for easier manipulation
    df = df.reset_index()
    df["date"] = df["datetime"].dt.date

    return df


# -----------------------
# Training two models
# -----------------------
def _make_model(model_type="rf"):
    """Create optimized model with better hyperparameters"""
    if model_type == "lgbm":
        return LGBMClassifier(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=7,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )
    elif model_type == "gb":
        return GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            min_samples_split=20,
            min_samples_leaf=10,
            subsample=0.8,
            random_state=42,
        )
    elif model_type == "ensemble":
        # Ensemble of multiple models
        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=7,
            min_samples_leaf=8,
            min_samples_split=15,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )
        lgbm = LGBMClassifier(
            n_estimators=400, learning_rate=0.03, max_depth=6, random_state=42, class_weight="balanced", n_jobs=-1, verbose=-1
        )
        gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
        return VotingClassifier(estimators=[("rf", rf), ("lgbm", lgbm), ("gb", gb)], voting="soft", n_jobs=-1)
    else:  # rf
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=7,
            min_samples_leaf=8,
            min_samples_split=15,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )


def add_advanced_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add comprehensive feature engineering - consolidates all creative, ultra-creative, and comprehensive features.

    This function combines:
    - Bar filter patterns (tribar, tribar_hl, body_dominant with various multipliers)
    - Basic momentum and volatility indicators
    - RSI-based features
    - Consecutive bar patterns
    - EMA alignment features
    - ATR volatility features
    - Momentum strength features
    - ADX variations
    - Body size features
    - Stochastic oscillator features
    - MACD features
    - Bollinger Bands features
    - Kalman filter features:
      * Price smoothing and trend extraction
      * ADX Kalman filter for trend strength smoothing
      * ATR Kalman filter for volatility smoothing
      * Close-to-close ATR with Kalman filter
      * Gap vs continuous volatility detection
      * Combined market regime detection
    - EMA ribbon features
    - Price action patterns
    - Candlestick patterns (hammer, shooting star, engulfing)
    - Momentum scores and trend strength
    """
    # ===== BAR FILTER PATTERNS =====
    # Tribar patterns (if not already present)
    if "is_tribar" not in df.columns:
        df["is_tribar"] = add_tribar_column(df)

    df["is_tribar_hl"] = add_tribar_hl_column(df)

    # Body dominant patterns with various multipliers (commonly used in filter analysis)
    df["body_dominant_0_5x"] = add_body_dominant_column(df, multiplier=0.5)
    df["body_dominant_1_0x"] = add_body_dominant_column(df, multiplier=1.0)
    df["body_dominant_1_5x"] = add_body_dominant_column(df, multiplier=1.5)
    df["body_dominant_2_0x"] = add_body_dominant_column(df, multiplier=2.0)
    df["body_dominant_2_5x"] = add_body_dominant_column(df, multiplier=2.5)

    # ===== BASIC FEATURES =====
    # Momentum indicators
    df["roc_5"] = df["close"].pct_change(5) * 100  # Rate of change
    df["roc_10"] = df["close"].pct_change(10) * 100
    df["momentum_14"] = df["close"] - df["close"].shift(14)

    # Volatility indicators
    df["atr_pct"] = df["atr"] / df["close"]  # ATR as % of price
    df["bar_range_pct"] = (df["high"] - df["low"]) / df["close"]  # Bar range as % of price
    df["volatility_20_cv"] = df["close"].rolling(20).std() / df["close"].rolling(20).mean()  # Coefficient of variation

    # Price position relative to moving averages (deviation = distance from MA as %)
    df["price_vs_9ema_dev"] = (df["close"] - df["9ema"]) / df["close"]
    df["price_vs_200sma_dev"] = (df["close"] - df["200sma"]) / df["close"]
    df["9ema_to_200sma"] = (df["9ema"] - df["200sma"]) / df["200sma"]  # This one is correct ✅

    # Boolean position indicators
    df["is_close_above_200sma"] = (df["close"] > df["200sma"]).astype(int)
    df["is_close_above_9ema"] = (df["close"] > df["9ema"]).astype(int)

    # Market regime indicators
    df["sma_slope_5"] = df["200sma"].diff(5) / df["200sma"].shift(5)
    df["ema_slope_5"] = df["9ema"].diff(5) / df["9ema"].shift(5)

    # Candle characteristics
    df["body_size_pct"] = abs(df["close"] - df["open"]) / df["close"]  # Body as % of price
    df["upper_shadow"] = np.where(df["is_green"], df["high"] - df["close"], df["high"] - df["open"])
    df["lower_shadow"] = np.where(df["is_green"], df["open"] - df["low"], df["close"] - df["low"])
    df["upper_shadow_ratio"] = df["upper_shadow"] / (df["high"] - df["low"]).replace(0, np.nan)
    df["lower_shadow_ratio"] = df["lower_shadow"] / (df["high"] - df["low"]).replace(0, np.nan)

    # Sequential pattern features
    df["prev_tribar"] = df["is_tribar"].shift(1)
    df["prev_green"] = df["is_green"].shift(1)
    df["consecutive_green"] = df["is_green"].rolling(3).sum()

    # ADX momentum
    df["adx_change"] = df["adx"].diff()
    df["adx_sma"] = df["adx"].rolling(5).mean()

    # ===== RSI FEATURES =====
    df["rsi_14"] = pandas_ta.rsi(df["close"], length=14)
    df["rsi_7"] = pandas_ta.rsi(df["close"], length=7)
    df["rsi_21"] = pandas_ta.rsi(df["close"], length=21)

    # RSI-based features (both long and short)
    df["is_rsi_oversold_recovery"] = ((df["rsi_14"].shift(1) < 30) & (df["rsi_14"] > 30)).astype(int)
    df["is_rsi_overbought_recovery"] = ((df["rsi_14"].shift(1) > 70) & (df["rsi_14"] < 70)).astype(int)
    df["rsi_bullish"] = (df["rsi_14"] > 50).astype(int)
    df["rsi_bearish"] = (df["rsi_14"] < 50).astype(int)
    df["rsi_very_bullish"] = (df["rsi_14"] > 60).astype(int)
    df["rsi_very_bearish"] = (df["rsi_14"] < 40).astype(int)

    # ===== CONSECUTIVE BAR PATTERNS =====
    df["is_red"] = (df["is_green"] == 0).astype(int)
    df["consecutive_green_2"] = ((df["is_green"] == 1) & (df["is_green"].shift(1) == 1)).astype(int)
    df["consecutive_green_3"] = ((df["is_green"] == 1) & (df["is_green"].shift(1) == 1) & (df["is_green"].shift(2) == 1)).astype(
        int
    )
    df["consecutive_red_2"] = ((df["is_red"] == 1) & (df["is_red"].shift(1) == 1)).astype(int)
    df["consecutive_red_3"] = ((df["is_red"] == 1) & (df["is_red"].shift(1) == 1) & (df["is_red"].shift(2) == 1)).astype(int)

    # ===== ADDITIONAL EMAs =====
    df["20ema"] = pandas_ta.ema(df["close"], length=20)
    df["50ema"] = pandas_ta.ema(df["close"], length=50)

    # EMA position features
    df["above_20ema"] = (df["close"] > df["20ema"]).astype(int)
    df["above_50ema"] = (df["close"] > df["50ema"]).astype(int)
    df["below_20ema"] = (df["close"] < df["20ema"]).astype(int)
    df["below_50ema"] = (df["close"] < df["50ema"]).astype(int)

    # EMA alignment (trend strength)
    df["all_emas_aligned"] = ((df["close"] > df["9ema"]) & (df["9ema"] > df["20ema"]) & (df["20ema"] > df["50ema"])).astype(int)
    df["all_emas_dealigned"] = ((df["close"] < df["9ema"]) & (df["9ema"] < df["20ema"]) & (df["20ema"] < df["50ema"])).astype(int)

    # ===== ATR VOLATILITY FEATURES =====
    df["is_high_volatility"] = (df["atr"] > df["atr"].rolling(20).mean() * 1.2).astype(int)
    df["is_low_volatility"] = (df["atr"] < df["atr"].rolling(20).mean() * 0.8).astype(int)
    df["expanding_atr"] = (df["atr"] > df["atr"].shift(1)).astype(int)

    # ===== MOMENTUM STRENGTH =====
    df["strong_upward_momentum"] = (
        (df["roc_5"] > df["roc_5"].quantile(0.75)) & (df["roc_10"] > df["roc_10"].quantile(0.75))
    ).astype(int)
    df["strong_downward_momentum"] = (
        (df["roc_5"] < df["roc_5"].quantile(0.25)) & (df["roc_10"] < df["roc_10"].quantile(0.25))
    ).astype(int)

    # ===== ADX VARIATIONS =====
    df["adx_rising"] = (df["adx"] > df["adx"].shift(1)).astype(int)
    df["adx_accelerating"] = ((df["adx"] > df["adx"].shift(1)) & (df["adx"].shift(1) > df["adx"].shift(2))).astype(int)

    # ===== BODY SIZE FEATURES =====
    df["is_large_body"] = (df["body_size_pct"] > df["body_size_pct"].rolling(20).mean() * 1.5).astype(int)
    df["is_very_large_body"] = (df["body_size_pct"] > df["body_size_pct"].rolling(20).mean() * 2.0).astype(int)

    # ===== STOCHASTIC OSCILLATOR =====
    stoch = pandas_ta.stoch(df["high"], df["low"], df["close"], k=14, d=3)
    df["stoch_k"] = stoch["STOCHk_14_3_3"]
    df["stoch_d"] = stoch["STOCHd_14_3_3"]
    df["is_stoch_oversold"] = (df["stoch_k"] < 20).astype(int)
    df["is_stoch_overbought"] = (df["stoch_k"] > 80).astype(int)
    df["stoch_bullish_cross"] = ((df["stoch_k"] > df["stoch_d"]) & (df["stoch_k"].shift(1) <= df["stoch_d"].shift(1))).astype(int)
    df["stoch_bearish_cross"] = ((df["stoch_k"] < df["stoch_d"]) & (df["stoch_k"].shift(1) >= df["stoch_d"].shift(1))).astype(int)

    # ===== MACD =====
    macd = pandas_ta.macd(df["close"], fast=12, slow=26, signal=9)
    df["macd"] = macd["MACD_12_26_9"]
    df["macd_signal"] = macd["MACDs_12_26_9"]
    df["macd_hist"] = macd["MACDh_12_26_9"]
    df["macd_bullish"] = (df["macd"] > df["macd_signal"]).astype(int)
    df["macd_bearish"] = (df["macd"] < df["macd_signal"]).astype(int)
    df["macd_cross_up"] = ((df["macd"] > df["macd_signal"]) & (df["macd"].shift(1) <= df["macd_signal"].shift(1))).astype(int)
    df["macd_cross_down"] = ((df["macd"] < df["macd_signal"]) & (df["macd"].shift(1) >= df["macd_signal"].shift(1))).astype(int)
    df["macd_histogram_increasing"] = (df["macd_hist"] > df["macd_hist"].shift(1)).astype(int)
    df["macd_histogram_decreasing"] = (df["macd_hist"] < df["macd_hist"].shift(1)).astype(int)

    # ===== BOLLINGER BANDS =====
    df["bb_middle"] = df["close"].rolling(window=20).mean()
    df["bb_std"] = df["close"].rolling(window=20).std()
    df["bb_upper"] = df["bb_middle"] + (df["bb_std"] * 2)
    df["bb_lower"] = df["bb_middle"] - (df["bb_std"] * 2)
    df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-10)
    df["is_bb_squeeze"] = (df["bb_std"] < df["bb_std"].rolling(20).mean() * 0.8).astype(int)
    df["above_bb_middle"] = (df["close"] > df["bb_middle"]).astype(int)
    df["below_bb_middle"] = (df["close"] < df["bb_middle"]).astype(int)

    # ===== KALMAN FILTER =====
    # Apply Kalman filter for price smoothing and prediction
    def apply_kalman_filter(prices, process_variance=0.01, observation_variance=0.1):
        """
        Apply Kalman filter to price series for smoothing and prediction

        Args:
            prices: Price series to filter
            process_variance: Process noise variance (Q)
            observation_variance: Measurement noise variance (R)

        Returns:
            filtered_prices: Kalman filtered price series
            prediction_errors: Prediction errors (innovations)
        """
        n = len(prices)
        filtered_prices = np.zeros(n)
        prediction_errors = np.zeros(n)

        # Initialize state and covariance
        x = prices.iloc[0] if hasattr(prices, "iloc") else prices[0]
        P = 1.0

        for i in range(n):
            # Prediction step
            x_pred = x
            P_pred = P + process_variance

            # Update step
            z = prices.iloc[i] if hasattr(prices, "iloc") else prices[i]
            if not np.isnan(z):
                # Kalman gain
                K = P_pred / (P_pred + observation_variance)

                # State update
                innovation = z - x_pred
                x = x_pred + K * innovation
                P = (1 - K) * P_pred

                filtered_prices[i] = x
                prediction_errors[i] = innovation
            else:
                filtered_prices[i] = x_pred
                prediction_errors[i] = 0

        return filtered_prices, prediction_errors

    # Apply Kalman filter to close prices
    kf_close, kf_innovations = apply_kalman_filter(df["close"], process_variance=0.01, observation_variance=0.1)
    kf_trend, _ = apply_kalman_filter(df["close"], process_variance=0.001, observation_variance=0.5)

    kf_close_series = pd.Series(kf_close, index=df.index)
    kf_innovation_series = pd.Series(kf_innovations, index=df.index)
    kf_trend_series = pd.Series(kf_trend, index=df.index)

    kf_close_momentum = kf_close_series - kf_close_series.shift(1)
    kf_trend_momentum = kf_trend_series - kf_trend_series.shift(5)
    kf_innovation_abs = kf_innovation_series.abs()
    kf_innovation_roll_mean = kf_innovation_abs.rolling(20).mean()
    kf_innovation_roll_std = kf_innovation_series.rolling(20).std()
    kf_slope_5 = (kf_close_series - kf_close_series.shift(5)) / 5
    kf_slope_10 = (kf_close_series - kf_close_series.shift(10)) / 10

    kalman_price_features = pd.DataFrame(
        {
            "kf_close": kf_close_series,
            "kf_innovation": kf_innovation_series,
            "kf_trend": kf_trend_series,
            "kf_price_deviation": (df["close"] - kf_close_series) / df["close"],
            "kf_trend_deviation": (df["close"] - kf_trend_series) / df["close"],
            "kf_above_smooth": (df["close"] > kf_close_series).astype(int),
            "kf_below_smooth": (df["close"] < kf_close_series).astype(int),
            "kf_above_trend": (df["close"] > kf_trend_series).astype(int),
            "kf_below_trend": (df["close"] < kf_trend_series).astype(int),
            "kf_close_momentum": kf_close_momentum,
            "kf_trend_momentum": kf_trend_momentum,
            "kf_momentum_increasing": (kf_close_momentum > kf_close_momentum.shift(1)).astype(int),
            "kf_momentum_decreasing": (kf_close_momentum < kf_close_momentum.shift(1)).astype(int),
            "kf_innovation_abs": kf_innovation_abs,
            "is_kf_innovation_large": (kf_innovation_abs > kf_innovation_roll_mean * 1.5).astype(int),
            "is_kf_positive_surprise": (kf_innovation_series > kf_innovation_roll_std).astype(int),
            "is_kf_negative_surprise": (kf_innovation_series < -kf_innovation_roll_std).astype(int),
            "kf_slope_5": kf_slope_5,
            "kf_slope_10": kf_slope_10,
            "kf_slope_increasing": (kf_slope_5 > kf_slope_5.shift(1)).astype(int),
            "kf_slope_decreasing": (kf_slope_5 < kf_slope_5.shift(1)).astype(int),
            "kf_vs_9ema": (kf_close_series - df["9ema"]) / df["9ema"],
            "kf_vs_200sma": (kf_close_series - df["200sma"]) / df["200sma"],
            "kf_ema_aligned": ((kf_close_series > df["9ema"]) & (df["9ema"] > df["200sma"])).astype(int),
            "kf_ema_divergence": (kf_close_series - df["9ema"]).abs() / df["9ema"],
            "is_close_above_kf_ma": (df["close"] > kf_close_series).astype(int),
        },
        index=df.index,
    )

    # ===== KALMAN FILTER FOR ADX =====
    # Apply Kalman filter to ADX for smoother trend strength signals
    kf_adx, kf_adx_innovations = apply_kalman_filter(df["adx"], process_variance=0.005, observation_variance=0.2)
    kf_adx_series = pd.Series(kf_adx, index=df.index)
    kf_adx_innovation_series = pd.Series(kf_adx_innovations, index=df.index)
    kf_adx_momentum = kf_adx_series - kf_adx_series.shift(1)
    kf_adx_momentum_5 = kf_adx_series - kf_adx_series.shift(5)
    kf_adx_innovation_abs = kf_adx_innovation_series.abs()

    kf_adx_features = pd.DataFrame(
        {
            "kf_adx": kf_adx_series,
            "kf_adx_innovation": kf_adx_innovation_series,
            "kf_adx_deviation": (df["adx"] - kf_adx_series) / (kf_adx_series + 1e-10),
            "kf_adx_above_25": (kf_adx_series > 25).astype(int),
            "kf_adx_above_40": (kf_adx_series > 40).astype(int),
            "kf_adx_below_20": (kf_adx_series < 20).astype(int),
            "kf_adx_momentum": kf_adx_momentum,
            "kf_adx_momentum_5": kf_adx_momentum_5,
            "kf_adx_increasing": (kf_adx_series > kf_adx_series.shift(1)).astype(int),
            "kf_adx_decreasing": (kf_adx_series < kf_adx_series.shift(1)).astype(int),
            "kf_adx_accelerating": ((kf_adx_momentum > 0) & (kf_adx_momentum > kf_adx_momentum.shift(1))).astype(int),
            "kf_adx_decelerating": ((kf_adx_momentum < 0) & (kf_adx_momentum < kf_adx_momentum.shift(1))).astype(int),
            "kf_adx_innovation_abs": kf_adx_innovation_abs,
            "is_kf_adx_surprise": (kf_adx_innovation_abs > kf_adx_innovation_abs.rolling(20).mean() * 1.5).astype(int),
            "kf_adx_trend_emerging": ((kf_adx_series > 20) & (kf_adx_series.shift(5) < 20)).astype(int),
            "kf_adx_trend_fading": ((kf_adx_series < 25) & (kf_adx_series.shift(5) > 30)).astype(int),
        },
        index=df.index,
    )

    # ===== KALMAN FILTER FOR ATR =====
    # Apply Kalman filter to ATR for smoother volatility signals
    kf_atr, kf_atr_innovations = apply_kalman_filter(df["atr"], process_variance=0.01, observation_variance=0.15)
    kf_atr_series = pd.Series(kf_atr, index=df.index)
    kf_atr_innovation_series = pd.Series(kf_atr_innovations, index=df.index)
    kf_atr_momentum = kf_atr_series - kf_atr_series.shift(1)
    kf_atr_momentum_5 = kf_atr_series - kf_atr_series.shift(5)
    kf_atr_innovation_abs = kf_atr_innovation_series.abs()
    kf_atr_mean = kf_atr_series.rolling(20).mean()
    kf_atr_std = kf_atr_series.rolling(20).std()
    is_kf_atr_high_volatility = (kf_atr_series > kf_atr_mean * 1.2).astype(int)
    is_kf_atr_low_volatility = (kf_atr_series < kf_atr_mean * 0.8).astype(int)
    is_kf_atr_very_high_volatility = (kf_atr_series > kf_atr_mean + 2 * kf_atr_std).astype(int)
    is_kf_atr_squeeze = (kf_atr_series < kf_atr_series.rolling(50).min() * 1.1).astype(int)
    kf_atr_expanding = (kf_atr_series > kf_atr_series.shift(1)).astype(int)
    kf_atr_contracting = (kf_atr_series < kf_atr_series.shift(1)).astype(int)

    kf_atr_pct = kf_atr_series / df["close"]

    kf_atr_features = pd.DataFrame(
        {
            "kf_atr": kf_atr_series,
            "kf_atr_innovation": kf_atr_innovation_series,
            "kf_atr_deviation": (df["atr"] - kf_atr_series) / (kf_atr_series + 1e-10),
            "kf_atr_pct": kf_atr_pct,
            "kf_atr_momentum": kf_atr_momentum,
            "kf_atr_momentum_5": kf_atr_momentum_5,
            "kf_atr_expanding": kf_atr_expanding,
            "kf_atr_contracting": kf_atr_contracting,
            "is_kf_atr_high_volatility": is_kf_atr_high_volatility,
            "is_kf_atr_low_volatility": is_kf_atr_low_volatility,
            "is_kf_atr_very_high_volatility": is_kf_atr_very_high_volatility,
            "is_kf_atr_squeeze": is_kf_atr_squeeze,
            "kf_atr_innovation_abs": kf_atr_innovation_abs,
            "is_kf_atr_volatility_spike": (kf_atr_innovation_series > kf_atr_innovation_series.rolling(20).std() * 2).astype(int),
            "is_kf_atr_volatility_drop": (kf_atr_innovation_series < -kf_atr_innovation_series.rolling(20).std() * 2).astype(int),
            "kf_trending_volatile": ((kf_adx_series > 25) & (is_kf_atr_high_volatility == 1)).astype(int),
            "kf_trending_quiet": ((kf_adx_series > 25) & (is_kf_atr_low_volatility == 1)).astype(int),
            "kf_ranging_volatile": ((kf_adx_series < 20) & (is_kf_atr_high_volatility == 1)).astype(int),
            "kf_ranging_quiet": ((kf_adx_series < 20) & (is_kf_atr_low_volatility == 1)).astype(int),
            "kf_trend_volatility_ratio": kf_adx_series / (kf_atr_pct * 100 + 1e-10),
            "is_kf_strong_trend_low_vol": ((kf_adx_series > 30) & (is_kf_atr_low_volatility == 1)).astype(int),
            "is_kf_breakout_potential": ((kf_adx_features["kf_adx_increasing"] == 1) & (kf_atr_expanding == 1)).astype(int),
            "is_kf_consolidation": ((kf_adx_features["kf_adx_decreasing"] == 1) & (kf_atr_contracting == 1)).astype(int),
        },
        index=df.index,
    )

    # Combine all Kalman-derived columns at once to avoid DataFrame fragmentation warnings
    df = pd.concat([df, kalman_price_features, kf_adx_features, kf_atr_features], axis=1)

    # ===== CLOSE-TO-CLOSE ATR =====
    # Calculate ATR based on close-to-close price movements
    # This captures volatility from closing price changes specifically

    # Calculate close-to-close range (absolute close-to-close changes)
    close_to_close_range = abs(df["close"] - df["close"].shift(1))

    # Calculate ATR using exponential moving average (traditional ATR uses 14-period)
    # Using EMA for smoother calculation similar to traditional ATR
    atr_c2c_period = 14
    atr_c2c_multiplier = 2.0 / (atr_c2c_period + 1)

    # Initialize with simple moving average for first period
    atr_c2c = close_to_close_range.rolling(window=atr_c2c_period, min_periods=1).mean()

    # Apply EMA formula for rest of the series
    for i in range(atr_c2c_period, len(df)):
        if not pd.isna(close_to_close_range.iloc[i]):
            atr_c2c.iloc[i] = close_to_close_range.iloc[i] * atr_c2c_multiplier + atr_c2c.iloc[i - 1] * (1 - atr_c2c_multiplier)

    # ===== KALMAN FILTER FOR CLOSE-TO-CLOSE ATR =====
    # Apply Kalman filter to close-to-close ATR for smoother volatility signals
    # Initialize kf_atr_c2c with NaN for proper handling
    kf_atr_c2c_series = pd.Series(np.nan, index=df.index)
    kf_atr_c2c_innovation_series = pd.Series(np.nan, index=df.index)

    # Only apply Kalman filter where atr_c2c is not NaN
    valid_idx = atr_c2c.notna()
    if valid_idx.sum() > 0:
        # Get valid values
        valid_atr_c2c = atr_c2c.loc[valid_idx]
        # Apply Kalman filter to valid values only
        kf_values, kf_innovations = apply_kalman_filter(valid_atr_c2c, process_variance=0.008, observation_variance=0.12)
        # Assign back to the valid indices
        kf_atr_c2c_series.loc[valid_idx] = kf_values
        kf_atr_c2c_innovation_series.loc[valid_idx] = kf_innovations

    # Volatility regime detection using Kalman close-to-close ATR
    kf_atr_c2c_mean = kf_atr_c2c_series.rolling(20).mean()
    kf_atr_c2c_std = kf_atr_c2c_series.rolling(20).std()

    # Create all ATR C2C features at once using pd.concat to avoid fragmentation
    atr_c2c_features = pd.DataFrame(
        {
            "atr_c2c": atr_c2c,
            "atr_c2c_pct": atr_c2c / df["close"],
            "atr_c2c_high": (atr_c2c > atr_c2c.rolling(20).mean() * 1.2).astype(int),
            "atr_c2c_low": (atr_c2c < atr_c2c.rolling(20).mean() * 0.8).astype(int),
            "atr_c2c_expanding": (atr_c2c > atr_c2c.shift(1)).astype(int),
            "atr_c2c_contracting": (atr_c2c < atr_c2c.shift(1)).astype(int),
            "atr_c2c_vs_atr": atr_c2c / (df["atr"] + 1e-10),
            "atr_c2c_dominance": (atr_c2c > df["atr"] * 0.7).astype(int),
            "kf_atr_c2c": kf_atr_c2c_series,
            "kf_atr_c2c_innovation": kf_atr_c2c_innovation_series,
            "kf_atr_c2c_deviation": (atr_c2c - kf_atr_c2c_series) / (kf_atr_c2c_series + 1e-10),
            "kf_atr_c2c_pct": kf_atr_c2c_series / df["close"],
            "kf_atr_c2c_momentum": kf_atr_c2c_series - kf_atr_c2c_series.shift(1),
            "kf_atr_c2c_momentum_5": kf_atr_c2c_series - kf_atr_c2c_series.shift(5),
            "kf_atr_c2c_expanding": (kf_atr_c2c_series > kf_atr_c2c_series.shift(1)).astype(int),
            "kf_atr_c2c_contracting": (kf_atr_c2c_series < kf_atr_c2c_series.shift(1)).astype(int),
            "is_kf_atr_c2c_high_volatility": (kf_atr_c2c_series > kf_atr_c2c_mean * 1.2).astype(int),
            "is_kf_atr_c2c_low_volatility": (kf_atr_c2c_series < kf_atr_c2c_mean * 0.8).astype(int),
            "is_kf_atr_c2c_very_high": (kf_atr_c2c_series > kf_atr_c2c_mean + 2 * kf_atr_c2c_std).astype(int),
            "is_kf_atr_c2c_squeeze": (kf_atr_c2c_series < kf_atr_c2c_series.rolling(50).min() * 1.1).astype(int),
            "kf_atr_c2c_innovation_abs": abs(kf_atr_c2c_innovation_series),
            "is_kf_atr_c2c_spike": (kf_atr_c2c_innovation_series > kf_atr_c2c_innovation_series.rolling(20).std() * 2).astype(
                int
            ),
            "is_kf_atr_c2c_drop": (kf_atr_c2c_innovation_series < -kf_atr_c2c_innovation_series.rolling(20).std() * 2).astype(
                int
            ),
            "kf_atr_vs_c2c": df["kf_atr"] / (kf_atr_c2c_series + 1e-10),
            "kf_c2c_dominance": (kf_atr_c2c_series > df["kf_atr"] * 0.7).astype(int),
            "is_kf_gap_volatility": ((df["kf_atr"] / (kf_atr_c2c_series + 1e-10)) > 1.5).astype(int),
            "is_kf_continuous_volatility": ((df["kf_atr"] / (kf_atr_c2c_series + 1e-10)) < 1.2).astype(int),
        },
        index=df.index,
    )

    # Combined market regime features (depend on features just created)
    atr_c2c_features["kf_trending_c2c_volatile"] = (
        (df["kf_adx"] > 25) & (atr_c2c_features["is_kf_atr_c2c_high_volatility"] == 1)
    ).astype(int)
    atr_c2c_features["kf_trending_c2c_quiet"] = (
        (df["kf_adx"] > 25) & (atr_c2c_features["is_kf_atr_c2c_low_volatility"] == 1)
    ).astype(int)
    atr_c2c_features["is_kf_gap_trading_opportunity"] = (
        (atr_c2c_features["is_kf_gap_volatility"] == 1) & (df["kf_adx"] < 25)
    ).astype(int)
    atr_c2c_features["is_kf_smooth_trend"] = (
        (atr_c2c_features["is_kf_continuous_volatility"] == 1) & (df["kf_adx"] > 30)
    ).astype(int)
    atr_c2c_features["kf_volatility_divergence"] = (
        ((df["kf_atr_expanding"] == 1) & (atr_c2c_features["kf_atr_c2c_contracting"] == 1))
        | ((df["kf_atr_contracting"] == 1) & (atr_c2c_features["kf_atr_c2c_expanding"] == 1))
    ).astype(int)

    # Add all ATR C2C features at once
    df = pd.concat([df, atr_c2c_features], axis=1)

    # ===== EMA RIBBON =====
    ema_ribbon_features = pd.DataFrame(
        {
            "ema_ribbon_aligned": (
                (df["9ema"] > df["20ema"]) & (df["20ema"] > df["50ema"]) & (df["50ema"] > df["200sma"])
            ).astype(int),
            "ema_ribbon_dealigned": (
                (df["9ema"] < df["20ema"]) & (df["20ema"] < df["50ema"]) & (df["50ema"] < df["200sma"])
            ).astype(int),
        },
        index=df.index,
    )

    # Add EMA ribbon features
    df = pd.concat([df, ema_ribbon_features], axis=1)

    # ===== CALCULATE ALL NEW COLUMNS IN TEMPORARY VARIABLES =====
    # Price action patterns
    higher_high = (df["high"] > df["high"].shift(1)).astype(int)
    higher_low = (df["low"] > df["low"].shift(1)).astype(int)
    lower_high = (df["high"] < df["high"].shift(1)).astype(int)
    lower_low = (df["low"] < df["low"].shift(1)).astype(int)
    bullish_bar_sequence = (higher_high & higher_low).astype(int)
    bearish_bar_sequence = (lower_high & lower_low).astype(int)

    # Candlestick patterns - calculate wicks
    lower_wick = df[["open", "close"]].min(axis=1) - df["low"]
    upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
    body_size_abs = abs(df["close"] - df["open"])

    is_hammer = ((lower_wick > 2 * body_size_abs) & (upper_wick < 0.3 * body_size_abs)).astype(int)
    is_shooting_star = ((upper_wick > 2 * body_size_abs) & (lower_wick < 0.3 * body_size_abs)).astype(int)

    bullish_engulfing = (
        (df["is_green"] == 1)
        & (df["is_green"].shift(1) == 0)
        & (df["open"] < df["close"].shift(1))
        & (df["close"] > df["open"].shift(1))
    ).astype(int)

    bearish_engulfing = (
        (df["is_red"] == 1)
        & (df["is_red"].shift(1) == 0)
        & (df["open"] > df["close"].shift(1))
        & (df["close"] < df["open"].shift(1))
    ).astype(int)

    # Momentum scores
    momentum_score = (df["rsi_14"].rank(pct=True) + df["roc_5"].rank(pct=True) + df["roc_10"].rank(pct=True)) / 3
    is_strong_momentum_score = (momentum_score > 0.75).astype(int)
    is_weak_momentum_score = (momentum_score < 0.25).astype(int)

    # Trend strength
    trend_strength = (df["adx"] / 100) * 0.4 + abs(df["close"] / df["200sma"] - 1) * 10 * 0.6
    is_very_strong_trend = (trend_strength > trend_strength.quantile(0.8)).astype(int)

    # Large bar patterns
    is_very_large_green = ((df["is_green"] == 1) & (body_size_abs > body_size_abs.rolling(20).mean() * 2.0)).astype(int)
    is_very_large_red = ((df["is_red"] == 1) & (body_size_abs > body_size_abs.rolling(20).mean() * 2.0)).astype(int)

    # Bar type filters
    custom_high = df[["open", "close"]].max(axis=1)
    custom_low = df[["open", "close"]].min(axis=1)
    upper_wick_calc = df["high"] - custom_high
    lower_wick_calc = custom_low - df["low"]
    body_size = abs(df["close"] - df["open"])
    max_wick = pd.concat([upper_wick_calc, lower_wick_calc], axis=1).max(axis=1)
    bar_body_to_wick_ratio = body_size / (max_wick + 1e-10)

    # Tribar filters
    bar_tribar_custom = add_tribar_column(df)
    bar_tribar_hl = add_tribar_hl_column(df)

    # EMA position indicators
    ema_above_9 = (df["close"] > df["9ema"]).astype(int)
    ema_below_9 = (df["close"] < df["9ema"]).astype(int)
    close_above_9ema = (df["close"] > df["9ema"]).astype(int)
    close_below_9ema = (df["close"] < df["9ema"]).astype(int)

    # Stochastic indicators
    stoch_bullish = (df["stoch_k"] > df["stoch_d"]).astype(int)
    stoch_bearish = (df["stoch_k"] < df["stoch_d"]).astype(int)

    # Bollinger Band indicators
    bb_upper_half = (df["bb_position"] > 0.5).astype(int)
    bb_lower_half = (df["bb_position"] < 0.5).astype(int)

    # Temporal stochastic
    stoch_oversold_prev = df["is_stoch_oversold"].shift(1).fillna(0).astype(int)
    stoch_overbought_prev = df["is_stoch_overbought"].shift(1).fillna(0).astype(int)

    # ===== ADD ALL NEW COLUMNS AT ONCE USING pd.concat() =====
    new_columns = pd.DataFrame(
        {
            "higher_high": higher_high,
            "higher_low": higher_low,
            "lower_high": lower_high,
            "lower_low": lower_low,
            "bullish_bar_sequence": bullish_bar_sequence,
            "bearish_bar_sequence": bearish_bar_sequence,
            "lower_wick": lower_wick,
            "upper_wick": upper_wick,
            "body_size_abs": body_size_abs,
            "is_hammer": is_hammer,
            "is_shooting_star": is_shooting_star,
            "bullish_engulfing": bullish_engulfing,
            "bearish_engulfing": bearish_engulfing,
            "momentum_score": momentum_score,
            "is_strong_momentum_score": is_strong_momentum_score,
            "is_weak_momentum_score": is_weak_momentum_score,
            "trend_strength": trend_strength,
            "is_very_strong_trend": is_very_strong_trend,
            "is_very_large_green": is_very_large_green,
            "is_very_large_red": is_very_large_red,
            "bar_body_to_wick_ratio": bar_body_to_wick_ratio,
            "bar_tribar_custom": bar_tribar_custom,
            "bar_tribar_hl": bar_tribar_hl,
            "ema_above_9": ema_above_9,
            "ema_below_9": ema_below_9,
            "close_above_9ema": close_above_9ema,
            "close_below_9ema": close_below_9ema,
            "stoch_bullish": stoch_bullish,
            "stoch_bearish": stoch_bearish,
            "bb_upper_half": bb_upper_half,
            "bb_lower_half": bb_lower_half,
            "stoch_oversold_prev": stoch_oversold_prev,
            "stoch_overbought_prev": stoch_overbought_prev,
        },
        index=df.index,
    )

    df = pd.concat([df, new_columns], axis=1)

    return df


def train_three_models(
    df: pd.DataFrame,
    features=None,
    rr_model_type="rf",
    color_model_type="rf",
    sl_model_type="rf",
    test_size=0.2,
    use_advanced_features=True,
):
    """
    Train three separate classifiers:
      - rr_model: predict next bar risk_reward >= 2 (target_rr)
      - color_model: predict next bar is_green == 1 (target_color)
      - sl_model: predict if 1.5x wick SL will be hit (target_sl_hit)
    Returns: (rr_model, color_model, sl_model, dict_of_eval_results)
    """

    # Add advanced features
    if use_advanced_features:
        df = add_advanced_indicators(df)

    if features is None:
        if use_advanced_features:
            features = [
                # OHLC
                "open",
                "high",
                "low",
                "close",
                # Moving Averages
                "9ema",
                "200sma",
                # Trend & Momentum
                "adx",
                "adx_change",
                "adx_sma",
                # Volatility
                "atr",
                "atr_pct",
                "bar_range_pct",
                "volatility_20_cv",
                # Wicks (CRITICAL for SL model!)
                "wicks_diff",
                "wicks_diff_sma14",
                # Extensions
                "ext",
                "ext_sma14",
                # Price Position
                "price_vs_9ema_dev",
                "price_vs_200sma_dev",
                "9ema_to_200sma",
                # MA Slopes
                "sma_slope_5",
                "ema_slope_5",
                # Candle Characteristics
                "body_size_pct",
                "upper_shadow",
                "lower_shadow",
                "upper_shadow_ratio",
                "lower_shadow_ratio",
                # Pattern Recognition
                "is_tribar",
                "prev_tribar",
                "prev_green",
                "consecutive_green",
                # Momentum Indicators
                "roc_5",
                "roc_10",
                "momentum_14",
            ]
        else:
            features = ["open", "high", "low", "close", "9ema", "200sma", "adx", "atr", "is_tribar"]

    # Clean up
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # Targets (shift -1 so current row maps to next-bar outcome)
    df["target_rr"] = df["risk_reward"].shift(-1) >= 2
    df["target_color"] = df["is_green"].shift(-1) == 1
    df["target_sl_hit"] = df["sl_hit"].shift(-1)

    # Drop last row because it has no next-bar target (NaN from shift)
    df = df.iloc[:-1].copy()

    # Convert to int after dropping NaN rows
    df["target_rr"] = df["target_rr"].astype(int)
    df["target_color"] = df["target_color"].astype(int)
    df["target_sl_hit"] = df["target_sl_hit"].astype(int)

    X = df[features]
    y_rr = df["target_rr"]
    y_color = df["target_color"]
    y_sl = df["target_sl_hit"]

    # Time-series split (no shuffle)
    X_train_rr, X_test_rr, y_train_rr, y_test_rr = train_test_split(X, y_rr, test_size=test_size, shuffle=False, random_state=42)
    X_train_col, X_test_col, y_train_col, y_test_col = train_test_split(
        X, y_color, test_size=test_size, shuffle=False, random_state=42
    )
    X_train_sl, X_test_sl, y_train_sl, y_test_sl = train_test_split(X, y_sl, test_size=test_size, shuffle=False, random_state=42)

    rr_model = _make_model(rr_model_type)
    color_model = _make_model(color_model_type)
    sl_model = _make_model(sl_model_type)

    # Cross-validation with TimeSeriesSplit
    print("\nPerforming time-series cross-validation...")
    tscv = TimeSeriesSplit(n_splits=5)

    rr_cv_scores = cross_val_score(rr_model, X_train_rr, y_train_rr, cv=tscv, scoring="f1", n_jobs=-1)
    color_cv_scores = cross_val_score(color_model, X_train_col, y_train_col, cv=tscv, scoring="f1", n_jobs=-1)
    sl_cv_scores = cross_val_score(sl_model, X_train_sl, y_train_sl, cv=tscv, scoring="f1", n_jobs=-1)

    print(f"RR Model CV F1 Scores: {rr_cv_scores}")
    print(f"RR Model Mean CV F1: {rr_cv_scores.mean():.4f} (+/- {rr_cv_scores.std() * 2:.4f})")
    print(f"Color Model CV F1 Scores: {color_cv_scores}")
    print(f"Color Model Mean CV F1: {color_cv_scores.mean():.4f} (+/- {color_cv_scores.std() * 2:.4f})")
    print(f"SL Hit Model CV F1 Scores: {sl_cv_scores}")
    print(f"SL Hit Model Mean CV F1: {sl_cv_scores.mean():.4f} (+/- {sl_cv_scores.std() * 2:.4f})")

    # Train
    print("\nTraining final models on full training set...")
    rr_model.fit(X_train_rr, y_train_rr)
    color_model.fit(X_train_col, y_train_col)
    sl_model.fit(X_train_sl, y_train_sl)

    # Predict
    rr_pred = rr_model.predict(X_test_rr)
    rr_proba = rr_model.predict_proba(X_test_rr)[:, 1]

    col_pred = color_model.predict(X_test_col)
    col_proba = color_model.predict_proba(X_test_col)[:, 1]

    sl_pred = sl_model.predict(X_test_sl)
    sl_proba = sl_model.predict_proba(X_test_sl)[:, 1]

    # Evaluate helper
    def eval_model(y_true, y_pred, y_proba):
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else np.nan,
            "classification_report": classification_report(y_true, y_pred, zero_division=0),
            "confusion_matrix": confusion_matrix(y_true, y_pred),
        }

    rr_eval = eval_model(y_test_rr, rr_pred, rr_proba)
    col_eval = eval_model(y_test_col, col_pred, col_proba)
    sl_eval = eval_model(y_test_sl, sl_pred, sl_proba)

    # Feature importances (handle ensemble models)
    def get_feature_importance(model, features):
        if hasattr(model, "feature_importances_"):
            return pd.DataFrame({"feature": features, "importance": model.feature_importances_}).sort_values(
                "importance", ascending=False
            )
        elif hasattr(model, "estimators_"):
            # For VotingClassifier, average feature importances across estimators
            importances = []
            for estimator in model.estimators_:
                if hasattr(estimator, "feature_importances_"):
                    importances.append(estimator.feature_importances_)
            if importances:
                avg_importance = np.mean(importances, axis=0)
                return pd.DataFrame({"feature": features, "importance": avg_importance}).sort_values(
                    "importance", ascending=False
                )
        return pd.DataFrame({"feature": features, "importance": [0] * len(features)})

    rr_importance = get_feature_importance(rr_model, features)
    col_importance = get_feature_importance(color_model, features)
    sl_importance = get_feature_importance(sl_model, features)

    # Next-bar sample prediction (last available features row)
    next_row = X.iloc[[-1]]
    next_rr_pred = rr_model.predict(next_row)[0]
    next_rr_proba = rr_model.predict_proba(next_row)[0][1]
    next_col_pred = color_model.predict(next_row)[0]
    next_col_proba = color_model.predict_proba(next_row)[0][1]
    next_sl_pred = sl_model.predict(next_row)[0]
    next_sl_proba = sl_model.predict_proba(next_row)[0][1]

    evals = {
        "rr": {
            "model": rr_model,
            "X_test": X_test_rr,
            "y_test": y_test_rr,
            "y_pred": rr_pred,
            "y_proba": rr_proba,
            "metrics": rr_eval,
            "feature_importance": rr_importance,
            "next_pred": {"pred": int(next_rr_pred), "proba": float(next_rr_proba)},
        },
        "color": {
            "model": color_model,
            "X_test": X_test_col,
            "y_test": y_test_col,
            "y_pred": col_pred,
            "y_proba": col_proba,
            "metrics": col_eval,
            "feature_importance": col_importance,
            "next_pred": {"pred": int(next_col_pred), "proba": float(next_col_proba)},
        },
        "sl_hit": {
            "model": sl_model,
            "X_test": X_test_sl,
            "y_test": y_test_sl,
            "y_pred": sl_pred,
            "y_proba": sl_proba,
            "metrics": sl_eval,
            "feature_importance": sl_importance,
            "next_pred": {"pred": int(next_sl_pred), "proba": float(next_sl_proba)},
        },
    }

    # Print concise summaries
    print("\n" + "=" * 60)
    print("RR MODEL EVAL (NEXT BAR RR >= 2)")
    print("=" * 60)
    print(f"Train size: {len(X_train_rr)}, Test size: {len(X_test_rr)}")
    print(
        f"Accuracy: {rr_eval['accuracy']:.4f}, Precision: {rr_eval['precision']:.4f}, Recall: {rr_eval['recall']:.4f}, F1: {rr_eval['f1']:.4f}"
    )
    print("Confusion matrix:\n", rr_eval["confusion_matrix"])
    print("\nTop features:\n", rr_importance.head())

    print("\n" + "=" * 60)
    print("COLOR MODEL EVAL (NEXT BAR GREEN)")
    print("=" * 60)
    print(f"Train size: {len(X_train_col)}, Test size: {len(X_test_col)}")
    print(
        f"Accuracy: {col_eval['accuracy']:.4f}, Precision: {col_eval['precision']:.4f}, Recall: {col_eval['recall']:.4f}, F1: {col_eval['f1']:.4f}"
    )
    print("Confusion matrix:\n", col_eval["confusion_matrix"])
    print("\nTop features:\n", col_importance.head())

    print("\n" + "=" * 60)
    print("SL HIT MODEL EVAL (1.5x WICK SL HIT)")
    print("=" * 60)
    print(f"Train size: {len(X_train_sl)}, Test size: {len(X_test_sl)}")
    print(
        f"Accuracy: {sl_eval['accuracy']:.4f}, Precision: {sl_eval['precision']:.4f}, Recall: {sl_eval['recall']:.4f}, F1: {sl_eval['f1']:.4f}"
    )
    print("Confusion matrix:\n", sl_eval["confusion_matrix"])
    print("\nTop features:\n", sl_importance.head())

    print("\n" + "=" * 60)
    print("NEXT BAR COMBINED VIEW")
    print("=" * 60)
    print(f"Next-bar probability RR>=2: {evals['rr']['next_pred']['proba']:.4f} (pred={evals['rr']['next_pred']['pred']})")
    print(f"Next-bar probability GREEN: {evals['color']['next_pred']['proba']:.4f} (pred={evals['color']['next_pred']['pred']})")
    print(
        f"Next-bar probability SL HIT: {evals['sl_hit']['next_pred']['proba']:.4f} (pred={evals['sl_hit']['next_pred']['pred']})"
    )

    return rr_model, color_model, sl_model, evals


def train_profitability_model(
    df: pd.DataFrame,
    features=None,
    model_type="rf",
    test_size=0.2,
    use_advanced_features=True,
):
    """
    Train TWO models together: color prediction + profitability prediction

    Strategy modeled:
    - LONG (green bars): Entry=open, SL=open-1.5*wicks, Exit=close
    - SHORT (red bars): Entry=open, SL=open+1.5*wicks, Exit=close

    Returns: (profitability_model, color_model, dict_of_eval_results)

    The combined models predict:
    1. What direction will next bar be? (color_model)
    2. Will that trade be profitable? (profitability_model)
    """

    # Add advanced features
    if use_advanced_features:
        df = add_advanced_indicators(df)

    if features is None:
        if use_advanced_features:
            features = [
                # OHLC
                "open",
                "high",
                "low",
                "close",
                # Moving Averages
                "9ema",
                "200sma",
                # Trend & Momentum
                "adx",
                "adx_change",
                "adx_sma",
                # Volatility
                "atr",
                "atr_pct",
                "bar_range_pct",
                "volatility_20_cv",
                # Wicks (CRITICAL for SL model!)
                "wicks_diff",
                "wicks_diff_sma14",
                # Extensions
                "ext",
                "ext_sma14",
                # Price Position
                "price_vs_9ema_dev",
                "price_vs_200sma_dev",
                "9ema_to_200sma",
                # MA Slopes
                "sma_slope_5",
                "ema_slope_5",
                # Candle Characteristics
                "body_size_pct",
                "upper_shadow",
                "lower_shadow",
                "upper_shadow_ratio",
                "lower_shadow_ratio",
                # Pattern Recognition
                "is_tribar",
                "prev_tribar",
                "prev_green",
                "consecutive_green",
                # Momentum Indicators
                "roc_5",
                "roc_10",
                "momentum_14",
            ]
        else:
            features = ["open", "high", "low", "close", "9ema", "200sma", "adx", "atr", "is_tribar"]

    # Clean up
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # Create TWO targets (shift -1 so current row maps to next-bar outcome)
    df["target_profitable"] = df["trade_profitable"].shift(-1)
    df["target_color"] = df["is_green"].shift(-1)

    # Drop last row because it has no next-bar target
    df = df.iloc[:-1].copy()

    # Convert to int after dropping NaN rows
    df["target_profitable"] = df["target_profitable"].astype(int)
    df["target_color"] = df["target_color"].astype(int)

    X = df[features]
    y_profitable = df["target_profitable"]
    y_color = df["target_color"]

    # Time-series split (no shuffle) - use same split for both models
    X_train, X_test, y_train_prof, y_test_prof = train_test_split(
        X, y_profitable, test_size=test_size, shuffle=False, random_state=42
    )
    _, _, y_train_color, y_test_color = train_test_split(X, y_color, test_size=test_size, shuffle=False, random_state=42)

    # Train TWO models
    profitability_model = _make_model(model_type)
    color_model = _make_model(model_type)

    # Cross-validation with TimeSeriesSplit
    print("\nPerforming time-series cross-validation...")
    tscv = TimeSeriesSplit(n_splits=5)

    prof_cv_scores = cross_val_score(profitability_model, X_train, y_train_prof, cv=tscv, scoring="f1", n_jobs=-1)
    color_cv_scores = cross_val_score(color_model, X_train, y_train_color, cv=tscv, scoring="f1", n_jobs=-1)

    print(f"Profitability Model CV F1 Scores: {prof_cv_scores}")
    print(f"Profitability Mean CV F1: {prof_cv_scores.mean():.4f} (+/- {prof_cv_scores.std() * 2:.4f})")
    print(f"Color Model CV F1 Scores: {color_cv_scores}")
    print(f"Color Mean CV F1: {color_cv_scores.mean():.4f} (+/- {color_cv_scores.std() * 2:.4f})")

    # Train both models
    print("\nTraining final models on full training set...")
    profitability_model.fit(X_train, y_train_prof)
    color_model.fit(X_train, y_train_color)

    # Predict with both models
    y_pred_prof = profitability_model.predict(X_test)
    y_proba_prof = profitability_model.predict_proba(X_test)[:, 1]

    y_pred_color = color_model.predict(X_test)
    y_proba_color = color_model.predict_proba(X_test)[:, 1]

    # Evaluate helper
    def eval_model(y_true, y_pred, y_proba):
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else np.nan,
            "classification_report": classification_report(y_true, y_pred, zero_division=0),
            "confusion_matrix": confusion_matrix(y_true, y_pred),
        }

    metrics_prof = eval_model(y_test_prof, y_pred_prof, y_proba_prof)
    metrics_color = eval_model(y_test_color, y_pred_color, y_proba_color)

    # Feature importances
    def get_feature_importance(model, features):
        if hasattr(model, "feature_importances_"):
            return pd.DataFrame({"feature": features, "importance": model.feature_importances_}).sort_values(
                "importance", ascending=False
            )
        elif hasattr(model, "estimators_"):
            # For VotingClassifier, average feature importances across estimators
            importances = []
            for estimator in model.estimators_:
                if hasattr(estimator, "feature_importances_"):
                    importances.append(estimator.feature_importances_)
            if importances:
                avg_importance = np.mean(importances, axis=0)
                return pd.DataFrame({"feature": features, "importance": avg_importance}).sort_values(
                    "importance", ascending=False
                )
        return pd.DataFrame({"feature": features, "importance": [0] * len(features)})

    feature_importance_prof = get_feature_importance(profitability_model, features)
    feature_importance_color = get_feature_importance(color_model, features)

    # Walk-forward validation on test set for both models
    walk_forward_results = []
    test_indices = X_test.index.tolist()

    for i in range(len(X_test)):
        test_idx = test_indices[i]
        test_row = X_test.loc[[test_idx]]

        # Profitability predictions
        true_prof = y_test_prof.loc[test_idx]
        pred_prof = profitability_model.predict(test_row)[0]
        proba_prof = profitability_model.predict_proba(test_row)[0][1]

        # Color predictions
        true_color = y_test_color.loc[test_idx]
        pred_color = color_model.predict(test_row)[0]
        proba_color = color_model.predict_proba(test_row)[0][1]

        walk_forward_results.append(
            {
                "index": test_idx,
                "true_profitable": true_prof,
                "pred_profitable": pred_prof,
                "proba_profitable": proba_prof,
                "true_color": true_color,
                "pred_color": pred_color,
                "proba_color": proba_color,
                "correct_prof": true_prof == pred_prof,
                "correct_color": true_color == pred_color,
            }
        )

    walk_forward_df = pd.DataFrame(walk_forward_results)
    walk_forward_accuracy_prof = walk_forward_df["correct_prof"].mean()
    walk_forward_accuracy_color = walk_forward_df["correct_color"].mean()

    evals = {
        "profitability_model": profitability_model,
        "color_model": color_model,
        "X_test": X_test,
        "y_test_prof": y_test_prof,
        "y_test_color": y_test_color,
        "y_pred_prof": y_pred_prof,
        "y_proba_prof": y_proba_prof,
        "y_pred_color": y_pred_color,
        "y_proba_color": y_proba_color,
        "metrics_prof": metrics_prof,
        "metrics_color": metrics_color,
        "feature_importance_prof": feature_importance_prof,
        "feature_importance_color": feature_importance_color,
        "walk_forward_results": walk_forward_df,
        "walk_forward_accuracy_prof": walk_forward_accuracy_prof,
        "walk_forward_accuracy_color": walk_forward_accuracy_color,
    }

    # Print concise summary
    print("\n" + "=" * 60)
    print("PROFITABILITY MODEL EVAL (TRADE WILL BE PROFITABLE)")
    print("=" * 60)
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(
        f"Accuracy: {metrics_prof['accuracy']:.4f}, Precision: {metrics_prof['precision']:.4f}, Recall: {metrics_prof['recall']:.4f}, F1: {metrics_prof['f1']:.4f}"
    )
    print("Confusion matrix:\n", metrics_prof["confusion_matrix"])
    print("\nTop features:\n", feature_importance_prof.head(10))

    print("\n" + "=" * 60)
    print("COLOR MODEL EVAL (NEXT BAR WILL BE GREEN)")
    print("=" * 60)
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(
        f"Accuracy: {metrics_color['accuracy']:.4f}, Precision: {metrics_color['precision']:.4f}, Recall: {metrics_color['recall']:.4f}, F1: {metrics_color['f1']:.4f}"
    )
    print("Confusion matrix:\n", metrics_color["confusion_matrix"])
    print("\nTop features:\n", feature_importance_color.head(10))

    print("\n" + "=" * 60)
    print("WALK-FORWARD VALIDATION (TRUE OUT-OF-SAMPLE)")
    print("=" * 60)
    print(f"Profitability walk-forward accuracy: {walk_forward_accuracy_prof:.4f}")
    print(f"Color walk-forward accuracy: {walk_forward_accuracy_color:.4f}")
    print(f"Number of predictions: {len(walk_forward_df)}")
    print(
        f"\nProfitability - Correct: {walk_forward_df['correct_prof'].sum()}, Incorrect: {(~walk_forward_df['correct_prof']).sum()}"
    )
    print(f"Color - Correct: {walk_forward_df['correct_color'].sum()}, Incorrect: {(~walk_forward_df['correct_color']).sum()}")
    print("\nFirst 5 walk-forward predictions:")
    print(walk_forward_df.head().to_string(index=False))
    print("\nLast 5 walk-forward predictions:")
    print(walk_forward_df.tail().to_string(index=False))

    # Class distribution
    print("\n" + "=" * 60)
    print("CLASS DISTRIBUTION")
    print("=" * 60)
    print("Profitability:")
    print(f"  Profitable trades: {y_profitable.sum()} ({y_profitable.mean():.2%})")
    print(f"  Unprofitable trades: {len(y_profitable) - y_profitable.sum()} ({1 - y_profitable.mean():.2%})")
    print("\nColor:")
    print(f"  Green bars: {y_color.sum()} ({y_color.mean():.2%})")
    print(f"  Red bars: {len(y_color) - y_color.sum()} ({1 - y_color.mean():.2%})")

    return profitability_model, color_model, evals


def predict_next_bar_profitability(
    profitability_model, color_model, current_bar_features: pd.DataFrame, features_list: list = None
):
    """
    PRODUCTION USE: Predict DIRECTION and PROFITABILITY of the NEXT bar (not yet opened)

    Usage:
    ------
    1. Wait for current bar to close completely
    2. Calculate all features from the closed bar (OHLC + indicators)
    3. Pass those features to this function
    4. Get combined prediction for the NEXT bar that will open

    Parameters:
    -----------
    profitability_model : trained profitability model from train_profitability_model()
    color_model : trained color model from train_profitability_model()
    current_bar_features : pd.DataFrame with one row containing features from the completed bar
    features_list : list of feature names (must match training features)

    Returns:
    --------
    dict with:
        - 'profitable': 1 (profitable) or 0 (unprofitable)
        - 'profitable_probability': float between 0-1
        - 'direction': 'LONG' or 'SHORT'
        - 'direction_probability': float between 0-1 (probability of being green/long)
        - 'confidence': str ('HIGH', 'MODERATE', 'LOW', 'VERY_LOW')
        - 'recommendation': str trading recommendation
        - 'trade_signal': str ('BUY', 'SELL', 'NO_TRADE')

    Example:
    --------
    # Current bar just closed at 2024-01-10 16:00
    latest_bar = get_latest_closed_bar()  # Your data source
    features_df = compute_indicators_standalone(latest_bar)

    # Predict next bar direction and profitability (2024-01-10 16:30, not yet opened)
    prediction = predict_next_bar_profitability(prof_model, color_model, features_df)

    if prediction['trade_signal'] == 'BUY' and prediction['confidence'] in ['HIGH', 'MODERATE']:
        execute_long_trade_at_next_open()
    elif prediction['trade_signal'] == 'SELL' and prediction['confidence'] in ['HIGH', 'MODERATE']:
        execute_short_trade_at_next_open()
    """
    if features_list is None:
        # Default to advanced features
        features_list = [
            "open",
            "high",
            "low",
            "close",
            "9ema",
            "200sma",
            "adx",
            "adx_change",
            "adx_sma",
            "atr",
            "atr_pct",
            "bar_range_pct",
            "volatility_20_cv",
            "wicks_diff",
            "wicks_diff_sma14",
            "ext",
            "ext_sma14",
            "price_vs_9ema_dev",
            "price_vs_200sma_dev",
            "9ema_to_200sma",
            "sma_slope_5",
            "ema_slope_5",
            "body_size_pct",
            "upper_shadow",
            "lower_shadow",
            "upper_shadow_ratio",
            "lower_shadow_ratio",
            "is_tribar",
            "prev_tribar",
            "prev_green",
            "consecutive_green",
            "roc_5",
            "roc_10",
            "momentum_14",
        ]

    # Ensure we have the right features
    X = current_bar_features[features_list]

    # Predict profitability
    pred_profitable = profitability_model.predict(X)[0]
    proba_profitable = profitability_model.predict_proba(X)[0][1]

    # Predict color/direction
    pred_color = color_model.predict(X)[0]  # 1=green/long, 0=red/short
    proba_color = color_model.predict_proba(X)[0][1]  # probability of green/long

    # Determine trade direction
    if pred_color == 1:
        direction = "LONG"
        trade_signal = "BUY" if pred_profitable == 1 else "NO_TRADE"
    else:
        direction = "SHORT"
        trade_signal = "SELL" if pred_profitable == 1 else "NO_TRADE"

    # Determine overall confidence based on BOTH models
    # Use the lower of the two probabilities to be conservative
    min_proba = min(proba_profitable, proba_color if pred_color == 1 else (1 - proba_color))

    if min_proba > 0.70:
        confidence = "HIGH"
    elif min_proba > 0.60:
        confidence = "MODERATE"
    elif min_proba > 0.55:
        confidence = "LOW"
    else:
        confidence = "VERY_LOW"

    # Generate recommendation
    if pred_profitable == 1:
        if confidence == "HIGH":
            recommendation = f"STRONG {direction} - High probability profitable trade"
        elif confidence == "MODERATE":
            recommendation = f"{direction} - Good probability profitable trade"
        elif confidence == "LOW":
            recommendation = f"WEAK {direction} - Low confidence, consider passing"
        else:
            recommendation = "NO SIGNAL - Very low confidence, skip trade"
    else:
        recommendation = "AVOID - Trade predicted to be unprofitable"

    return {
        "profitable": int(pred_profitable),
        "profitable_probability": float(proba_profitable),
        "direction": direction,
        "direction_probability": float(proba_color),
        "confidence": confidence,
        "recommendation": recommendation,
        "trade_signal": trade_signal,
    }


# -----------------------
# Entrypoint
# -----------------------
def main():
    print("Loading and preprocessing data...")
    df = preprocess_data()
    print(f"Total bars: {len(df)}. Columns: {df.columns.tolist()}")

    # Train models with advanced features and optimized settings
    # Model types: "rf", "lgbm", "gb", "ensemble"
    rr_model, color_model, sl_model, evals = train_three_models(
        df,
        features=None,  # None = use all advanced features automatically
        rr_model_type="ensemble",  # Use ensemble for best accuracy
        color_model_type="ensemble",  # Use ensemble for best accuracy
        sl_model_type="ensemble",  # Use ensemble for best accuracy
        test_size=0.2,
        use_advanced_features=True,  # Enable advanced feature engineering
    )

    # Print final recommendations
    print("\n" + "=" * 60)
    print("MODEL RECOMMENDATIONS")
    print("=" * 60)

    # Calculate combined confidence score
    rr_conf = evals["rr"]["next_pred"]["proba"]
    color_conf = evals["color"]["next_pred"]["proba"]
    sl_conf = evals["sl_hit"]["next_pred"]["proba"]

    print(f"Next bar RR>=2 probability: {rr_conf:.2%}")
    print(f"Next bar GREEN probability: {color_conf:.2%}")
    print(f"Next bar SL HIT probability: {sl_conf:.2%}")

    # Trading recommendation with SL risk assessment
    if rr_conf > 0.6 and color_conf > 0.6 and sl_conf < 0.3:
        print("\n✓ STRONG SIGNAL: Consider LONG position (High RR + Safe SL)")
    elif rr_conf > 0.6 and color_conf < 0.4 and sl_conf < 0.3:
        print("\n✓ STRONG SIGNAL: Consider SHORT position (High RR + Safe SL)")
    elif rr_conf > 0.6 and sl_conf > 0.7:
        print("\n⚠ RISKY: Good RR but high SL hit probability - Consider skip or widen SL")
    elif sl_conf > 0.7:
        print("\n✗ HIGH RISK: SL likely to be hit - Skip trade")
    else:
        print("\n✗ WEAK SIGNAL: No trade recommended")

    print("=" * 60)

    # models and evals returned for programmatic use
    return rr_model, color_model, sl_model, evals


def save_models(rr_model, color_model, sl_model, evals, model_type: str, csv_path: str = "~/Desktop/es.csv"):
    """Save trained models and their evaluation metrics"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = model_type  # Simple name without timestamp

    # Create model directory (overwrite if exists)
    model_path = MODEL_DIR / model_name
    model_path.mkdir(exist_ok=True)

    # Save models
    joblib.dump(rr_model, model_path / "rr_model.joblib")
    joblib.dump(color_model, model_path / "color_model.joblib")
    joblib.dump(sl_model, model_path / "sl_model.joblib")

    # Save evaluation metrics
    metrics = {
        "model_type": model_type,
        "timestamp": timestamp,
        "csv_path": csv_path,
        "rr_metrics": {
            "accuracy": float(evals["rr"]["metrics"]["accuracy"]),
            "precision": float(evals["rr"]["metrics"]["precision"]),
            "recall": float(evals["rr"]["metrics"]["recall"]),
            "f1": float(evals["rr"]["metrics"]["f1"]),
            "roc_auc": float(evals["rr"]["metrics"]["roc_auc"]) if not np.isnan(evals["rr"]["metrics"]["roc_auc"]) else None,
        },
        "color_metrics": {
            "accuracy": float(evals["color"]["metrics"]["accuracy"]),
            "precision": float(evals["color"]["metrics"]["precision"]),
            "recall": float(evals["color"]["metrics"]["recall"]),
            "f1": float(evals["color"]["metrics"]["f1"]),
            "roc_auc": (
                float(evals["color"]["metrics"]["roc_auc"]) if not np.isnan(evals["color"]["metrics"]["roc_auc"]) else None
            ),
        },
        "sl_hit_metrics": {
            "accuracy": float(evals["sl_hit"]["metrics"]["accuracy"]),
            "precision": float(evals["sl_hit"]["metrics"]["precision"]),
            "recall": float(evals["sl_hit"]["metrics"]["recall"]),
            "f1": float(evals["sl_hit"]["metrics"]["f1"]),
            "roc_auc": (
                float(evals["sl_hit"]["metrics"]["roc_auc"]) if not np.isnan(evals["sl_hit"]["metrics"]["roc_auc"]) else None
            ),
        },
        "next_predictions": {
            "rr_proba": float(evals["rr"]["next_pred"]["proba"]),
            "rr_pred": int(evals["rr"]["next_pred"]["pred"]),
            "color_proba": float(evals["color"]["next_pred"]["proba"]),
            "color_pred": int(evals["color"]["next_pred"]["pred"]),
            "sl_hit_proba": float(evals["sl_hit"]["next_pred"]["proba"]),
            "sl_hit_pred": int(evals["sl_hit"]["next_pred"]["pred"]),
        },
    }

    with open(model_path / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save feature importance
    evals["rr"]["feature_importance"].to_csv(model_path / "rr_feature_importance.csv", index=False)
    evals["color"]["feature_importance"].to_csv(model_path / "color_feature_importance.csv", index=False)
    evals["sl_hit"]["feature_importance"].to_csv(model_path / "sl_feature_importance.csv", index=False)

    print(f"\n✓ Models saved to: {model_path}")
    return model_path


def load_models(model_name: str):
    """Load saved models"""
    model_path = MODEL_DIR / model_name

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    rr_model = joblib.load(model_path / "rr_model.joblib")
    color_model = joblib.load(model_path / "color_model.joblib")
    sl_model = joblib.load(model_path / "sl_model.joblib")

    with open(model_path / "metrics.json", "r") as f:
        metrics = json.load(f)

    return rr_model, color_model, sl_model, metrics


def save_profitability_model(profitability_model, color_model, evals, model_type: str, csv_path: str = "~/Desktop/es.csv"):
    """Save trained profitability and color models with evaluation metrics"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"profitability_{model_type}"  # e.g., profitability_ensemble

    # Create model directory (overwrite if exists)
    model_path = MODEL_DIR / model_name
    model_path.mkdir(exist_ok=True)

    # Save BOTH models
    joblib.dump(profitability_model, model_path / "profitability_model.joblib")
    joblib.dump(color_model, model_path / "color_model.joblib")

    # Save evaluation metrics for BOTH models
    metrics = {
        "model_type": model_type,
        "timestamp": timestamp,
        "csv_path": csv_path,
        "model_purpose": "profitability_and_direction",
        "strategy": "LONG on green bars (open to close), SHORT on red bars (open to close), SL=1.5x wicks",
        "profitability_metrics": {
            "accuracy": float(evals["metrics_prof"]["accuracy"]),
            "precision": float(evals["metrics_prof"]["precision"]),
            "recall": float(evals["metrics_prof"]["recall"]),
            "f1": float(evals["metrics_prof"]["f1"]),
            "roc_auc": float(evals["metrics_prof"]["roc_auc"]) if not np.isnan(evals["metrics_prof"]["roc_auc"]) else None,
        },
        "color_metrics": {
            "accuracy": float(evals["metrics_color"]["accuracy"]),
            "precision": float(evals["metrics_color"]["precision"]),
            "recall": float(evals["metrics_color"]["recall"]),
            "f1": float(evals["metrics_color"]["f1"]),
            "roc_auc": float(evals["metrics_color"]["roc_auc"]) if not np.isnan(evals["metrics_color"]["roc_auc"]) else None,
        },
        "walk_forward_accuracy_prof": float(evals["walk_forward_accuracy_prof"]),
        "walk_forward_accuracy_color": float(evals["walk_forward_accuracy_color"]),
        "test_set_size": len(evals["walk_forward_results"]),
    }

    with open(model_path / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save feature importance for BOTH models
    evals["feature_importance_prof"].to_csv(model_path / "profitability_feature_importance.csv", index=False)
    evals["feature_importance_color"].to_csv(model_path / "color_feature_importance.csv", index=False)

    print(f"\n✓ Profitability model saved to: {model_path}")
    return model_path


def load_profitability_model(model_name: str):
    """Load saved profitability and color models"""
    # Handle both formats: "profitability_ensemble" or just "ensemble"
    if not model_name.startswith("profitability_"):
        model_name = f"profitability_{model_name}"

    model_path = MODEL_DIR / model_name

    if not model_path.exists():
        raise FileNotFoundError(f"Profitability model not found: {model_path}")

    profitability_model = joblib.load(model_path / "profitability_model.joblib")
    color_model = joblib.load(model_path / "color_model.joblib")

    with open(model_path / "metrics.json", "r") as f:
        metrics = json.load(f)

    return profitability_model, color_model, metrics


def train_and_compare_profitability_models(csv_path: str = "~/Desktop/es.csv", save: bool = True):
    """Train all profitability model types and compare results"""
    print("=" * 80)
    print("TRAINING AND COMPARING ALL PROFITABILITY MODEL TYPES")
    print("=" * 80)

    df = preprocess_data(csv_path)
    model_types = ["rf", "lgbm", "gb", "ensemble"]
    results = []

    for model_type in model_types:
        print(f"\n{'=' * 80}")
        print(f"TRAINING: {model_type.upper()}")
        print(f"{'=' * 80}")

        profitability_model, color_model, evals = train_profitability_model(
            df,
            features=None,
            model_type=model_type,
            test_size=0.2,
            use_advanced_features=True,
        )

        if save:
            save_profitability_model(profitability_model, color_model, evals, model_type, csv_path)

        results.append(
            {
                "model_type": model_type,
                "prof_f1": evals["metrics_prof"]["f1"],
                "prof_accuracy": evals["metrics_prof"]["accuracy"],
                "color_f1": evals["metrics_color"]["f1"],
                "color_accuracy": evals["metrics_color"]["accuracy"],
                "walk_forward_prof": evals["walk_forward_accuracy_prof"],
                "walk_forward_color": evals["walk_forward_accuracy_color"],
            }
        )

    # Create comparison table
    print("\n" + "=" * 80)
    print("PROFITABILITY & DIRECTION MODEL COMPARISON RESULTS")
    print("=" * 80)

    comparison_df = pd.DataFrame(results)
    print("\n=== COMBINED RESULTS ===")
    print(comparison_df.to_string(index=False))

    # Find best models
    best_prof = comparison_df.loc[comparison_df["prof_f1"].idxmax()]
    best_color = comparison_df.loc[comparison_df["color_f1"].idxmax()]

    print("\n" + "=" * 80)
    print("BEST MODELS")
    print("=" * 80)
    print(f"Best Profitability Model: {best_prof['model_type'].upper()} (F1: {best_prof['prof_f1']:.4f})")
    print(f"Best Direction Model: {best_color['model_type'].upper()} (F1: {best_color['color_f1']:.4f})")
    print("=" * 80)

    if save:
        comparison_df.to_csv(MODEL_DIR / f"profitability_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False)
        print(f"\n✓ Comparison saved to: {MODEL_DIR}")

    return comparison_df


def train_and_compare_all(csv_path: str = "~/Desktop/es.csv", save: bool = True):
    """Train all model types and compare results"""
    print("=" * 80)
    print("TRAINING AND COMPARING ALL MODEL TYPES")
    print("=" * 80)

    df = preprocess_data(csv_path)
    model_types = ["rf", "lgbm", "gb", "ensemble"]
    results = []

    for model_type in model_types:
        print(f"\n{'=' * 80}")
        print(f"TRAINING: {model_type.upper()}")
        print(f"{'=' * 80}")

        rr_model, color_model, sl_model, evals = train_three_models(
            df,
            features=None,
            rr_model_type=model_type,
            color_model_type=model_type,
            sl_model_type=model_type,
            test_size=0.2,
            use_advanced_features=True,
        )

        if save:
            save_models(rr_model, color_model, sl_model, evals, model_type, csv_path)

        results.append(
            {
                "model_type": model_type,
                "rr_f1": evals["rr"]["metrics"]["f1"],
                "rr_accuracy": evals["rr"]["metrics"]["accuracy"],
                "rr_recall": evals["rr"]["metrics"]["recall"],
                "rr_precision": evals["rr"]["metrics"]["precision"],
                "color_f1": evals["color"]["metrics"]["f1"],
                "color_accuracy": evals["color"]["metrics"]["accuracy"],
                "color_recall": evals["color"]["metrics"]["recall"],
                "color_precision": evals["color"]["metrics"]["precision"],
                "sl_f1": evals["sl_hit"]["metrics"]["f1"],
                "sl_accuracy": evals["sl_hit"]["metrics"]["accuracy"],
                "sl_recall": evals["sl_hit"]["metrics"]["recall"],
                "sl_precision": evals["sl_hit"]["metrics"]["precision"],
                "rr_next_proba": evals["rr"]["next_pred"]["proba"],
                "color_next_proba": evals["color"]["next_pred"]["proba"],
                "sl_next_proba": evals["sl_hit"]["next_pred"]["proba"],
            }
        )

    # Create comparison table
    print("\n" + "=" * 80)
    print("MODEL COMPARISON RESULTS")
    print("=" * 80)

    comparison_df = pd.DataFrame(results)
    print("\n=== RR MODEL (Predict RR>=2) ===")
    print(comparison_df[["model_type", "rr_f1", "rr_accuracy", "rr_precision", "rr_recall"]].to_string(index=False))

    print("\n=== COLOR MODEL (Predict GREEN) ===")
    print(comparison_df[["model_type", "color_f1", "color_accuracy", "color_precision", "color_recall"]].to_string(index=False))

    print("\n=== SL HIT MODEL (Predict 1.5x Wick SL Hit) ===")
    print(comparison_df[["model_type", "sl_f1", "sl_accuracy", "sl_precision", "sl_recall"]].to_string(index=False))

    print("\n=== NEXT BAR PREDICTIONS ===")
    print(comparison_df[["model_type", "rr_next_proba", "color_next_proba", "sl_next_proba"]].to_string(index=False))

    # Find best models
    best_rr = comparison_df.loc[comparison_df["rr_f1"].idxmax()]
    best_color = comparison_df.loc[comparison_df["color_f1"].idxmax()]
    best_sl = comparison_df.loc[comparison_df["sl_f1"].idxmax()]

    print("\n" + "=" * 80)
    print("BEST MODELS")
    print("=" * 80)
    print(f"Best RR Model: {best_rr['model_type'].upper()} (F1: {best_rr['rr_f1']:.4f})")
    print(f"Best Color Model: {best_color['model_type'].upper()} (F1: {best_color['color_f1']:.4f})")
    print(f"Best SL Hit Model: {best_sl['model_type'].upper()} (F1: {best_sl['sl_f1']:.4f})")
    print("=" * 80)

    if save:
        comparison_df.to_csv(MODEL_DIR / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False)
        print(f"\n✓ Comparison saved to: {MODEL_DIR}")

    return comparison_df


if __name__ == "__main__":
    main()
