"""
Monte Carlo Simulation for Profitability Model

This module simulates trading performance using the profitability model
to determine optimal entry thresholds and expected returns.
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def simulate_trading(
    model,
    X_test,
    y_test,
    df_test,
    probability_threshold=0.60,
    initial_capital=10000,
    risk_per_trade_pct=1.0,
    commission_per_trade=2.0,
    point_value=5.0,  # MES futures: $5 per point
    validate_columns=True,
):
    """
    Simulate trading based on model predictions

    Args:
        model: Trained profitability model
        X_test: Test features
        y_test: Actual outcomes (1=profitable, 0=unprofitable)
                NOTE: For NextBarColorAndWicksTarget, y_test=0 means EITHER:
                - SL was hit (full loss), OR
                - Bar closed wrong direction but SL NOT hit (smaller loss)
                This function now checks actual SL hits for accurate P&L.
        df_test: Test dataframe with OHLC data (must include 'high' and 'low')
        probability_threshold: Minimum probability to enter trade (0-1)
        initial_capital: Starting capital
        risk_per_trade_pct: Risk per trade as % of capital
        commission_per_trade: Commission per trade (both entry and exit)
        validate_columns: Validate required columns exist (default: True)

    Returns:
        dict with simulation results including:
        - P&L calculated from actual bar movement (not just target label)
        - SL hit tracking for accurate loss calculation
        - Outcome categorization: WIN, LOSS, or LOSS_SL
    """
    # Validate required columns if requested
    if validate_columns:
        required_cols = ["wicks_diff_sma14", "is_green", "open", "close", "high", "low"]
        missing_cols = [col for col in required_cols if col not in df_test.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns for Monte Carlo simulation: {missing_cols}. "
                f"Make sure data is processed with add_advanced_indicators(). "
                f"'high' and 'low' are required for accurate SL hit detection."
            )

    # Get predictions
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= probability_threshold).astype(int)

    # Initialize tracking
    capital = initial_capital
    equity_curve = [capital]
    trades = []
    total_trades = 0
    winning_trades = 0
    losing_trades = 0

    # Simulate each potential trade
    for i in range(len(X_test)):
        # Check if model says to enter
        if predictions[i] == 0:
            equity_curve.append(capital)
            continue

        total_trades += 1
        prob = probabilities[i]
        actual_profitable = y_test.iloc[i]

        # Get bar data
        bar = df_test.iloc[i]
        entry = bar["open"]
        exit_price = bar["close"]
        is_green = bar["is_green"]

        # Get datetime (handle both column and index)
        if "datetime" in bar.index:
            bar_datetime = bar["datetime"]
        elif hasattr(df_test.index, "to_pydatetime"):
            bar_datetime = df_test.index[i]
        else:
            bar_datetime = i  # Use index number as fallback

        # Calculate position size based on risk
        risk_amount = capital * (risk_per_trade_pct / 100)

        # Calculate SL distance in points
        wicks = bar["wicks_diff_sma14"]
        sl_distance_points = wicks * 1.5

        # Number of contracts = risk / (SL distance in points × point value)
        contracts = risk_amount / (sl_distance_points * point_value)

        # Limit to maximum reasonable contracts based on capital
        max_contracts = capital / (entry * point_value)  # Can't trade more than capital allows
        contracts = min(contracts, max_contracts, 10)  # Cap at 10 contracts for safety

        # If contracts < 0.01, skip (position too small)
        if contracts < 0.01:
            equity_curve.append(capital)
            continue

        # Calculate P&L using futures contract logic
        # IMPORTANT: For NextBarColorAndWicksTarget, actual_profitable=0 can mean:
        # 1. SL was hit (full SL loss), OR
        # 2. Bar closed in wrong direction but SL NOT hit (smaller loss/gain)
        # We need to check if SL was actually hit to calculate accurate P&L

        # Check if SL was actually hit during the bar
        sl_was_hit = False
        if is_green:
            # LONG trade
            sl_price = entry - sl_distance_points
            sl_was_hit = bar["low"] <= sl_price
        else:
            # SHORT trade
            sl_price = entry + sl_distance_points
            sl_was_hit = bar["high"] >= sl_price

        if sl_was_hit:
            # Stop loss was hit - use fixed SL loss
            pnl = -contracts * sl_distance_points * point_value
            pnl -= commission_per_trade
            capital += pnl
            losing_trades += 1
            outcome = "LOSS_SL"
        else:
            # SL NOT hit - calculate actual P&L to exit (could be profit or loss)
            if is_green:
                # LONG trade: profit from price increase
                points_gained = exit_price - entry
            else:
                # SHORT trade: profit from price decrease
                points_gained = entry - exit_price

            # P&L = contracts × points × point_value
            pnl = contracts * points_gained * point_value
            pnl -= commission_per_trade

            capital += pnl

            # Categorize as win or loss based on actual P&L
            if pnl > 0:
                winning_trades += 1
                outcome = "WIN"
            else:
                losing_trades += 1
                outcome = "LOSS"

        # Prevent capital from going negative (bankruptcy)
        if capital < 100:
            capital = 0
            break

        # Track trade
        trades.append(
            {
                "trade_num": total_trades,
                "datetime": bar_datetime,
                "direction": "LONG" if is_green else "SHORT",
                "entry": entry,
                "exit": exit_price,
                "probability": prob,
                "actual_outcome": actual_profitable,
                "sl_was_hit": sl_was_hit,
                "pnl": pnl,
                "capital": capital,
                "outcome": outcome,
            }
        )

        equity_curve.append(capital)

    # Calculate statistics
    if total_trades > 0:
        win_rate = (winning_trades / total_trades) * 100
        trades_df = pd.DataFrame(trades)

        winning_pnl = trades_df[trades_df["outcome"] == "WIN"]["pnl"]
        losing_pnl = trades_df[trades_df["outcome"] == "LOSS"]["pnl"]

        avg_win = winning_pnl.mean() if len(winning_pnl) > 0 else 0
        avg_loss = abs(losing_pnl.mean()) if len(losing_pnl) > 0 else 0
        profit_factor = abs(winning_pnl.sum() / losing_pnl.sum()) if losing_pnl.sum() != 0 else float("inf")

        # Drawdown calculation
        equity_series = pd.Series(equity_curve)
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max * 100
        max_drawdown = drawdown.min()

        # Sharpe ratio (simplified)
        returns = equity_series.pct_change().dropna()
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
    else:
        win_rate = 0
        avg_win = 0
        avg_loss = 0
        profit_factor = 0
        max_drawdown = 0
        sharpe = 0
        trades_df = pd.DataFrame()

    return {
        "threshold": probability_threshold,
        "initial_capital": initial_capital,
        "final_capital": capital,
        "total_return": ((capital - initial_capital) / initial_capital) * 100,
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe,
        "equity_curve": equity_curve,
        "trades": trades_df,
    }


def run_monte_carlo(
    model,
    X_test,
    y_test,
    df_test,
    thresholds=[0.50, 0.55, 0.60, 0.65, 0.70, 0.75],
    n_simulations=100,
    initial_capital=10000,
    risk_per_trade_pct=1.0,
):
    """
    Run Monte Carlo simulation across different probability thresholds

    Args:
        model: Trained model
        X_test: Test features
        y_test: Actual outcomes
        df_test: Test dataframe
        thresholds: List of probability thresholds to test
        n_simulations: Number of simulations per threshold
        initial_capital: Starting capital
        risk_per_trade_pct: Risk per trade

    Returns:
        dict with results for each threshold
    """
    results = {}

    for threshold in thresholds:
        print(f"\nRunning simulations for threshold: {threshold:.2%}")

        sim_results = []
        for sim_num in range(n_simulations):
            # Add random noise to simulate market uncertainty
            # Resample test set with replacement
            indices = np.random.choice(len(X_test), size=len(X_test), replace=True)

            X_sim = X_test.iloc[indices].reset_index(drop=True)
            y_sim = y_test.iloc[indices].reset_index(drop=True)
            df_sim = df_test.iloc[indices].reset_index(drop=True)

            # Run simulation
            result = simulate_trading(
                model=model,
                X_test=X_sim,
                y_test=y_sim,
                df_test=df_sim,
                probability_threshold=threshold,
                initial_capital=initial_capital,
                risk_per_trade_pct=risk_per_trade_pct,
            )

            sim_results.append(result)

        # Aggregate results
        final_capitals = [r["final_capital"] for r in sim_results]
        total_returns = [r["total_return"] for r in sim_results]
        win_rates = [r["win_rate"] for r in sim_results]
        profit_factors = [r["profit_factor"] for r in sim_results if r["profit_factor"] != float("inf")]
        max_drawdowns = [r["max_drawdown"] for r in sim_results]
        total_trades = [r["total_trades"] for r in sim_results]

        results[threshold] = {
            "avg_final_capital": np.mean(final_capitals),
            "std_final_capital": np.std(final_capitals),
            "avg_return": np.mean(total_returns),
            "std_return": np.std(total_returns),
            "min_return": np.min(total_returns),
            "max_return": np.max(total_returns),
            "avg_win_rate": np.mean(win_rates),
            "avg_profit_factor": np.mean(profit_factors) if profit_factors else 0,
            "avg_max_drawdown": np.mean(max_drawdowns),
            "avg_trades": np.mean(total_trades),
            "probability_of_profit": (np.array(total_returns) > 0).sum() / n_simulations * 100,
        }

    return results


def print_monte_carlo_results(results):
    """Print Monte Carlo results in a formatted table"""
    print("\n" + "=" * 120)
    print("MONTE CARLO SIMULATION RESULTS")
    print("=" * 120)

    print(
        f"\n{'Threshold':<12} {'Avg Return':<12} {'Std Return':<12} {'Win Rate':<12} "
        f"{'Profit Factor':<15} {'Max DD':<12} {'Avg Trades':<12} {'Prob Profit':<12}"
    )
    print("-" * 120)

    for threshold, result in sorted(results.items()):
        print(
            f"{threshold:<12.2%} "
            f"{result['avg_return']:<12.2f}% "
            f"{result['std_return']:<12.2f}% "
            f"{result['avg_win_rate']:<12.2f}% "
            f"{result['avg_profit_factor']:<15.2f} "
            f"{result['avg_max_drawdown']:<12.2f}% "
            f"{result['avg_trades']:<12.0f} "
            f"{result['probability_of_profit']:<12.2f}%"
        )

    # Find optimal threshold
    best_threshold = max(results.items(), key=lambda x: x[1]["avg_return"])
    print("\n" + "=" * 120)
    print(f"OPTIMAL THRESHOLD: {best_threshold[0]:.2%}")
    print(f"  Average Return: {best_threshold[1]['avg_return']:.2f}%")
    print(f"  Probability of Profit: {best_threshold[1]['probability_of_profit']:.2f}%")
    print(f"  Average Win Rate: {best_threshold[1]['avg_win_rate']:.2f}%")
    print("=" * 120)


def save_simulation_results(results, model_name, output_dir="trained"):
    """Save simulation results to JSON"""
    output_path = Path(output_dir) / f"{model_name}_monte_carlo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # Convert to serializable format
    serializable_results = {}
    for threshold, result in results.items():
        serializable_results[str(threshold)] = {
            k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in result.items()
        }

    with open(output_path, "w") as f:
        json.dump(serializable_results, f, indent=2)

    print(f"\n✓ Simulation results saved to: {output_path}")
    return output_path
