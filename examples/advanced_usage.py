#!/usr/bin/env python3
"""
Advanced usage examples for the Earnings Move Predictor.
Demonstrates premium features and complex analysis.
"""

import sys
import os
sys.path.insert(0, '..')

from datetime import date, datetime, timedelta
from typing import List, Dict
import json


def example_premium_prediction():
    """
    Full prediction using premium data sources.
    Requires API keys to be set in environment.
    """
    from src.main import EarningsMovePredictor

    # Check for API keys
    api_keys = {
        'Unusual Whales': os.getenv('UNUSUAL_WHALES_API_KEY'),
        'Trading Economics': os.getenv('TRADING_ECONOMICS_API_KEY'),
        'Polygon': os.getenv('POLYGON_API_KEY'),
        'Alpha Vantage': os.getenv('ALPHA_VANTAGE_API_KEY'),
    }

    print("=== API Key Status ===")
    premium_available = False
    for name, key in api_keys.items():
        status = "Set" if key else "Not Set"
        print(f"  {name}: {status}")
        if key:
            premium_available = True

    if not premium_available:
        print("\nNo premium API keys found. Running in free mode.")
        mode = 'free'
    else:
        mode = 'premium'

    predictor = EarningsMovePredictor(mode=mode)

    # Get detailed prediction
    prediction = predictor.predict_single('NVDA', output_format='text')

    if prediction:
        # Export to JSON
        with open('prediction_output.json', 'w') as f:
            json.dump(prediction.to_dict(), f, indent=2)
        print(f"\nPrediction saved to prediction_output.json")


def example_custom_data_sources():
    """
    Example of using custom data source configuration.
    """
    from src.data_sources.free_sources import YahooFinanceSource, NasdaqEarningsSource
    from src.analysis.earnings_predictor import EarningsPredictor

    # Create predictor with specific sources
    yahoo = YahooFinanceSource()
    nasdaq = NasdaqEarningsSource()

    predictor = EarningsPredictor(
        earnings_sources=[yahoo, nasdaq],
        options_sources=[yahoo],
        flow_sources=[],  # No flow sources
        use_free_sources=False,  # Don't auto-add
        use_premium_sources=False
    )

    print("=== Custom Source Prediction ===")
    prediction = predictor.predict('AAPL')
    print(prediction.summary())


def example_options_chain_deep_dive():
    """
    Deep analysis of options chain.
    """
    from src.data_sources.free_sources import YahooFinanceSource
    from src.analysis.options_analyzer import OptionsAnalyzer
    from src.analysis.expected_move import ExpectedMoveCalculator
    import pandas as pd

    yahoo = YahooFinanceSource()
    analyzer = OptionsAnalyzer()
    calc = ExpectedMoveCalculator(analyzer)

    symbol = 'TSLA'
    print(f"\n=== Deep Options Analysis: {symbol} ===")

    # Get all expirations
    expirations = yahoo.get_option_expirations(symbol)
    print(f"\nAvailable Expirations: {len(expirations)}")
    for exp in expirations[:5]:
        print(f"  {exp}")

    # Analyze multiple expirations
    for exp in expirations[:3]:
        chain = yahoo.get_option_chain(symbol, exp)
        if chain:
            metrics = analyzer.analyze_chain(chain)
            expected_move = calc.calculate_expected_move(chain)

            print(f"\n--- Expiration: {exp} ---")
            print(f"  Expected Move: ±{expected_move.expected_move_percent:.2f}%")
            print(f"  ATM IV: {metrics.atm_iv*100:.1f}%")
            print(f"  Straddle Price: ${metrics.atm_straddle_price:.2f}")
            print(f"  Put/Call Ratio: {metrics.put_call_ratio:.2f}")
            print(f"  Max Pain: ${metrics.max_pain_strike:.2f}")


def example_earnings_screener():
    """
    Screen for high expected move earnings.
    """
    from src.main import EarningsMovePredictor

    predictor = EarningsMovePredictor(mode='free')

    # Define watchlist
    watchlist = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
        'NVDA', 'TSLA', 'AMD', 'NFLX', 'CRM',
        'SNOW', 'PLTR', 'COIN', 'SHOP', 'SQ'
    ]

    print("=== Earnings Screener ===")
    print("Scanning for high expected moves...\n")

    results = []
    predictions = predictor.predict_batch(watchlist, output_format='text')

    for p in predictions:
        if p.expected_move_percent > 5:
            results.append({
                'symbol': p.symbol,
                'price': p.current_price,
                'expected_move': p.expected_move_percent,
                'direction': p.predicted_direction,
                'confidence': p.overall_confidence,
                'iv_percentile': p.iv_percentile
            })

    # Sort by expected move
    results.sort(key=lambda x: x['expected_move'], reverse=True)

    print("\n=== High Expected Move Stocks ===")
    for r in results:
        print(f"{r['symbol']}: ±{r['expected_move']:.1f}% (IV Pctl: {r['iv_percentile']*100:.0f}%)")


def example_iron_condor_strategy():
    """
    Calculate optimal iron condor strikes based on expected move.
    """
    from src.data_sources.free_sources import YahooFinanceSource
    from src.analysis.options_analyzer import OptionsAnalyzer
    from src.analysis.expected_move import ExpectedMoveCalculator

    yahoo = YahooFinanceSource()
    analyzer = OptionsAnalyzer()
    calc = ExpectedMoveCalculator(analyzer)

    symbol = 'AAPL'
    chain = yahoo.get_option_chain(symbol)

    if not chain:
        print(f"Could not get chain for {symbol}")
        return

    expected_move = calc.calculate_expected_move(chain)
    metrics = analyzer.analyze_chain(chain)

    print(f"\n=== Iron Condor Strategy for {symbol} ===")
    print(f"Current Price: ${chain.underlying_price:.2f}")
    print(f"Expected Move: ±{expected_move.expected_move_percent:.2f}%")
    print(f"Expected Range: ${expected_move.lower_bound:.2f} - ${expected_move.upper_bound:.2f}")

    # Find strikes outside expected move
    short_put_target = expected_move.lower_bound * 0.95  # 5% buffer
    short_call_target = expected_move.upper_bound * 1.05

    # Find closest strikes
    put_strikes = sorted([p.strike for p in chain.puts if p.strike < chain.underlying_price])
    call_strikes = sorted([c.strike for c in chain.calls if c.strike > chain.underlying_price])

    if put_strikes and call_strikes:
        short_put = max([s for s in put_strikes if s <= short_put_target], default=put_strikes[0])
        short_call = min([s for s in call_strikes if s >= short_call_target], default=call_strikes[-1])

        # Long strikes (next strike out)
        long_put = max([s for s in put_strikes if s < short_put], default=short_put - 5)
        long_call = min([s for s in call_strikes if s > short_call], default=short_call + 5)

        print(f"\nSuggested Iron Condor:")
        print(f"  Long Put:  ${long_put:.2f}")
        print(f"  Short Put: ${short_put:.2f}")
        print(f"  Short Call: ${short_call:.2f}")
        print(f"  Long Call: ${long_call:.2f}")

        # Calculate theoretical credit
        short_put_opt = next((p for p in chain.puts if p.strike == short_put), None)
        short_call_opt = next((c for c in chain.calls if c.strike == short_call), None)
        long_put_opt = next((p for p in chain.puts if p.strike == long_put), None)
        long_call_opt = next((c for c in chain.calls if c.strike == long_call), None)

        if all([short_put_opt, short_call_opt, long_put_opt, long_call_opt]):
            credit = (
                (short_put_opt.bid + short_call_opt.bid) -
                (long_put_opt.ask + long_call_opt.ask)
            )
            max_risk = max(short_put - long_put, long_call - short_call) - credit

            print(f"\n  Estimated Credit: ${credit:.2f} per spread")
            print(f"  Max Risk: ${max_risk:.2f} per spread")
            print(f"  Max Profit: ${credit:.2f} per spread")
            if max_risk > 0:
                print(f"  Risk/Reward: 1:{credit/max_risk:.2f}")


def example_straddle_strategy():
    """
    Analyze straddle strategy for earnings play.
    """
    from src.data_sources.free_sources import YahooFinanceSource
    from src.analysis.options_analyzer import OptionsAnalyzer
    from src.analysis.expected_move import ExpectedMoveCalculator

    yahoo = YahooFinanceSource()
    analyzer = OptionsAnalyzer()
    calc = ExpectedMoveCalculator(analyzer)

    symbol = 'NVDA'
    chain = yahoo.get_option_chain(symbol)

    if not chain:
        print(f"Could not get chain for {symbol}")
        return

    expected_move = calc.calculate_expected_move(chain)
    metrics = analyzer.analyze_chain(chain)

    print(f"\n=== Straddle Analysis for {symbol} ===")
    print(f"Current Price: ${chain.underlying_price:.2f}")
    print(f"ATM Strike: ${metrics.atm_strike:.2f}")
    print(f"Straddle Cost: ${metrics.atm_straddle_price:.2f}")
    print(f"Implied Move: ±{expected_move.expected_move_percent:.2f}%")

    # Calculate breakevens
    upper_be = metrics.atm_strike + metrics.atm_straddle_price
    lower_be = metrics.atm_strike - metrics.atm_straddle_price

    print(f"\nBreakeven Points:")
    print(f"  Upper: ${upper_be:.2f} ({(upper_be/chain.underlying_price - 1)*100:+.2f}%)")
    print(f"  Lower: ${lower_be:.2f} ({(lower_be/chain.underlying_price - 1)*100:+.2f}%)")

    # Historical comparison
    print(f"\nHistorical Context:")
    print(f"  Avg Historical Move: ±{expected_move.avg_historical_move:.2f}%")
    if expected_move.avg_historical_move > 0:
        ratio = expected_move.expected_move_percent / expected_move.avg_historical_move
        if ratio > 1.2:
            print(f"  Straddle appears EXPENSIVE (IV {ratio:.1f}x historical)")
        elif ratio < 0.8:
            print(f"  Straddle appears CHEAP (IV {ratio:.1f}x historical)")
        else:
            print(f"  Straddle appears FAIRLY PRICED")


def example_iv_term_structure():
    """
    Analyze IV term structure across expirations.
    """
    from src.data_sources.free_sources import YahooFinanceSource
    from src.analysis.options_analyzer import OptionsAnalyzer
    import numpy as np

    yahoo = YahooFinanceSource()
    analyzer = OptionsAnalyzer()

    symbol = 'SPY'
    print(f"\n=== IV Term Structure: {symbol} ===")

    expirations = yahoo.get_option_expirations(symbol)

    iv_data = []
    for exp in expirations[:8]:
        chain = yahoo.get_option_chain(symbol, exp)
        if chain:
            metrics = analyzer.analyze_chain(chain)
            days_to_exp = (exp - date.today()).days
            iv_data.append({
                'expiration': exp,
                'days': days_to_exp,
                'iv': metrics.atm_iv
            })

    if iv_data:
        print(f"\n{'Expiration':<12} {'Days':>6} {'IV':>8}")
        print("-" * 28)
        for d in iv_data:
            print(f"{d['expiration']} {d['days']:>6} {d['iv']*100:>7.1f}%")

        # Check for term structure shape
        if len(iv_data) >= 3:
            front_iv = iv_data[0]['iv']
            back_iv = iv_data[-1]['iv']

            if front_iv > back_iv * 1.1:
                print("\nTerm Structure: INVERTED (backwardation)")
                print("Front month IV elevated - possible earnings/event pricing")
            elif back_iv > front_iv * 1.1:
                print("\nTerm Structure: NORMAL (contango)")
            else:
                print("\nTerm Structure: FLAT")


if __name__ == '__main__':
    print("=" * 60)
    print("Earnings Move Predictor - Advanced Examples")
    print("=" * 60)

    # Run examples
    print("\n[1] Premium Prediction")
    example_premium_prediction()

    print("\n[2] Options Chain Deep Dive")
    example_options_chain_deep_dive()

    print("\n[3] Straddle Strategy Analysis")
    example_straddle_strategy()

    print("\n[4] Iron Condor Strategy")
    example_iron_condor_strategy()

    print("\n[5] IV Term Structure")
    example_iv_term_structure()

    # Uncomment to run:
    # print("\n[6] Custom Data Sources")
    # example_custom_data_sources()

    # print("\n[7] Earnings Screener")
    # example_earnings_screener()
