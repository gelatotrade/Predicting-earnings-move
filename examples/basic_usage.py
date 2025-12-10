#!/usr/bin/env python3
"""
Basic usage examples for the Earnings Move Predictor.
"""

import sys
sys.path.insert(0, '..')

from datetime import date, timedelta

# Example 1: Quick prediction with free data sources
def example_quick_prediction():
    """Simple prediction using free data sources."""
    from src.main import EarningsMovePredictor

    # Initialize in free mode (no API keys required)
    predictor = EarningsMovePredictor(mode='free')

    # Predict for a single stock
    prediction = predictor.predict_single('AAPL', output_format='text')

    if prediction:
        print(f"\nExpected Move: ±{prediction.expected_move_percent:.2f}%")
        print(f"Direction: {prediction.predicted_direction}")
        print(f"Confidence: {prediction.overall_confidence*100:.0f}%")


# Example 2: Using the low-level API
def example_low_level_api():
    """Direct use of the analysis components."""
    from src.data_sources.free_sources import YahooFinanceSource
    from src.analysis.options_analyzer import OptionsAnalyzer
    from src.analysis.expected_move import ExpectedMoveCalculator, get_expected_move_summary

    # Get option chain
    yahoo = YahooFinanceSource()
    chain = yahoo.get_option_chain('MSFT')

    if chain:
        # Analyze options
        analyzer = OptionsAnalyzer()
        metrics = analyzer.analyze_chain(chain)

        print(f"\n=== Options Analysis for {chain.underlying_symbol} ===")
        print(f"Current Price: ${chain.underlying_price:.2f}")
        print(f"ATM Strike: ${metrics.atm_strike:.2f}")
        print(f"ATM Straddle: ${metrics.atm_straddle_price:.2f}")
        print(f"ATM IV: {metrics.atm_iv*100:.1f}%")
        print(f"Put/Call Ratio: {metrics.put_call_ratio:.2f}")
        print(f"Max Pain: ${metrics.max_pain_strike:.2f}")

        # Calculate expected move
        calc = ExpectedMoveCalculator(analyzer)
        expected_move = calc.calculate_expected_move(chain)

        print(f"\n{get_expected_move_summary(expected_move)}")


# Example 3: Batch prediction
def example_batch_prediction():
    """Predict multiple stocks at once."""
    from src.main import EarningsMovePredictor

    predictor = EarningsMovePredictor(mode='free')

    # Tech stocks with upcoming earnings
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA']

    print("\n=== Batch Prediction ===")
    predictions = predictor.predict_batch(symbols, output_format='table')

    # Additional analysis
    for p in predictions:
        if p.expected_move_percent > 5:
            print(f"\n{p.symbol}: High expected move of ±{p.expected_move_percent:.1f}%")


# Example 4: Get earnings calendar
def example_earnings_calendar():
    """Get upcoming earnings calendar."""
    from src.main import EarningsMovePredictor

    predictor = EarningsMovePredictor(mode='free')

    print("\n=== Earnings Calendar (Next 7 Days) ===")
    predictor.get_earnings_calendar(days_ahead=7, output_format='table')


# Example 5: Options flow analysis (requires premium API)
def example_flow_analysis():
    """Analyze options flow (premium feature)."""
    import os
    from src.main import EarningsMovePredictor

    # Check if API key is available
    if not os.getenv('UNUSUAL_WHALES_API_KEY'):
        print("\nOptions flow requires UNUSUAL_WHALES_API_KEY")
        print("Set your API key in .env file for premium features")
        return

    predictor = EarningsMovePredictor(mode='premium')
    prediction = predictor.predict_single('TSLA', output_format='text')

    if prediction and prediction.option_flows:
        print(f"\n=== Options Flow for {prediction.symbol} ===")
        print(f"Flow Sentiment: {prediction.flow_sentiment}")
        print(f"Net Premium: ${prediction.net_premium:,.0f}")
        print(f"Number of Unusual Trades: {len(prediction.option_flows)}")


# Example 6: Calculate expected move from IV
def example_iv_calculation():
    """Calculate expected move from implied volatility."""
    from src.analysis.expected_move import ExpectedMoveCalculator

    calc = ExpectedMoveCalculator()

    # Example: Stock at $150 with 45% IV
    stock_price = 150.0
    iv = 0.45  # 45% IV
    days = 1   # Single day (earnings)

    move_dollars, move_pct = calc.calculate_from_iv(stock_price, iv, days)

    print(f"\n=== IV-Based Expected Move ===")
    print(f"Stock Price: ${stock_price:.2f}")
    print(f"Implied Volatility: {iv*100:.0f}%")
    print(f"Expected Move: ±${move_dollars:.2f} (±{move_pct:.2f}%)")

    # Probability cone
    cone = calc.calculate_probability_cone(stock_price, iv, days)
    print(f"\nProbability Ranges:")
    for conf, (lower, upper) in cone.items():
        print(f"  {conf*100:.0f}% confidence: ${lower:.2f} - ${upper:.2f}")


# Example 7: Historical analysis
def example_historical_analysis():
    """Analyze historical earnings moves."""
    import yfinance as yf
    import numpy as np

    symbol = 'AAPL'
    ticker = yf.Ticker(symbol)

    try:
        earnings_hist = ticker.earnings_history
        if earnings_hist is not None and not earnings_hist.empty:
            print(f"\n=== Historical Earnings for {symbol} ===")

            surprises = []
            for idx, row in earnings_hist.iterrows():
                eps_actual = row.get('epsActual', 0)
                eps_estimate = row.get('epsEstimate', 0)

                if eps_estimate != 0:
                    surprise = ((eps_actual - eps_estimate) / abs(eps_estimate)) * 100
                    surprises.append(surprise)
                    print(f"{idx}: Est ${eps_estimate:.2f}, Actual ${eps_actual:.2f}, Surprise {surprise:+.1f}%")

            if surprises:
                print(f"\nAverage Surprise: {np.mean(surprises):+.1f}%")
                print(f"Beat Rate: {sum(1 for s in surprises if s > 0) / len(surprises) * 100:.0f}%")

    except Exception as e:
        print(f"Could not get historical data: {e}")


if __name__ == '__main__':
    print("=" * 60)
    print("Earnings Move Predictor - Examples")
    print("=" * 60)

    # Run examples
    print("\n[1] Quick Prediction")
    example_quick_prediction()

    print("\n[2] Low-Level API")
    example_low_level_api()

    print("\n[3] IV Calculation")
    example_iv_calculation()

    print("\n[4] Historical Analysis")
    example_historical_analysis()

    # Uncomment to run these:
    # print("\n[5] Batch Prediction")
    # example_batch_prediction()

    # print("\n[6] Earnings Calendar")
    # example_earnings_calendar()
