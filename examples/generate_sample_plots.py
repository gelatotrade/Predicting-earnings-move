#!/usr/bin/env python3
"""
Generate sample visualization plots for documentation.

This script creates example plots using simulated prediction data
to demonstrate the visualization capabilities of the Earnings Move Predictor.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, date, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional

# Import visualization functions
from src.visualization.plots import (
    plot_expected_move_range,
    plot_direction_signals,
    plot_historical_moves,
    plot_options_analysis,
    plot_probability_cone,
    plot_confidence_gauge
)

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt


def generate_sample_plots():
    """Generate all sample plots for documentation."""

    # Output directory
    output_dir = Path(__file__).parent / 'plots'
    output_dir.mkdir(exist_ok=True)

    print("Generating sample visualization plots...")

    # Sample data for AAPL
    symbol = "AAPL"
    current_price = 178.50
    expected_move_percent = 4.52
    expected_range = (current_price * (1 - expected_move_percent/100),
                      current_price * (1 + expected_move_percent/100))
    predicted_direction = "up"

    # Historical moves (quarterly earnings reactions)
    historical_moves = [2.45, -1.80, 3.20, -2.10, 4.50, -0.90, 2.80, 1.50, -3.20, 2.10, 1.80, -1.50]

    # Direction signals
    direction_signals = {
        'put_call_ratio': {'value': 0.85, 'signal': 'bullish'},
        'iv_skew': {'value': 0.02, 'signal': 'neutral'},
        'flow': {'bullish': 15, 'bearish': 8, 'signal': 'bullish'},
        'beat_rate': {'value': 0.87, 'signal': 'bullish'},
        'recent_trend': {'value': 2.15, 'signal': 'bullish'}
    }

    # Options metrics
    atm_strike = 180.0
    put_call_ratio = 0.85
    max_pain_strike = 175.0
    iv_percentile = 0.78
    atm_iv = 0.42
    flow_sentiment = "bullish"

    # Confidence metrics
    overall_confidence = 0.72
    direction_confidence = 0.625
    data_sources = ["Yahoo Finance", "Nasdaq", "Options Flow"]

    print(f"  Generating plots for {symbol}...")

    # 1. Expected Move Range Plot
    print("    - Expected move range plot...")
    fig1 = plot_expected_move_range(
        symbol=symbol,
        current_price=current_price,
        expected_move_percent=expected_move_percent,
        expected_range=expected_range,
        historical_moves=historical_moves[:5],
        predicted_direction=predicted_direction,
        save_path=str(output_dir / 'expected_move_range.png')
    )
    plt.close(fig1)

    # 2. Direction Signals Plot
    print("    - Direction signals plot...")
    fig2 = plot_direction_signals(
        symbol=symbol,
        direction_signals=direction_signals,
        direction_confidence=direction_confidence,
        predicted_direction=predicted_direction,
        save_path=str(output_dir / 'direction_signals.png')
    )
    plt.close(fig2)

    # 3. Historical Moves Plot
    print("    - Historical moves plot...")
    fig3 = plot_historical_moves(
        symbol=symbol,
        historical_moves=historical_moves,
        expected_move_percent=expected_move_percent,
        avg_historical_move=2.35,
        beat_rate=0.87,
        save_path=str(output_dir / 'historical_moves.png')
    )
    plt.close(fig3)

    # 4. Options Analysis Dashboard
    print("    - Options analysis dashboard...")
    fig4 = plot_options_analysis(
        symbol=symbol,
        current_price=current_price,
        atm_strike=atm_strike,
        put_call_ratio=put_call_ratio,
        max_pain_strike=max_pain_strike,
        iv_percentile=iv_percentile,
        atm_iv=atm_iv,
        flow_sentiment=flow_sentiment,
        save_path=str(output_dir / 'options_analysis.png')
    )
    plt.close(fig4)

    # 5. Probability Cone Plot
    print("    - Probability cone plot...")
    fig5 = plot_probability_cone(
        symbol=symbol,
        current_price=current_price,
        expected_move_percent=expected_move_percent,
        confidence_levels=[0.50, 0.68, 0.95],
        save_path=str(output_dir / 'probability_cone.png')
    )
    plt.close(fig5)

    # 6. Confidence Gauge
    print("    - Confidence gauge plot...")
    fig6 = plot_confidence_gauge(
        symbol=symbol,
        overall_confidence=overall_confidence,
        direction_confidence=direction_confidence,
        predicted_direction=predicted_direction,
        data_sources=data_sources,
        save_path=str(output_dir / 'confidence_gauge.png')
    )
    plt.close(fig6)

    print(f"\nAll plots saved to: {output_dir}")
    print("\nGenerated files:")
    for f in sorted(output_dir.glob('*.png')):
        print(f"  - {f.name}")


if __name__ == '__main__':
    generate_sample_plots()
