"""
Expected move calculator for earnings events.
Calculates the implied expected move from options pricing.
"""

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional, List, Dict, Tuple, Any
import math

import numpy as np
from scipy.stats import norm

from ..data_sources.base import OptionChain, OptionContract
from .options_analyzer import OptionsAnalyzer, OptionsMetrics

logger = logging.getLogger(__name__)


@dataclass
class ExpectedMoveResult:
    """Container for expected move calculation results."""
    symbol: str
    underlying_price: float
    expiration_date: date
    earnings_date: Optional[datetime] = None

    # Primary expected move (from ATM straddle)
    expected_move_dollars: float = 0
    expected_move_percent: float = 0

    # Upper and lower bounds
    upper_bound: float = 0
    lower_bound: float = 0

    # Alternative calculations
    straddle_expected_move: float = 0
    strangle_expected_move: float = 0
    iron_condor_expected_move: float = 0

    # Probability analysis
    prob_up: float = 0.5
    prob_down: float = 0.5
    prob_exceed_expected: float = 0

    # IV and time metrics
    implied_volatility: float = 0
    days_to_expiration: int = 0
    earnings_iv_premium: float = 0

    # Historical comparison
    avg_historical_move: float = 0
    historical_beat_rate: float = 0

    # Confidence metrics
    confidence_score: float = 0
    data_quality: str = "unknown"

    # Raw data used
    atm_straddle_price: float = 0
    atm_strangle_price: float = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'underlying_price': self.underlying_price,
            'expected_move_pct': round(self.expected_move_percent, 2),
            'expected_move_dollars': round(self.expected_move_dollars, 2),
            'upper_bound': round(self.upper_bound, 2),
            'lower_bound': round(self.lower_bound, 2),
            'implied_volatility': round(self.implied_volatility * 100, 2),
            'prob_up': round(self.prob_up * 100, 2),
            'prob_down': round(self.prob_down * 100, 2),
            'confidence': round(self.confidence_score * 100, 2),
            'data_quality': self.data_quality
        }


class ExpectedMoveCalculator:
    """
    Calculator for expected moves around earnings events.

    The expected move is primarily calculated from the ATM straddle price,
    which represents the market's implied expectation of price movement.
    """

    def __init__(
        self,
        analyzer: Optional[OptionsAnalyzer] = None,
        historical_data: Optional[Dict] = None
    ):
        """
        Initialize calculator.

        Args:
            analyzer: Options analyzer instance
            historical_data: Optional dict of historical earnings moves
        """
        self.analyzer = analyzer or OptionsAnalyzer()
        self.historical_data = historical_data or {}

    def calculate_expected_move(
        self,
        chain: OptionChain,
        earnings_date: Optional[datetime] = None,
        include_historical: bool = True
    ) -> ExpectedMoveResult:
        """
        Calculate the expected move from an option chain.

        The primary method uses the ATM straddle price:
        Expected Move % = (ATM Straddle Price / Stock Price) * 100

        Args:
            chain: Option chain data
            earnings_date: Date of earnings announcement
            include_historical: Whether to include historical analysis

        Returns:
            ExpectedMoveResult with all calculations
        """
        underlying_price = chain.underlying_price
        atm_strike = chain.get_atm_strike()

        # Get ATM options
        atm_call = self._get_closest_option(chain.calls, atm_strike)
        atm_put = self._get_closest_option(chain.puts, atm_strike)

        # Calculate ATM straddle price
        atm_straddle_price = chain.get_atm_straddle_price() or 0

        # Calculate strangle (one strike OTM)
        atm_strangle_price = self._calculate_strangle_price(chain, atm_strike)

        # Primary expected move from straddle
        expected_move_dollars = atm_straddle_price
        expected_move_percent = (atm_straddle_price / underlying_price) * 100 if underlying_price > 0 else 0

        # Upper and lower bounds
        upper_bound = underlying_price + expected_move_dollars
        lower_bound = underlying_price - expected_move_dollars

        # Alternative expected move calculations
        straddle_em = expected_move_percent
        strangle_em = (atm_strangle_price / underlying_price) * 100 if underlying_price > 0 else 0
        iron_condor_em = self._calculate_iron_condor_move(chain, atm_strike)

        # Days to expiration
        days_to_exp = max((chain.expiration_date - date.today()).days, 0)

        # Get IV from analyzer
        metrics = self.analyzer.analyze_chain(chain)
        atm_iv = metrics.atm_iv

        # Calculate probability estimates
        prob_up, prob_down = self._estimate_direction_probability(chain, metrics)
        prob_exceed = self._calculate_prob_exceed_expected(atm_iv, days_to_exp)

        # Historical comparison
        avg_historical = 0
        beat_rate = 0
        if include_historical and chain.underlying_symbol in self.historical_data:
            hist = self.historical_data[chain.underlying_symbol]
            avg_historical = hist.get('avg_move', 0)
            beat_rate = hist.get('beat_rate', 0)

        # IV premium (vs typical IV)
        earnings_iv_premium = self._calculate_iv_premium(chain, atm_iv)

        # Confidence score
        confidence = self._calculate_confidence(chain, atm_straddle_price)

        # Data quality assessment
        data_quality = self._assess_data_quality(chain, atm_call, atm_put)

        return ExpectedMoveResult(
            symbol=chain.underlying_symbol,
            underlying_price=underlying_price,
            expiration_date=chain.expiration_date,
            earnings_date=earnings_date,
            expected_move_dollars=expected_move_dollars,
            expected_move_percent=expected_move_percent,
            upper_bound=upper_bound,
            lower_bound=lower_bound,
            straddle_expected_move=straddle_em,
            strangle_expected_move=strangle_em,
            iron_condor_expected_move=iron_condor_em,
            prob_up=prob_up,
            prob_down=prob_down,
            prob_exceed_expected=prob_exceed,
            implied_volatility=atm_iv,
            days_to_expiration=days_to_exp,
            earnings_iv_premium=earnings_iv_premium,
            avg_historical_move=avg_historical,
            historical_beat_rate=beat_rate,
            confidence_score=confidence,
            data_quality=data_quality,
            atm_straddle_price=atm_straddle_price,
            atm_strangle_price=atm_strangle_price
        )

    def calculate_from_iv(
        self,
        underlying_price: float,
        implied_volatility: float,
        days_to_earnings: int = 1
    ) -> Tuple[float, float]:
        """
        Calculate expected move from implied volatility.

        Uses the formula: Expected Move = Stock Price * IV * sqrt(Days/365)

        For a single day (earnings), we typically use:
        Expected Move = Stock Price * IV * sqrt(1/365) * sqrt(2/pi)

        The sqrt(2/pi) adjustment (~0.798) converts from standard deviation
        to expected absolute move.

        Args:
            underlying_price: Current stock price
            implied_volatility: ATM implied volatility (as decimal)
            days_to_earnings: Days until earnings

        Returns:
            Tuple of (expected_move_dollars, expected_move_percent)
        """
        # One standard deviation move
        one_sd_move = underlying_price * implied_volatility * math.sqrt(days_to_earnings / 365)

        # Expected absolute move (mean absolute deviation)
        # For a normal distribution: E[|X|] = sigma * sqrt(2/pi)
        expected_move = one_sd_move * math.sqrt(2 / math.pi)

        expected_move_pct = (expected_move / underlying_price) * 100

        return expected_move, expected_move_pct

    def calculate_probability_cone(
        self,
        underlying_price: float,
        implied_volatility: float,
        days: int,
        confidence_levels: List[float] = [0.5, 0.68, 0.95]
    ) -> Dict[float, Tuple[float, float]]:
        """
        Calculate price ranges for different probability levels.

        Args:
            underlying_price: Current stock price
            implied_volatility: ATM IV
            days: Days to expiration
            confidence_levels: List of confidence levels

        Returns:
            Dict mapping confidence level to (lower, upper) bounds
        """
        daily_vol = implied_volatility / math.sqrt(252)
        period_vol = daily_vol * math.sqrt(days)

        result = {}
        for conf in confidence_levels:
            z_score = norm.ppf((1 + conf) / 2)
            move = underlying_price * period_vol * z_score
            result[conf] = (underlying_price - move, underlying_price + move)

        return result

    def _get_closest_option(
        self,
        options: List[OptionContract],
        target_strike: float
    ) -> Optional[OptionContract]:
        """Get option closest to target strike."""
        if not options:
            return None
        return min(options, key=lambda x: abs(x.strike - target_strike))

    def _get_mid_price(self, option: Optional[OptionContract]) -> float:
        """Get mid price of option."""
        if not option:
            return 0
        if option.bid > 0 and option.ask > 0:
            return (option.bid + option.ask) / 2
        return option.last_price

    def _calculate_strangle_price(
        self,
        chain: OptionChain,
        atm_strike: float
    ) -> float:
        """Calculate ATM strangle price (one strike OTM for each)."""
        # Get strikes
        call_strikes = sorted([c.strike for c in chain.calls if c.strike > atm_strike])
        put_strikes = sorted([p.strike for p in chain.puts if p.strike < atm_strike], reverse=True)

        if not call_strikes or not put_strikes:
            return 0

        otm_call_strike = call_strikes[0]
        otm_put_strike = put_strikes[0]

        otm_call = self._get_closest_option(chain.calls, otm_call_strike)
        otm_put = self._get_closest_option(chain.puts, otm_put_strike)

        return self._get_mid_price(otm_call) + self._get_mid_price(otm_put)

    def _calculate_iron_condor_move(
        self,
        chain: OptionChain,
        atm_strike: float
    ) -> float:
        """
        Estimate expected move from iron condor pricing.
        The move where a short IC would break even.
        """
        # Get strikes 1 and 2 steps OTM
        call_strikes = sorted([c.strike for c in chain.calls if c.strike > atm_strike])
        put_strikes = sorted([p.strike for p in chain.puts if p.strike < atm_strike], reverse=True)

        if len(call_strikes) < 2 or len(put_strikes) < 2:
            return 0

        # Short strikes
        short_call = self._get_closest_option(chain.calls, call_strikes[0])
        short_put = self._get_closest_option(chain.puts, put_strikes[0])

        # Long strikes (wings)
        long_call = self._get_closest_option(chain.calls, call_strikes[1])
        long_put = self._get_closest_option(chain.puts, put_strikes[1])

        if not all([short_call, short_put, long_call, long_put]):
            return 0

        # Credit received
        credit = (
            self._get_mid_price(short_call) + self._get_mid_price(short_put) -
            self._get_mid_price(long_call) - self._get_mid_price(long_put)
        )

        # Breakeven is roughly at short strikes +/- credit
        if credit > 0:
            upper_break = call_strikes[0] + credit
            lower_break = put_strikes[0] - credit
            move = max(upper_break - chain.underlying_price,
                      chain.underlying_price - lower_break)
            return (move / chain.underlying_price) * 100

        return 0

    def _estimate_direction_probability(
        self,
        chain: OptionChain,
        metrics: OptionsMetrics
    ) -> Tuple[float, float]:
        """
        Estimate probability of up vs down move.

        Uses put/call skew and volume ratios.
        """
        # Base probability (50/50)
        prob_up = 0.5
        prob_down = 0.5

        # Adjust based on IV skew
        if metrics.iv_skew != 0:
            # Higher put IV suggests more demand for downside protection
            skew_adjustment = min(abs(metrics.iv_skew) * 10, 0.15)
            if metrics.iv_skew > 0:  # Puts more expensive
                prob_down += skew_adjustment
                prob_up -= skew_adjustment
            else:
                prob_up += skew_adjustment
                prob_down -= skew_adjustment

        # Adjust based on put/call volume ratio
        if metrics.put_call_ratio != 0:
            if metrics.put_call_ratio > 1.5:  # Heavy put volume
                prob_down += 0.05
                prob_up -= 0.05
            elif metrics.put_call_ratio < 0.7:  # Heavy call volume
                prob_up += 0.05
                prob_down -= 0.05

        # Adjust based on flow sentiment
        if metrics.flow_sentiment == 'bullish':
            prob_up += 0.1
            prob_down -= 0.1
        elif metrics.flow_sentiment == 'bearish':
            prob_down += 0.1
            prob_up -= 0.1

        # Normalize
        total = prob_up + prob_down
        return prob_up / total, prob_down / total

    def _calculate_prob_exceed_expected(
        self,
        implied_volatility: float,
        days_to_exp: int
    ) -> float:
        """
        Calculate probability of exceeding the expected move.

        For a straddle priced at one standard deviation,
        probability of exceeding is about 32% (both tails combined).
        """
        # Roughly 32% chance of exceeding 1 SD in either direction
        # This is 2 * (1 - norm.cdf(1)) ≈ 0.317
        base_prob = 0.32

        # Adjust slightly based on days to expiration
        # More time = more chances for big moves
        if days_to_exp <= 1:
            return base_prob
        elif days_to_exp <= 7:
            return base_prob + 0.05
        else:
            return base_prob + 0.10

    def _calculate_iv_premium(
        self,
        chain: OptionChain,
        current_iv: float
    ) -> float:
        """
        Calculate how much IV is elevated vs typical levels.
        Requires historical IV data.
        """
        # Placeholder - would need historical IV data
        # For now, estimate based on IV level
        if current_iv > 0.8:  # Very high IV
            return 0.5
        elif current_iv > 0.5:  # High IV
            return 0.3
        elif current_iv > 0.3:  # Moderate IV
            return 0.1
        return 0

    def _calculate_confidence(
        self,
        chain: OptionChain,
        straddle_price: float
    ) -> float:
        """
        Calculate confidence score for the expected move estimate.
        Based on data quality and liquidity.
        """
        confidence = 0.5  # Base confidence

        # Liquidity check
        total_volume = sum(c.volume for c in chain.calls) + sum(p.volume for p in chain.puts)
        if total_volume > 10000:
            confidence += 0.2
        elif total_volume > 1000:
            confidence += 0.1

        # Bid-ask spread check
        atm_strike = chain.get_atm_strike()
        atm_call = self._get_closest_option(chain.calls, atm_strike)
        atm_put = self._get_closest_option(chain.puts, atm_strike)

        if atm_call and atm_put:
            call_spread = (atm_call.ask - atm_call.bid) / atm_call.ask if atm_call.ask > 0 else 1
            put_spread = (atm_put.ask - atm_put.bid) / atm_put.ask if atm_put.ask > 0 else 1
            avg_spread = (call_spread + put_spread) / 2

            if avg_spread < 0.05:
                confidence += 0.2
            elif avg_spread < 0.10:
                confidence += 0.1

        # Straddle price reasonableness
        if straddle_price > 0 and chain.underlying_price > 0:
            straddle_pct = straddle_price / chain.underlying_price
            if 0.02 < straddle_pct < 0.20:  # Reasonable range
                confidence += 0.1

        return min(confidence, 1.0)

    def _assess_data_quality(
        self,
        chain: OptionChain,
        atm_call: Optional[OptionContract],
        atm_put: Optional[OptionContract]
    ) -> str:
        """Assess the quality of the data used for calculation."""
        issues = []

        if not atm_call or not atm_put:
            return "poor"

        if atm_call.bid == 0 or atm_call.ask == 0:
            issues.append("missing_call_quotes")
        if atm_put.bid == 0 or atm_put.ask == 0:
            issues.append("missing_put_quotes")

        if atm_call.volume == 0 and atm_put.volume == 0:
            issues.append("no_volume")

        total_volume = sum(c.volume for c in chain.calls) + sum(p.volume for p in chain.puts)
        if total_volume < 100:
            issues.append("low_liquidity")

        if len(issues) == 0:
            return "good"
        elif len(issues) == 1:
            return "moderate"
        else:
            return "poor"


def get_expected_move_summary(result: ExpectedMoveResult) -> str:
    """Generate a human-readable summary of expected move."""
    lines = [
        f"Expected Move for {result.symbol}",
        "=" * 40,
        f"Current Price: ${result.underlying_price:.2f}",
        f"Expiration: {result.expiration_date}",
        "",
        f"Expected Move: ±{result.expected_move_percent:.2f}% (${result.expected_move_dollars:.2f})",
        f"Expected Range: ${result.lower_bound:.2f} - ${result.upper_bound:.2f}",
        "",
        f"ATM IV: {result.implied_volatility * 100:.1f}%",
        f"Probability Up: {result.prob_up * 100:.1f}%",
        f"Probability Down: {result.prob_down * 100:.1f}%",
        "",
        f"Confidence Score: {result.confidence_score * 100:.0f}%",
        f"Data Quality: {result.data_quality}",
    ]

    if result.avg_historical_move > 0:
        lines.extend([
            "",
            f"Historical Avg Move: ±{result.avg_historical_move:.2f}%",
            f"Historical Beat Rate: {result.historical_beat_rate * 100:.0f}%",
        ])

    return "\n".join(lines)
