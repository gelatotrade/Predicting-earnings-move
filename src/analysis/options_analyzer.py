"""
Options chain analyzer for extracting key metrics and insights.
"""

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional, List, Dict, Tuple, Any
import math

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq

from ..data_sources.base import OptionChain, OptionContract, OptionFlow

logger = logging.getLogger(__name__)


@dataclass
class OptionsMetrics:
    """Container for analyzed options metrics."""
    symbol: str
    underlying_price: float
    expiration_date: date

    # ATM data
    atm_strike: float
    atm_call_price: float
    atm_put_price: float
    atm_straddle_price: float
    atm_iv: float

    # Volume and OI
    total_call_volume: int
    total_put_volume: int
    total_call_oi: int
    total_put_oi: int
    put_call_ratio: float
    put_call_oi_ratio: float

    # IV analysis
    iv_skew: float  # Put IV - Call IV
    iv_term_structure: Dict[date, float] = field(default_factory=dict)

    # Max pain
    max_pain_strike: float = 0

    # Key levels
    highest_call_oi_strike: float = 0
    highest_put_oi_strike: float = 0
    highest_volume_strike: float = 0

    # Greeks summary
    net_delta: float = 0
    net_gamma: float = 0

    # Flow sentiment
    flow_sentiment: str = "neutral"
    bullish_premium: float = 0
    bearish_premium: float = 0


class OptionsAnalyzer:
    """
    Comprehensive options chain analyzer.
    Extracts key metrics for expected move predictions.
    """

    def __init__(self, risk_free_rate: float = 0.05):
        """
        Initialize analyzer.

        Args:
            risk_free_rate: Annual risk-free rate for calculations
        """
        self.risk_free_rate = risk_free_rate

    def analyze_chain(
        self,
        chain: OptionChain,
        flows: Optional[List[OptionFlow]] = None
    ) -> OptionsMetrics:
        """
        Perform comprehensive analysis of an option chain.

        Args:
            chain: Option chain data
            flows: Optional list of option flows for sentiment

        Returns:
            OptionsMetrics with all analyzed data
        """
        underlying_price = chain.underlying_price
        atm_strike = chain.get_atm_strike()

        # Get ATM options
        atm_call = self._get_closest_strike_option(chain.calls, atm_strike)
        atm_put = self._get_closest_strike_option(chain.puts, atm_strike)

        # Calculate ATM prices and IV
        atm_call_price = self._get_mid_price(atm_call) if atm_call else 0
        atm_put_price = self._get_mid_price(atm_put) if atm_put else 0
        atm_straddle_price = atm_call_price + atm_put_price

        atm_iv = self._calculate_atm_iv(atm_call, atm_put, underlying_price, chain.expiration_date)

        # Volume and OI analysis
        total_call_volume = sum(c.volume for c in chain.calls)
        total_put_volume = sum(p.volume for p in chain.puts)
        total_call_oi = sum(c.open_interest for c in chain.calls)
        total_put_oi = sum(p.open_interest for p in chain.puts)

        put_call_ratio = total_put_volume / total_call_volume if total_call_volume > 0 else 0
        put_call_oi_ratio = total_put_oi / total_call_oi if total_call_oi > 0 else 0

        # IV skew
        iv_skew = self._calculate_iv_skew(chain, underlying_price)

        # Max pain calculation
        max_pain_strike = self._calculate_max_pain(chain)

        # Key levels
        highest_call_oi_strike = self._get_highest_oi_strike(chain.calls)
        highest_put_oi_strike = self._get_highest_oi_strike(chain.puts)
        highest_volume_strike = self._get_highest_volume_strike(chain.calls + chain.puts)

        # Greeks summary (if available)
        net_delta = self._calculate_net_delta(chain)
        net_gamma = self._calculate_net_gamma(chain)

        # Flow sentiment
        flow_sentiment, bullish_premium, bearish_premium = self._analyze_flow_sentiment(flows)

        return OptionsMetrics(
            symbol=chain.underlying_symbol,
            underlying_price=underlying_price,
            expiration_date=chain.expiration_date,
            atm_strike=atm_strike,
            atm_call_price=atm_call_price,
            atm_put_price=atm_put_price,
            atm_straddle_price=atm_straddle_price,
            atm_iv=atm_iv,
            total_call_volume=total_call_volume,
            total_put_volume=total_put_volume,
            total_call_oi=total_call_oi,
            total_put_oi=total_put_oi,
            put_call_ratio=put_call_ratio,
            put_call_oi_ratio=put_call_oi_ratio,
            iv_skew=iv_skew,
            max_pain_strike=max_pain_strike,
            highest_call_oi_strike=highest_call_oi_strike,
            highest_put_oi_strike=highest_put_oi_strike,
            highest_volume_strike=highest_volume_strike,
            net_delta=net_delta,
            net_gamma=net_gamma,
            flow_sentiment=flow_sentiment,
            bullish_premium=bullish_premium,
            bearish_premium=bearish_premium
        )

    def calculate_implied_volatility(
        self,
        option_price: float,
        underlying_price: float,
        strike: float,
        time_to_expiry: float,
        option_type: str,
        risk_free_rate: Optional[float] = None
    ) -> float:
        """
        Calculate implied volatility using Black-Scholes.

        Args:
            option_price: Current option price
            underlying_price: Current stock price
            strike: Strike price
            time_to_expiry: Time to expiry in years
            option_type: 'call' or 'put'
            risk_free_rate: Risk-free rate (uses instance default if None)

        Returns:
            Implied volatility as decimal
        """
        r = risk_free_rate or self.risk_free_rate

        if time_to_expiry <= 0 or option_price <= 0:
            return 0

        def objective(sigma):
            return self._black_scholes_price(
                underlying_price, strike, time_to_expiry, r, sigma, option_type
            ) - option_price

        try:
            iv = brentq(objective, 0.001, 5.0, maxiter=100)
            return iv
        except (ValueError, RuntimeError):
            return 0

    def _black_scholes_price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str
    ) -> float:
        """Calculate Black-Scholes option price."""
        if T <= 0:
            if option_type == 'call':
                return max(0, S - K)
            else:
                return max(0, K - S)

        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)

        if option_type == 'call':
            price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

        return price

    def _get_closest_strike_option(
        self,
        options: List[OptionContract],
        target_strike: float
    ) -> Optional[OptionContract]:
        """Get option closest to target strike."""
        if not options:
            return None
        return min(options, key=lambda x: abs(x.strike - target_strike))

    def _get_mid_price(self, option: Optional[OptionContract]) -> float:
        """Get mid price of an option."""
        if not option:
            return 0
        if option.bid > 0 and option.ask > 0:
            return (option.bid + option.ask) / 2
        return option.last_price

    def _calculate_atm_iv(
        self,
        atm_call: Optional[OptionContract],
        atm_put: Optional[OptionContract],
        underlying_price: float,
        expiration: date
    ) -> float:
        """Calculate ATM implied volatility."""
        time_to_expiry = max((expiration - date.today()).days / 365.0, 0.001)

        ivs = []

        if atm_call and atm_call.implied_volatility > 0:
            ivs.append(atm_call.implied_volatility)
        elif atm_call:
            mid_price = self._get_mid_price(atm_call)
            if mid_price > 0:
                iv = self.calculate_implied_volatility(
                    mid_price, underlying_price, atm_call.strike,
                    time_to_expiry, 'call'
                )
                if iv > 0:
                    ivs.append(iv)

        if atm_put and atm_put.implied_volatility > 0:
            ivs.append(atm_put.implied_volatility)
        elif atm_put:
            mid_price = self._get_mid_price(atm_put)
            if mid_price > 0:
                iv = self.calculate_implied_volatility(
                    mid_price, underlying_price, atm_put.strike,
                    time_to_expiry, 'put'
                )
                if iv > 0:
                    ivs.append(iv)

        return np.mean(ivs) if ivs else 0

    def _calculate_iv_skew(self, chain: OptionChain, underlying_price: float) -> float:
        """
        Calculate IV skew (put IV - call IV at same moneyness).
        Positive skew indicates more demand for puts.
        """
        # Get OTM options at ~5% from ATM
        target_call_strike = underlying_price * 1.05
        target_put_strike = underlying_price * 0.95

        otm_call = self._get_closest_strike_option(chain.calls, target_call_strike)
        otm_put = self._get_closest_strike_option(chain.puts, target_put_strike)

        if otm_call and otm_put:
            call_iv = otm_call.implied_volatility
            put_iv = otm_put.implied_volatility

            if call_iv > 0 and put_iv > 0:
                return put_iv - call_iv

        return 0

    def _calculate_max_pain(self, chain: OptionChain) -> float:
        """
        Calculate max pain strike (price where option holders lose most).
        """
        if not chain.calls and not chain.puts:
            return chain.underlying_price

        all_strikes = sorted(set(
            [c.strike for c in chain.calls] +
            [p.strike for p in chain.puts]
        ))

        if not all_strikes:
            return chain.underlying_price

        min_pain = float('inf')
        max_pain_strike = chain.underlying_price

        for strike in all_strikes:
            total_pain = 0

            # Calculate call holder losses
            for call in chain.calls:
                if strike > call.strike:
                    # Calls are ITM, holders profit
                    pass
                else:
                    # Calls are OTM, holders lose premium
                    total_pain += call.open_interest * self._get_mid_price(call) * 100

            # Calculate put holder losses
            for put in chain.puts:
                if strike < put.strike:
                    # Puts are ITM, holders profit
                    pass
                else:
                    # Puts are OTM, holders lose premium
                    total_pain += put.open_interest * self._get_mid_price(put) * 100

            if total_pain < min_pain:
                min_pain = total_pain
                max_pain_strike = strike

        return max_pain_strike

    def _get_highest_oi_strike(self, options: List[OptionContract]) -> float:
        """Get strike with highest open interest."""
        if not options:
            return 0
        return max(options, key=lambda x: x.open_interest).strike

    def _get_highest_volume_strike(self, options: List[OptionContract]) -> float:
        """Get strike with highest volume."""
        if not options:
            return 0
        return max(options, key=lambda x: x.volume).strike

    def _calculate_net_delta(self, chain: OptionChain) -> float:
        """Calculate net delta exposure."""
        net_delta = 0

        for call in chain.calls:
            if call.delta is not None:
                net_delta += call.open_interest * call.delta * 100

        for put in chain.puts:
            if put.delta is not None:
                net_delta += put.open_interest * put.delta * 100

        return net_delta

    def _calculate_net_gamma(self, chain: OptionChain) -> float:
        """Calculate net gamma exposure."""
        net_gamma = 0

        for call in chain.calls:
            if call.gamma is not None:
                net_gamma += call.open_interest * call.gamma * 100

        for put in chain.puts:
            if put.gamma is not None:
                net_gamma += put.open_interest * put.gamma * 100

        return net_gamma

    def _analyze_flow_sentiment(
        self,
        flows: Optional[List[OptionFlow]]
    ) -> Tuple[str, float, float]:
        """
        Analyze option flow for sentiment.

        Returns:
            Tuple of (sentiment, bullish_premium, bearish_premium)
        """
        if not flows:
            return "neutral", 0, 0

        bullish_premium = sum(
            f.premium for f in flows
            if f.sentiment.lower() == 'bullish' or
            (f.option_type == 'call' and f.trade_type in ['sweep', 'block'])
        )

        bearish_premium = sum(
            f.premium for f in flows
            if f.sentiment.lower() == 'bearish' or
            (f.option_type == 'put' and f.trade_type in ['sweep', 'block'])
        )

        total_premium = bullish_premium + bearish_premium

        if total_premium == 0:
            return "neutral", 0, 0

        bull_ratio = bullish_premium / total_premium

        if bull_ratio > 0.65:
            sentiment = "bullish"
        elif bull_ratio < 0.35:
            sentiment = "bearish"
        else:
            sentiment = "neutral"

        return sentiment, bullish_premium, bearish_premium

    def get_strike_analysis(
        self,
        chain: OptionChain,
        num_strikes: int = 10
    ) -> pd.DataFrame:
        """
        Get detailed analysis around ATM strikes.

        Returns DataFrame with strike-level analysis.
        """
        atm_strike = chain.get_atm_strike()
        all_strikes = sorted(set(
            [c.strike for c in chain.calls] +
            [p.strike for p in chain.puts]
        ))

        # Get strikes around ATM
        atm_idx = min(range(len(all_strikes)), key=lambda i: abs(all_strikes[i] - atm_strike))
        start_idx = max(0, atm_idx - num_strikes // 2)
        end_idx = min(len(all_strikes), start_idx + num_strikes)
        relevant_strikes = all_strikes[start_idx:end_idx]

        data = []
        for strike in relevant_strikes:
            call = next((c for c in chain.calls if c.strike == strike), None)
            put = next((p for p in chain.puts if p.strike == strike), None)

            row = {
                'strike': strike,
                'call_bid': call.bid if call else 0,
                'call_ask': call.ask if call else 0,
                'call_volume': call.volume if call else 0,
                'call_oi': call.open_interest if call else 0,
                'call_iv': call.implied_volatility if call else 0,
                'put_bid': put.bid if put else 0,
                'put_ask': put.ask if put else 0,
                'put_volume': put.volume if put else 0,
                'put_oi': put.open_interest if put else 0,
                'put_iv': put.implied_volatility if put else 0,
                'straddle_price': (
                    self._get_mid_price(call) + self._get_mid_price(put)
                    if call and put else 0
                ),
                'distance_pct': (strike - chain.underlying_price) / chain.underlying_price * 100
            }
            data.append(row)

        return pd.DataFrame(data)
