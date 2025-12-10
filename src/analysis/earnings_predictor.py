"""
Earnings predictor engine that combines multiple data sources
and analysis methods to predict expected stock moves after earnings.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any, Tuple
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json

import numpy as np
import pandas as pd

from ..data_sources.base import (
    DataSource, OptionsDataSource, OptionFlowSource,
    EarningsData, OptionChain, OptionFlow
)
from .options_analyzer import OptionsAnalyzer, OptionsMetrics
from .expected_move import ExpectedMoveCalculator, ExpectedMoveResult

logger = logging.getLogger(__name__)


@dataclass
class EarningsPrediction:
    """Complete earnings prediction result."""
    symbol: str
    current_price: float
    earnings_date: Optional[datetime]
    earnings_time: str  # BMO/AMC

    # Expected move
    expected_move: ExpectedMoveResult
    expected_move_percent: float
    expected_range: Tuple[float, float]

    # Direction prediction
    predicted_direction: str  # 'up', 'down', 'neutral'
    direction_confidence: float
    direction_signals: Dict[str, Any] = field(default_factory=dict)

    # Options analysis
    options_metrics: Optional[OptionsMetrics] = None
    iv_percentile: float = 0
    iv_rank: float = 0

    # Flow analysis
    option_flows: List[OptionFlow] = field(default_factory=list)
    flow_sentiment: str = "neutral"
    net_premium: float = 0

    # Historical context
    historical_moves: List[float] = field(default_factory=list)
    avg_historical_move: float = 0
    historical_beat_rate: float = 0
    last_earnings_move: float = 0

    # Earnings estimates
    eps_estimate: Optional[float] = None
    eps_whisper: Optional[float] = None
    revenue_estimate: Optional[float] = None

    # Data quality
    data_sources_used: List[str] = field(default_factory=list)
    overall_confidence: float = 0
    warnings: List[str] = field(default_factory=list)

    # Timestamp
    generated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'symbol': self.symbol,
            'current_price': round(self.current_price, 2),
            'earnings_date': self.earnings_date.isoformat() if self.earnings_date else None,
            'earnings_time': self.earnings_time,
            'expected_move_percent': round(self.expected_move_percent, 2),
            'expected_range': [round(r, 2) for r in self.expected_range],
            'predicted_direction': self.predicted_direction,
            'direction_confidence': round(self.direction_confidence * 100, 1),
            'iv_percentile': round(self.iv_percentile * 100, 1),
            'flow_sentiment': self.flow_sentiment,
            'avg_historical_move': round(self.avg_historical_move, 2),
            'eps_estimate': self.eps_estimate,
            'overall_confidence': round(self.overall_confidence * 100, 1),
            'data_sources': self.data_sources_used,
            'warnings': self.warnings,
            'generated_at': self.generated_at.isoformat()
        }

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"\n{'='*60}",
            f"EARNINGS PREDICTION: {self.symbol}",
            f"{'='*60}",
            f"Current Price: ${self.current_price:.2f}",
            f"Earnings Date: {self.earnings_date.strftime('%Y-%m-%d') if self.earnings_date else 'TBD'} ({self.earnings_time})",
            "",
            "EXPECTED MOVE:",
            f"  Range: ±{self.expected_move_percent:.2f}%",
            f"  Price Range: ${self.expected_range[0]:.2f} - ${self.expected_range[1]:.2f}",
            "",
            "DIRECTION PREDICTION:",
            f"  Direction: {self.predicted_direction.upper()}",
            f"  Confidence: {self.direction_confidence*100:.1f}%",
            "",
            "OPTIONS ANALYSIS:",
            f"  IV Percentile: {self.iv_percentile*100:.1f}%",
            f"  Flow Sentiment: {self.flow_sentiment}",
        ]

        if self.options_metrics:
            lines.extend([
                f"  Put/Call Ratio: {self.options_metrics.put_call_ratio:.2f}",
                f"  Max Pain: ${self.options_metrics.max_pain_strike:.2f}",
            ])

        if self.avg_historical_move > 0:
            lines.extend([
                "",
                "HISTORICAL CONTEXT:",
                f"  Avg Historical Move: ±{self.avg_historical_move:.2f}%",
                f"  Last Earnings Move: {self.last_earnings_move:+.2f}%",
                f"  Beat Rate: {self.historical_beat_rate*100:.0f}%",
            ])

        if self.eps_estimate:
            lines.extend([
                "",
                "ESTIMATES:",
                f"  EPS Estimate: ${self.eps_estimate:.2f}",
            ])
            if self.eps_whisper:
                lines.append(f"  EPS Whisper: ${self.eps_whisper:.2f}")

        lines.extend([
            "",
            f"Overall Confidence: {self.overall_confidence*100:.0f}%",
            f"Data Sources: {', '.join(self.data_sources_used)}",
        ])

        if self.warnings:
            lines.extend([
                "",
                "WARNINGS:",
                *[f"  - {w}" for w in self.warnings]
            ])

        lines.append(f"{'='*60}\n")

        return "\n".join(lines)


class EarningsPredictor:
    """
    Main earnings prediction engine.

    Combines data from multiple sources to generate comprehensive
    predictions for stock moves after earnings announcements.
    """

    def __init__(
        self,
        earnings_sources: Optional[List[DataSource]] = None,
        options_sources: Optional[List[OptionsDataSource]] = None,
        flow_sources: Optional[List[OptionFlowSource]] = None,
        use_free_sources: bool = True,
        use_premium_sources: bool = False
    ):
        """
        Initialize the earnings predictor.

        Args:
            earnings_sources: List of earnings data sources
            options_sources: List of options data sources
            flow_sources: List of options flow sources
            use_free_sources: Whether to use free data sources
            use_premium_sources: Whether to use premium data sources
        """
        self.earnings_sources = earnings_sources or []
        self.options_sources = options_sources or []
        self.flow_sources = flow_sources or []

        self.analyzer = OptionsAnalyzer()
        self.move_calculator = ExpectedMoveCalculator(self.analyzer)

        self._historical_cache: Dict[str, List[Dict]] = {}
        self._executor = ThreadPoolExecutor(max_workers=5)

        if use_free_sources:
            self._setup_free_sources()
        if use_premium_sources:
            self._setup_premium_sources()

    def _setup_free_sources(self):
        """Initialize free data sources."""
        try:
            from ..data_sources.free_sources import (
                YahooFinanceSource,
                EarningsWhispersScraper,
                FreeOptionsDataSource,
                NasdaqEarningsSource
            )

            # Add Yahoo Finance (both earnings and options)
            yahoo = YahooFinanceSource()
            if yahoo not in self.earnings_sources:
                self.earnings_sources.append(yahoo)
            if yahoo not in self.options_sources:
                self.options_sources.append(yahoo)

            # Add free options aggregator
            free_opts = FreeOptionsDataSource()
            if free_opts not in self.options_sources:
                self.options_sources.append(free_opts)
            if free_opts not in self.flow_sources:
                self.flow_sources.append(free_opts)

            # Add Nasdaq earnings
            nasdaq = NasdaqEarningsSource()
            if nasdaq not in self.earnings_sources:
                self.earnings_sources.append(nasdaq)

            # Add Earnings Whispers scraper
            ew = EarningsWhispersScraper()
            if ew not in self.earnings_sources:
                self.earnings_sources.append(ew)

            logger.info("Free data sources initialized")

        except ImportError as e:
            logger.warning(f"Could not initialize some free sources: {e}")

    def _setup_premium_sources(self):
        """Initialize premium data sources (requires API keys)."""
        import os

        try:
            from ..data_sources.premium_sources import (
                UnusualWhalesSource,
                TradingEconomicsSource,
                PolygonSource,
                AlphaVantageSource
            )

            # Unusual Whales
            uw_key = os.getenv('UNUSUAL_WHALES_API_KEY')
            if uw_key:
                uw = UnusualWhalesSource(uw_key)
                self.options_sources.append(uw)
                self.flow_sources.append(uw)

            # Trading Economics
            te_key = os.getenv('TRADING_ECONOMICS_API_KEY')
            if te_key:
                te = TradingEconomicsSource(te_key)
                self.earnings_sources.append(te)

            # Polygon
            polygon_key = os.getenv('POLYGON_API_KEY')
            if polygon_key:
                polygon = PolygonSource(polygon_key)
                self.options_sources.append(polygon)

            # Alpha Vantage
            av_key = os.getenv('ALPHA_VANTAGE_API_KEY')
            if av_key:
                av = AlphaVantageSource(av_key)
                self.earnings_sources.append(av)

            logger.info("Premium data sources initialized")

        except ImportError as e:
            logger.warning(f"Could not initialize premium sources: {e}")

    def predict(
        self,
        symbol: str,
        include_flows: bool = True,
        include_historical: bool = True
    ) -> EarningsPrediction:
        """
        Generate a complete earnings prediction for a symbol.

        Args:
            symbol: Stock symbol
            include_flows: Whether to include options flow analysis
            include_historical: Whether to include historical analysis

        Returns:
            EarningsPrediction with all analysis
        """
        symbol = symbol.upper()
        warnings = []
        data_sources_used = []

        # Get earnings date
        earnings_data = self._get_earnings_data(symbol)
        if earnings_data:
            data_sources_used.append(earnings_data.source)
        else:
            warnings.append("Could not determine earnings date")

        # Get current price and option chain
        chain = self._get_option_chain(symbol, earnings_data)
        if chain:
            current_price = chain.underlying_price
            data_sources_used.extend([s.name for s in self.options_sources if s.name not in data_sources_used])
        else:
            # Fallback to just getting price
            current_price = self._get_current_price(symbol)
            warnings.append("Could not get complete option chain")

        if current_price == 0:
            raise ValueError(f"Could not get price for {symbol}")

        # Calculate expected move
        if chain:
            expected_move = self.move_calculator.calculate_expected_move(
                chain,
                earnings_date=earnings_data.earnings_date if earnings_data else None
            )
            options_metrics = self.analyzer.analyze_chain(chain)
        else:
            expected_move = ExpectedMoveResult(
                symbol=symbol,
                underlying_price=current_price,
                expiration_date=date.today() + timedelta(days=7)
            )
            options_metrics = None

        # Get option flows
        flows = []
        if include_flows:
            flows = self._get_option_flows(symbol)
            if flows:
                data_sources_used.extend([s.name for s in self.flow_sources if s.name not in data_sources_used])

        # Get historical data
        historical_moves = []
        avg_historical = 0
        beat_rate = 0
        last_move = 0
        if include_historical:
            hist_data = self._get_historical_earnings(symbol)
            if hist_data:
                historical_moves = [h.get('move_pct', 0) for h in hist_data]
                avg_historical = np.mean(np.abs(historical_moves)) if historical_moves else 0
                beat_rate = sum(1 for h in hist_data if h.get('beat', False)) / len(hist_data) if hist_data else 0
                last_move = historical_moves[0] if historical_moves else 0

        # Predict direction
        direction, direction_confidence, direction_signals = self._predict_direction(
            options_metrics, flows, beat_rate, historical_moves
        )

        # Calculate IV metrics
        iv_percentile = self._calculate_iv_percentile(symbol, expected_move.implied_volatility)
        iv_rank = self._calculate_iv_rank(symbol, expected_move.implied_volatility)

        # Calculate flow sentiment
        flow_sentiment = "neutral"
        net_premium = 0
        if flows:
            bullish = sum(f.premium for f in flows if f.sentiment == 'bullish')
            bearish = sum(f.premium for f in flows if f.sentiment == 'bearish')
            net_premium = bullish - bearish

            if bullish > bearish * 1.5:
                flow_sentiment = "bullish"
            elif bearish > bullish * 1.5:
                flow_sentiment = "bearish"

        # Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(
            expected_move, options_metrics, len(flows), len(historical_moves)
        )

        # Build prediction
        return EarningsPrediction(
            symbol=symbol,
            current_price=current_price,
            earnings_date=earnings_data.earnings_date if earnings_data else None,
            earnings_time=earnings_data.time_of_day if earnings_data else "unknown",
            expected_move=expected_move,
            expected_move_percent=expected_move.expected_move_percent,
            expected_range=(expected_move.lower_bound, expected_move.upper_bound),
            predicted_direction=direction,
            direction_confidence=direction_confidence,
            direction_signals=direction_signals,
            options_metrics=options_metrics,
            iv_percentile=iv_percentile,
            iv_rank=iv_rank,
            option_flows=flows,
            flow_sentiment=flow_sentiment,
            net_premium=net_premium,
            historical_moves=historical_moves,
            avg_historical_move=avg_historical,
            historical_beat_rate=beat_rate,
            last_earnings_move=last_move,
            eps_estimate=earnings_data.eps_estimate if earnings_data else None,
            eps_whisper=None,  # Would need Earnings Whispers scraper
            revenue_estimate=earnings_data.revenue_estimate if earnings_data else None,
            data_sources_used=list(set(data_sources_used)),
            overall_confidence=overall_confidence,
            warnings=warnings
        )

    def predict_batch(
        self,
        symbols: List[str],
        parallel: bool = True
    ) -> List[EarningsPrediction]:
        """
        Generate predictions for multiple symbols.

        Args:
            symbols: List of stock symbols
            parallel: Whether to process in parallel

        Returns:
            List of EarningsPrediction objects
        """
        if parallel:
            with ThreadPoolExecutor(max_workers=5) as executor:
                results = list(executor.map(self._safe_predict, symbols))
        else:
            results = [self._safe_predict(s) for s in symbols]

        return [r for r in results if r is not None]

    def _safe_predict(self, symbol: str) -> Optional[EarningsPrediction]:
        """Safely predict with error handling."""
        try:
            return self.predict(symbol)
        except Exception as e:
            logger.error(f"Error predicting {symbol}: {e}")
            return None

    def _get_earnings_data(self, symbol: str) -> Optional[EarningsData]:
        """Get earnings data from available sources."""
        for source in self.earnings_sources:
            try:
                data = source.get_earnings_for_symbol(symbol)
                if data:
                    return data
            except Exception as e:
                logger.debug(f"Error getting earnings from {source.name}: {e}")
                continue
        return None

    def _get_option_chain(
        self,
        symbol: str,
        earnings_data: Optional[EarningsData] = None
    ) -> Optional[OptionChain]:
        """Get option chain, preferring expiration after earnings."""
        for source in self.options_sources:
            try:
                # Get expiration after earnings
                expiration = None
                if earnings_data:
                    expiration = source.get_nearest_expiration_after_earnings(
                        symbol, earnings_data.earnings_date.date()
                    )

                chain = source.get_option_chain(symbol, expiration)
                if chain and chain.calls:
                    return chain

            except Exception as e:
                logger.debug(f"Error getting chain from {source.name}: {e}")
                continue

        return None

    def _get_current_price(self, symbol: str) -> float:
        """Get current stock price."""
        for source in self.options_sources:
            if hasattr(source, 'get_current_price'):
                try:
                    price = source.get_current_price(symbol)
                    if price > 0:
                        return price
                except:
                    continue

        # Fallback to yfinance
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d")
            if not hist.empty:
                return float(hist['Close'].iloc[-1])
        except:
            pass

        return 0

    def _get_option_flows(self, symbol: str) -> List[OptionFlow]:
        """Get option flows from available sources."""
        all_flows = []

        for source in self.flow_sources:
            try:
                flows = source.get_unusual_activity(symbol)
                all_flows.extend(flows)
            except Exception as e:
                logger.debug(f"Error getting flows from {source.name}: {e}")
                continue

        return all_flows

    def _get_historical_earnings(self, symbol: str) -> List[Dict]:
        """Get historical earnings data."""
        if symbol in self._historical_cache:
            return self._historical_cache[symbol]

        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)

            # Get earnings history
            earnings_hist = ticker.earnings_history
            if earnings_hist is not None and not earnings_hist.empty:
                results = []
                for idx, row in earnings_hist.iterrows():
                    eps_actual = row.get('epsActual', 0)
                    eps_estimate = row.get('epsEstimate', 0)

                    if eps_estimate != 0:
                        surprise = ((eps_actual - eps_estimate) / abs(eps_estimate)) * 100
                    else:
                        surprise = 0

                    results.append({
                        'date': idx,
                        'eps_actual': eps_actual,
                        'eps_estimate': eps_estimate,
                        'surprise_pct': surprise,
                        'beat': eps_actual > eps_estimate,
                        'move_pct': surprise * 2  # Rough approximation
                    })

                self._historical_cache[symbol] = results
                return results

        except Exception as e:
            logger.debug(f"Error getting historical earnings: {e}")

        return []

    def _predict_direction(
        self,
        metrics: Optional[OptionsMetrics],
        flows: List[OptionFlow],
        beat_rate: float,
        historical_moves: List[float]
    ) -> Tuple[str, float, Dict]:
        """
        Predict the direction of the earnings move.

        Returns:
            Tuple of (direction, confidence, signals)
        """
        signals = {}
        scores = {'up': 0, 'down': 0}

        # Signal 1: Put/Call ratio
        if metrics:
            pcr = metrics.put_call_ratio
            if pcr < 0.7:
                scores['up'] += 1
                signals['put_call_ratio'] = {'value': pcr, 'signal': 'bullish'}
            elif pcr > 1.3:
                scores['down'] += 1
                signals['put_call_ratio'] = {'value': pcr, 'signal': 'bearish'}
            else:
                signals['put_call_ratio'] = {'value': pcr, 'signal': 'neutral'}

        # Signal 2: IV Skew
        if metrics and metrics.iv_skew != 0:
            if metrics.iv_skew > 0.05:  # Puts more expensive
                scores['down'] += 0.5
                signals['iv_skew'] = {'value': metrics.iv_skew, 'signal': 'bearish'}
            elif metrics.iv_skew < -0.05:  # Calls more expensive
                scores['up'] += 0.5
                signals['iv_skew'] = {'value': metrics.iv_skew, 'signal': 'bullish'}
            else:
                signals['iv_skew'] = {'value': metrics.iv_skew, 'signal': 'neutral'}

        # Signal 3: Flow sentiment
        if flows:
            bullish_count = sum(1 for f in flows if f.sentiment == 'bullish')
            bearish_count = sum(1 for f in flows if f.sentiment == 'bearish')

            if bullish_count > bearish_count * 1.5:
                scores['up'] += 1.5
                signals['flow'] = {'bullish': bullish_count, 'bearish': bearish_count, 'signal': 'bullish'}
            elif bearish_count > bullish_count * 1.5:
                scores['down'] += 1.5
                signals['flow'] = {'bullish': bullish_count, 'bearish': bearish_count, 'signal': 'bearish'}
            else:
                signals['flow'] = {'bullish': bullish_count, 'bearish': bearish_count, 'signal': 'neutral'}

        # Signal 4: Historical beat rate
        if beat_rate > 0:
            if beat_rate > 0.7:
                scores['up'] += 1
                signals['beat_rate'] = {'value': beat_rate, 'signal': 'bullish'}
            elif beat_rate < 0.3:
                scores['down'] += 1
                signals['beat_rate'] = {'value': beat_rate, 'signal': 'bearish'}
            else:
                signals['beat_rate'] = {'value': beat_rate, 'signal': 'neutral'}

        # Signal 5: Recent trend
        if len(historical_moves) >= 3:
            recent_avg = np.mean(historical_moves[:3])
            if recent_avg > 2:
                scores['up'] += 0.5
                signals['recent_trend'] = {'value': recent_avg, 'signal': 'bullish'}
            elif recent_avg < -2:
                scores['down'] += 0.5
                signals['recent_trend'] = {'value': recent_avg, 'signal': 'bearish'}

        # Determine direction and confidence
        total_score = scores['up'] + scores['down']

        if total_score == 0:
            return 'neutral', 0.5, signals

        if scores['up'] > scores['down']:
            direction = 'up'
            confidence = min(0.5 + (scores['up'] - scores['down']) / (total_score * 2), 0.85)
        elif scores['down'] > scores['up']:
            direction = 'down'
            confidence = min(0.5 + (scores['down'] - scores['up']) / (total_score * 2), 0.85)
        else:
            direction = 'neutral'
            confidence = 0.5

        return direction, confidence, signals

    def _calculate_iv_percentile(self, symbol: str, current_iv: float) -> float:
        """Calculate IV percentile (where current IV ranks historically)."""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1y")

            if hist.empty:
                return 0.5

            # Calculate historical volatility
            returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
            rolling_vol = returns.rolling(window=20).std() * np.sqrt(252)

            if rolling_vol.empty:
                return 0.5

            percentile = (rolling_vol < current_iv).sum() / len(rolling_vol)
            return percentile

        except Exception as e:
            logger.debug(f"Error calculating IV percentile: {e}")
            return 0.5

    def _calculate_iv_rank(self, symbol: str, current_iv: float) -> float:
        """Calculate IV rank ((current - min) / (max - min))."""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1y")

            if hist.empty:
                return 0.5

            returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
            rolling_vol = returns.rolling(window=20).std() * np.sqrt(252)

            if rolling_vol.empty:
                return 0.5

            min_vol = rolling_vol.min()
            max_vol = rolling_vol.max()

            if max_vol == min_vol:
                return 0.5

            rank = (current_iv - min_vol) / (max_vol - min_vol)
            return max(0, min(1, rank))

        except Exception as e:
            logger.debug(f"Error calculating IV rank: {e}")
            return 0.5

    def _calculate_overall_confidence(
        self,
        expected_move: ExpectedMoveResult,
        metrics: Optional[OptionsMetrics],
        num_flows: int,
        num_historical: int
    ) -> float:
        """Calculate overall confidence in the prediction."""
        confidence = 0.3  # Base confidence

        # Data quality
        if expected_move.data_quality == 'good':
            confidence += 0.2
        elif expected_move.data_quality == 'moderate':
            confidence += 0.1

        # Expected move confidence
        confidence += expected_move.confidence_score * 0.2

        # Have metrics
        if metrics:
            confidence += 0.1

        # Have flows
        if num_flows > 5:
            confidence += 0.1
        elif num_flows > 0:
            confidence += 0.05

        # Have historical data
        if num_historical > 4:
            confidence += 0.1
        elif num_historical > 0:
            confidence += 0.05

        return min(confidence, 0.95)

    def get_earnings_calendar(
        self,
        start_date: date,
        end_date: date
    ) -> pd.DataFrame:
        """
        Get earnings calendar for date range.

        Returns DataFrame with upcoming earnings.
        """
        all_earnings = []

        for source in self.earnings_sources:
            try:
                earnings = source.get_earnings_calendar(start_date, end_date)
                all_earnings.extend(earnings)
            except Exception as e:
                logger.debug(f"Error getting calendar from {source.name}: {e}")
                continue

        if not all_earnings:
            return pd.DataFrame()

        # Deduplicate by symbol and date
        seen = set()
        unique_earnings = []
        for e in all_earnings:
            key = (e.symbol, e.earnings_date.date())
            if key not in seen:
                seen.add(key)
                unique_earnings.append({
                    'symbol': e.symbol,
                    'earnings_date': e.earnings_date,
                    'eps_estimate': e.eps_estimate,
                    'time': e.time_of_day,
                    'source': e.source
                })

        df = pd.DataFrame(unique_earnings)
        df = df.sort_values('earnings_date')

        return df
