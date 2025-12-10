"""
Base classes for data sources.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional, List, Dict, Any
import pandas as pd


@dataclass
class EarningsData:
    """Container for earnings information."""
    symbol: str
    earnings_date: datetime
    eps_estimate: Optional[float] = None
    eps_actual: Optional[float] = None
    revenue_estimate: Optional[float] = None
    revenue_actual: Optional[float] = None
    surprise_percent: Optional[float] = None
    time_of_day: str = "unknown"  # BMO (Before Market Open), AMC (After Market Close)
    source: str = "unknown"


@dataclass
class OptionContract:
    """Container for a single option contract."""
    symbol: str
    underlying: str
    strike: float
    expiration: date
    option_type: str  # 'call' or 'put'
    bid: float
    ask: float
    last_price: float
    volume: int
    open_interest: int
    implied_volatility: float
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    in_the_money: bool = False


@dataclass
class OptionChain:
    """Container for option chain data."""
    underlying_symbol: str
    underlying_price: float
    expiration_date: date
    calls: List[OptionContract] = field(default_factory=list)
    puts: List[OptionContract] = field(default_factory=list)
    fetch_time: datetime = field(default_factory=datetime.now)

    def get_atm_strike(self) -> float:
        """Get the at-the-money strike price."""
        if not self.calls:
            return self.underlying_price
        strikes = sorted(set(c.strike for c in self.calls))
        return min(strikes, key=lambda x: abs(x - self.underlying_price))

    def get_atm_straddle_price(self) -> Optional[float]:
        """Get the at-the-money straddle price (call + put mid prices)."""
        atm_strike = self.get_atm_strike()

        atm_call = next((c for c in self.calls if c.strike == atm_strike), None)
        atm_put = next((p for p in self.puts if p.strike == atm_strike), None)

        if atm_call and atm_put:
            call_mid = (atm_call.bid + atm_call.ask) / 2
            put_mid = (atm_put.bid + atm_put.ask) / 2
            return call_mid + put_mid
        return None


@dataclass
class OptionFlow:
    """Container for unusual options activity/flow."""
    symbol: str
    underlying: str
    strike: float
    expiration: date
    option_type: str
    sentiment: str  # 'bullish', 'bearish', 'neutral'
    volume: int
    open_interest: int
    premium: float
    trade_type: str  # 'sweep', 'block', 'split'
    timestamp: datetime
    spot_price: float


class DataSource(ABC):
    """Abstract base class for all data sources."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self._rate_limit_remaining = None
        self._last_request_time = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the data source."""
        pass

    @property
    @abstractmethod
    def requires_api_key(self) -> bool:
        """Whether this source requires an API key."""
        pass

    @abstractmethod
    def get_earnings_calendar(
        self,
        start_date: date,
        end_date: date
    ) -> List[EarningsData]:
        """Fetch earnings calendar for date range."""
        pass

    @abstractmethod
    def get_earnings_for_symbol(
        self,
        symbol: str
    ) -> Optional[EarningsData]:
        """Get next earnings date for a symbol."""
        pass


class OptionsDataSource(ABC):
    """Abstract base class for options data sources."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the data source."""
        pass

    @abstractmethod
    def get_option_chain(
        self,
        symbol: str,
        expiration: Optional[date] = None
    ) -> Optional[OptionChain]:
        """Fetch option chain for a symbol."""
        pass

    @abstractmethod
    def get_option_expirations(
        self,
        symbol: str
    ) -> List[date]:
        """Get available expiration dates for a symbol."""
        pass

    def get_nearest_expiration_after_earnings(
        self,
        symbol: str,
        earnings_date: date
    ) -> Optional[date]:
        """Get the nearest option expiration after earnings date."""
        expirations = self.get_option_expirations(symbol)
        future_expirations = [exp for exp in expirations if exp >= earnings_date]
        return min(future_expirations) if future_expirations else None


class OptionFlowSource(ABC):
    """Abstract base class for options flow data sources."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the data source."""
        pass

    @abstractmethod
    def get_unusual_activity(
        self,
        symbol: Optional[str] = None,
        min_premium: float = 100000
    ) -> List[OptionFlow]:
        """Fetch unusual options activity."""
        pass
