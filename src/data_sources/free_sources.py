"""
Free data sources for earnings and options data.
Uses Yahoo Finance, web scraping, and free APIs.
"""

import logging
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any
import time
import re

import requests
from bs4 import BeautifulSoup
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None

from .base import (
    DataSource, OptionsDataSource, OptionFlowSource,
    EarningsData, OptionContract, OptionChain, OptionFlow
)

logger = logging.getLogger(__name__)


class YahooFinanceSource(DataSource, OptionsDataSource):
    """
    Yahoo Finance data source for earnings and options data.
    This is a free source that doesn't require an API key.
    """

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key)
        if yf is None:
            raise ImportError("yfinance is required. Install with: pip install yfinance")
        self._session = requests.Session()
        self._session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    @property
    def name(self) -> str:
        return "Yahoo Finance"

    @property
    def requires_api_key(self) -> bool:
        return False

    def get_stock_info(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive stock information."""
        ticker = yf.Ticker(symbol)
        return ticker.info

    def get_current_price(self, symbol: str) -> float:
        """Get current stock price."""
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1d")
        if not hist.empty:
            return float(hist['Close'].iloc[-1])
        info = ticker.info
        return info.get('regularMarketPrice', info.get('currentPrice', 0))

    def get_earnings_calendar(
        self,
        start_date: date,
        end_date: date
    ) -> List[EarningsData]:
        """Fetch earnings calendar for date range from Yahoo Finance."""
        earnings_list = []

        # Yahoo Finance calendar endpoint
        url = "https://finance.yahoo.com/calendar/earnings"

        current_date = start_date
        while current_date <= end_date:
            try:
                params = {
                    'day': current_date.strftime('%Y-%m-%d')
                }
                response = self._session.get(url, params=params, timeout=10)
                response.raise_for_status()

                soup = BeautifulSoup(response.text, 'lxml')
                table = soup.find('table')

                if table:
                    rows = table.find_all('tr')[1:]  # Skip header
                    for row in rows:
                        cols = row.find_all('td')
                        if len(cols) >= 5:
                            symbol = cols[0].get_text(strip=True)
                            eps_est_text = cols[3].get_text(strip=True)
                            eps_est = self._parse_number(eps_est_text)

                            earnings_list.append(EarningsData(
                                symbol=symbol,
                                earnings_date=datetime.combine(current_date, datetime.min.time()),
                                eps_estimate=eps_est,
                                source="Yahoo Finance"
                            ))

            except Exception as e:
                logger.warning(f"Error fetching earnings for {current_date}: {e}")

            current_date += timedelta(days=1)
            time.sleep(0.5)  # Rate limiting

        return earnings_list

    def get_earnings_for_symbol(self, symbol: str) -> Optional[EarningsData]:
        """Get next earnings date for a specific symbol."""
        try:
            ticker = yf.Ticker(symbol)
            calendar = ticker.calendar

            if calendar is None or calendar.empty:
                # Try alternative method
                info = ticker.info
                if 'earningsDate' in info:
                    earnings_date = info['earningsDate']
                    if isinstance(earnings_date, list) and len(earnings_date) > 0:
                        earnings_date = earnings_date[0]
                    if isinstance(earnings_date, (int, float)):
                        earnings_date = datetime.fromtimestamp(earnings_date)

                    return EarningsData(
                        symbol=symbol,
                        earnings_date=earnings_date,
                        source="Yahoo Finance"
                    )
                return None

            # Parse calendar data
            earnings_date = None
            eps_estimate = None
            revenue_estimate = None

            if isinstance(calendar, pd.DataFrame):
                if 'Earnings Date' in calendar.index:
                    earnings_date = calendar.loc['Earnings Date'].iloc[0]
                if 'EPS Estimate' in calendar.index:
                    eps_estimate = calendar.loc['EPS Estimate'].iloc[0]
                if 'Revenue Estimate' in calendar.index:
                    revenue_estimate = calendar.loc['Revenue Estimate'].iloc[0]

            if earnings_date is None:
                return None

            if isinstance(earnings_date, str):
                earnings_date = pd.to_datetime(earnings_date)

            return EarningsData(
                symbol=symbol,
                earnings_date=earnings_date,
                eps_estimate=eps_estimate if pd.notna(eps_estimate) else None,
                revenue_estimate=revenue_estimate if pd.notna(revenue_estimate) else None,
                source="Yahoo Finance"
            )

        except Exception as e:
            logger.error(f"Error getting earnings for {symbol}: {e}")
            return None

    def get_option_expirations(self, symbol: str) -> List[date]:
        """Get available expiration dates for options."""
        try:
            ticker = yf.Ticker(symbol)
            expirations = ticker.options
            return [datetime.strptime(exp, '%Y-%m-%d').date() for exp in expirations]
        except Exception as e:
            logger.error(f"Error getting expirations for {symbol}: {e}")
            return []

    def get_option_chain(
        self,
        symbol: str,
        expiration: Optional[date] = None
    ) -> Optional[OptionChain]:
        """Fetch option chain for a symbol."""
        try:
            ticker = yf.Ticker(symbol)
            current_price = self.get_current_price(symbol)

            if expiration is None:
                expirations = ticker.options
                if not expirations:
                    return None
                expiration_str = expirations[0]
            else:
                expiration_str = expiration.strftime('%Y-%m-%d')

            opt_chain = ticker.option_chain(expiration_str)

            calls = self._parse_options_df(
                opt_chain.calls, symbol, expiration_str, 'call', current_price
            )
            puts = self._parse_options_df(
                opt_chain.puts, symbol, expiration_str, 'put', current_price
            )

            return OptionChain(
                underlying_symbol=symbol,
                underlying_price=current_price,
                expiration_date=datetime.strptime(expiration_str, '%Y-%m-%d').date(),
                calls=calls,
                puts=puts
            )

        except Exception as e:
            logger.error(f"Error getting option chain for {symbol}: {e}")
            return None

    def _parse_options_df(
        self,
        df: pd.DataFrame,
        underlying: str,
        expiration: str,
        option_type: str,
        current_price: float
    ) -> List[OptionContract]:
        """Parse options DataFrame into OptionContract objects."""
        contracts = []

        for _, row in df.iterrows():
            try:
                strike = float(row.get('strike', 0))
                contracts.append(OptionContract(
                    symbol=row.get('contractSymbol', ''),
                    underlying=underlying,
                    strike=strike,
                    expiration=datetime.strptime(expiration, '%Y-%m-%d').date(),
                    option_type=option_type,
                    bid=float(row.get('bid', 0)),
                    ask=float(row.get('ask', 0)),
                    last_price=float(row.get('lastPrice', 0)),
                    volume=int(row.get('volume', 0)) if pd.notna(row.get('volume')) else 0,
                    open_interest=int(row.get('openInterest', 0)) if pd.notna(row.get('openInterest')) else 0,
                    implied_volatility=float(row.get('impliedVolatility', 0)),
                    in_the_money=row.get('inTheMoney', False)
                ))
            except Exception as e:
                logger.debug(f"Error parsing option contract: {e}")
                continue

        return contracts

    def _parse_number(self, text: str) -> Optional[float]:
        """Parse a number from text, handling various formats."""
        if not text or text == '-':
            return None
        try:
            text = text.replace(',', '').replace('$', '')
            return float(text)
        except ValueError:
            return None


class EarningsWhispersScraper(DataSource):
    """
    Scraper for Earnings Whispers website.
    Provides earnings estimates and whisper numbers.
    """

    BASE_URL = "https://www.earningswhispers.com"

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key)
        self._session = requests.Session()
        self._session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        })

    @property
    def name(self) -> str:
        return "Earnings Whispers"

    @property
    def requires_api_key(self) -> bool:
        return False

    def get_earnings_calendar(
        self,
        start_date: date,
        end_date: date
    ) -> List[EarningsData]:
        """Fetch earnings calendar from Earnings Whispers."""
        earnings_list = []

        try:
            url = f"{self.BASE_URL}/calendar"
            response = self._session.get(url, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'lxml')

            # Find earnings entries
            earnings_divs = soup.find_all('div', class_=re.compile(r'ticker|stock'))

            for div in earnings_divs:
                try:
                    symbol_elem = div.find(['a', 'span'], class_=re.compile(r'symbol|ticker'))
                    if symbol_elem:
                        symbol = symbol_elem.get_text(strip=True).upper()

                        # Try to find date
                        date_elem = div.find(['span', 'div'], class_=re.compile(r'date'))
                        if date_elem:
                            date_text = date_elem.get_text(strip=True)
                            try:
                                earnings_date = self._parse_date(date_text)
                                if start_date <= earnings_date.date() <= end_date:
                                    earnings_list.append(EarningsData(
                                        symbol=symbol,
                                        earnings_date=earnings_date,
                                        source="Earnings Whispers"
                                    ))
                            except:
                                pass
                except Exception:
                    continue

        except Exception as e:
            logger.error(f"Error scraping Earnings Whispers calendar: {e}")

        return earnings_list

    def get_earnings_for_symbol(self, symbol: str) -> Optional[EarningsData]:
        """Get earnings data for a specific symbol from Earnings Whispers."""
        try:
            url = f"{self.BASE_URL}/stocks/{symbol.lower()}"
            response = self._session.get(url, timeout=15)

            if response.status_code == 404:
                return None

            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'lxml')

            # Parse earnings date
            earnings_date = None
            date_elem = soup.find(['div', 'span'], class_=re.compile(r'earnings.*date|report.*date'))
            if date_elem:
                date_text = date_elem.get_text(strip=True)
                earnings_date = self._parse_date(date_text)

            # Parse EPS estimate
            eps_estimate = None
            eps_elem = soup.find(['div', 'span'], class_=re.compile(r'estimate|eps'))
            if eps_elem:
                eps_text = eps_elem.get_text(strip=True)
                eps_match = re.search(r'[\$]?([-]?\d+\.?\d*)', eps_text)
                if eps_match:
                    eps_estimate = float(eps_match.group(1))

            # Determine time of day (BMO/AMC)
            time_of_day = "unknown"
            time_elem = soup.find(string=re.compile(r'before|after|BMO|AMC', re.I))
            if time_elem:
                if re.search(r'before|BMO', time_elem, re.I):
                    time_of_day = "BMO"
                elif re.search(r'after|AMC', time_elem, re.I):
                    time_of_day = "AMC"

            if earnings_date:
                return EarningsData(
                    symbol=symbol.upper(),
                    earnings_date=earnings_date,
                    eps_estimate=eps_estimate,
                    time_of_day=time_of_day,
                    source="Earnings Whispers"
                )

            return None

        except Exception as e:
            logger.error(f"Error getting earnings for {symbol} from Earnings Whispers: {e}")
            return None

    def get_whisper_number(self, symbol: str) -> Optional[float]:
        """Get the whisper number (unofficial EPS estimate) for a symbol."""
        try:
            url = f"{self.BASE_URL}/stocks/{symbol.lower()}"
            response = self._session.get(url, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'lxml')

            whisper_elem = soup.find(['div', 'span'], class_=re.compile(r'whisper'))
            if whisper_elem:
                whisper_text = whisper_elem.get_text(strip=True)
                whisper_match = re.search(r'[\$]?([-]?\d+\.?\d*)', whisper_text)
                if whisper_match:
                    return float(whisper_match.group(1))

            return None

        except Exception as e:
            logger.error(f"Error getting whisper number for {symbol}: {e}")
            return None

    def _parse_date(self, date_text: str) -> datetime:
        """Parse various date formats."""
        formats = [
            '%B %d, %Y',
            '%b %d, %Y',
            '%m/%d/%Y',
            '%Y-%m-%d',
            '%d-%b-%Y',
        ]

        for fmt in formats:
            try:
                return datetime.strptime(date_text.strip(), fmt)
            except ValueError:
                continue

        # Try relative dates
        today = datetime.now()
        if 'today' in date_text.lower():
            return today
        if 'tomorrow' in date_text.lower():
            return today + timedelta(days=1)

        raise ValueError(f"Could not parse date: {date_text}")


class FreeOptionsDataSource(OptionsDataSource, OptionFlowSource):
    """
    Combined free options data source using multiple free APIs and scraping.
    Aggregates data from Yahoo Finance and other free sources.
    """

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key)
        self._yahoo = YahooFinanceSource()
        self._session = requests.Session()
        self._session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    @property
    def name(self) -> str:
        return "Free Options Aggregator"

    def get_option_chain(
        self,
        symbol: str,
        expiration: Optional[date] = None
    ) -> Optional[OptionChain]:
        """Get option chain from best available free source."""
        # Primary: Yahoo Finance
        chain = self._yahoo.get_option_chain(symbol, expiration)
        if chain and chain.calls:
            return chain

        # Fallback: CBOE (if available)
        chain = self._try_cboe_chain(symbol, expiration)
        if chain:
            return chain

        return None

    def get_option_expirations(self, symbol: str) -> List[date]:
        """Get available expiration dates."""
        return self._yahoo.get_option_expirations(symbol)

    def get_unusual_activity(
        self,
        symbol: Optional[str] = None,
        min_premium: float = 100000
    ) -> List[OptionFlow]:
        """
        Get unusual options activity from free sources.
        Note: Free sources have limited flow data.
        """
        flows = []

        # Try to scrape from public sources
        try:
            flows.extend(self._scrape_barchart_unusual(symbol, min_premium))
        except Exception as e:
            logger.debug(f"Could not get Barchart data: {e}")

        return flows

    def _try_cboe_chain(
        self,
        symbol: str,
        expiration: Optional[date] = None
    ) -> Optional[OptionChain]:
        """Try to get option chain from CBOE."""
        try:
            url = f"https://www.cboe.com/delayed_quotes/{symbol.lower()}/quote_table"
            response = self._session.get(url, timeout=10)

            if response.status_code != 200:
                return None

            # CBOE uses JavaScript rendering, limited data available via scraping
            # This is a placeholder for potential future implementation
            return None

        except Exception as e:
            logger.debug(f"CBOE chain fetch failed: {e}")
            return None

    def _scrape_barchart_unusual(
        self,
        symbol: Optional[str] = None,
        min_premium: float = 100000
    ) -> List[OptionFlow]:
        """Scrape unusual options from Barchart."""
        flows = []

        try:
            url = "https://www.barchart.com/options/unusual-activity/stocks"
            response = self._session.get(url, timeout=15)

            if response.status_code != 200:
                return flows

            soup = BeautifulSoup(response.text, 'lxml')
            table = soup.find('table', class_=re.compile(r'data|unusual'))

            if not table:
                return flows

            rows = table.find_all('tr')[1:]  # Skip header

            for row in rows:
                try:
                    cols = row.find_all('td')
                    if len(cols) < 6:
                        continue

                    row_symbol = cols[0].get_text(strip=True).upper()

                    if symbol and row_symbol != symbol.upper():
                        continue

                    # Parse flow data
                    strike_text = cols[2].get_text(strip=True)
                    strike = float(re.sub(r'[^\d.]', '', strike_text)) if strike_text else 0

                    option_type = 'call' if 'call' in cols[1].get_text().lower() else 'put'

                    volume_text = cols[3].get_text(strip=True)
                    volume = int(re.sub(r'[^\d]', '', volume_text)) if volume_text else 0

                    # Estimate premium (simplified)
                    price_text = cols[4].get_text(strip=True) if len(cols) > 4 else '0'
                    price = float(re.sub(r'[^\d.]', '', price_text)) if price_text else 0
                    premium = volume * price * 100

                    if premium >= min_premium:
                        flows.append(OptionFlow(
                            symbol=f"{row_symbol}_{strike}_{option_type[0].upper()}",
                            underlying=row_symbol,
                            strike=strike,
                            expiration=date.today() + timedelta(days=30),  # Approximate
                            option_type=option_type,
                            sentiment='bullish' if option_type == 'call' else 'bearish',
                            volume=volume,
                            open_interest=0,
                            premium=premium,
                            trade_type='sweep',
                            timestamp=datetime.now(),
                            spot_price=0
                        ))

                except Exception as e:
                    logger.debug(f"Error parsing Barchart row: {e}")
                    continue

        except Exception as e:
            logger.warning(f"Barchart scraping failed: {e}")

        return flows


class NasdaqEarningsSource(DataSource):
    """
    Nasdaq earnings calendar data source.
    Free source without API key requirement.
    """

    BASE_URL = "https://api.nasdaq.com/api"

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key)
        self._session = requests.Session()
        self._session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json'
        })

    @property
    def name(self) -> str:
        return "Nasdaq"

    @property
    def requires_api_key(self) -> bool:
        return False

    def get_earnings_calendar(
        self,
        start_date: date,
        end_date: date
    ) -> List[EarningsData]:
        """Fetch earnings calendar from Nasdaq."""
        earnings_list = []

        try:
            url = f"{self.BASE_URL}/calendar/earnings"
            params = {
                'date': start_date.strftime('%Y-%m-%d')
            }

            response = self._session.get(url, params=params, timeout=15)
            response.raise_for_status()

            data = response.json()

            if data.get('data', {}).get('rows'):
                for row in data['data']['rows']:
                    try:
                        symbol = row.get('symbol', '')
                        date_str = row.get('date', '')

                        if not symbol or not date_str:
                            continue

                        earnings_date = datetime.strptime(date_str, '%m/%d/%Y')

                        eps_est = None
                        if row.get('eps'):
                            try:
                                eps_est = float(row['eps'].replace('$', ''))
                            except:
                                pass

                        time_of_day = row.get('time', 'unknown')
                        if 'before' in time_of_day.lower():
                            time_of_day = 'BMO'
                        elif 'after' in time_of_day.lower():
                            time_of_day = 'AMC'

                        earnings_list.append(EarningsData(
                            symbol=symbol,
                            earnings_date=earnings_date,
                            eps_estimate=eps_est,
                            time_of_day=time_of_day,
                            source="Nasdaq"
                        ))

                    except Exception as e:
                        logger.debug(f"Error parsing Nasdaq earnings row: {e}")
                        continue

        except Exception as e:
            logger.error(f"Error fetching Nasdaq earnings calendar: {e}")

        return earnings_list

    def get_earnings_for_symbol(self, symbol: str) -> Optional[EarningsData]:
        """Get earnings data for a specific symbol from Nasdaq."""
        try:
            url = f"{self.BASE_URL}/company/{symbol.lower()}/earnings-date"

            response = self._session.get(url, timeout=15)
            response.raise_for_status()

            data = response.json()

            if data.get('data'):
                earnings_data = data['data']
                date_str = earnings_data.get('earningsDate', '')

                if date_str:
                    earnings_date = datetime.strptime(date_str, '%m/%d/%Y')

                    return EarningsData(
                        symbol=symbol.upper(),
                        earnings_date=earnings_date,
                        eps_estimate=earnings_data.get('epsEstimate'),
                        source="Nasdaq"
                    )

            return None

        except Exception as e:
            logger.error(f"Error getting Nasdaq earnings for {symbol}: {e}")
            return None
