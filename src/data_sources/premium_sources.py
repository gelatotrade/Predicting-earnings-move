"""
Premium/Paid data sources for earnings and options data.
Requires API keys for access.
"""

import logging
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any
import time
import asyncio

import requests
import aiohttp

from .base import (
    DataSource, OptionsDataSource, OptionFlowSource,
    EarningsData, OptionContract, OptionChain, OptionFlow
)

logger = logging.getLogger(__name__)


class UnusualWhalesSource(OptionsDataSource, OptionFlowSource):
    """
    Unusual Whales API for options flow and unusual activity data.
    Requires API key subscription.
    """

    BASE_URL = "https://api.unusualwhales.com"
    API_VERSION = "v2"

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key)
        if not api_key:
            raise ValueError("Unusual Whales requires an API key")
        self._session = requests.Session()
        self._session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Accept': 'application/json'
        })

    @property
    def name(self) -> str:
        return "Unusual Whales"

    def get_option_chain(
        self,
        symbol: str,
        expiration: Optional[date] = None
    ) -> Optional[OptionChain]:
        """Fetch option chain from Unusual Whales."""
        try:
            url = f"{self.BASE_URL}/{self.API_VERSION}/stock/{symbol}/option-chain"
            params = {}
            if expiration:
                params['expiration'] = expiration.strftime('%Y-%m-%d')

            response = self._session.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            if not data.get('data'):
                return None

            # Get current price
            spot_price = float(data['data'].get('underlying_price', 0))

            calls = []
            puts = []

            for contract in data['data'].get('contracts', []):
                option = self._parse_contract(contract, symbol)
                if option:
                    if option.option_type == 'call':
                        calls.append(option)
                    else:
                        puts.append(option)

            exp_date = expiration or (date.today() + timedelta(days=7))

            return OptionChain(
                underlying_symbol=symbol,
                underlying_price=spot_price,
                expiration_date=exp_date,
                calls=calls,
                puts=puts
            )

        except Exception as e:
            logger.error(f"Error fetching Unusual Whales chain for {symbol}: {e}")
            return None

    def get_option_expirations(self, symbol: str) -> List[date]:
        """Get available expiration dates."""
        try:
            url = f"{self.BASE_URL}/{self.API_VERSION}/stock/{symbol}/expirations"
            response = self._session.get(url, timeout=15)
            response.raise_for_status()

            data = response.json()
            expirations = []

            for exp_str in data.get('data', {}).get('expirations', []):
                try:
                    expirations.append(datetime.strptime(exp_str, '%Y-%m-%d').date())
                except:
                    continue

            return sorted(expirations)

        except Exception as e:
            logger.error(f"Error getting Unusual Whales expirations for {symbol}: {e}")
            return []

    def get_unusual_activity(
        self,
        symbol: Optional[str] = None,
        min_premium: float = 100000
    ) -> List[OptionFlow]:
        """Fetch unusual options activity from Unusual Whales."""
        try:
            url = f"{self.BASE_URL}/{self.API_VERSION}/flow/unusual"
            params = {
                'min_premium': int(min_premium)
            }
            if symbol:
                params['ticker'] = symbol.upper()

            response = self._session.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()
            flows = []

            for flow_data in data.get('data', []):
                try:
                    flows.append(OptionFlow(
                        symbol=flow_data.get('option_symbol', ''),
                        underlying=flow_data.get('ticker', ''),
                        strike=float(flow_data.get('strike', 0)),
                        expiration=datetime.strptime(
                            flow_data.get('expiration', ''),
                            '%Y-%m-%d'
                        ).date(),
                        option_type=flow_data.get('put_call', 'call').lower(),
                        sentiment=flow_data.get('sentiment', 'neutral').lower(),
                        volume=int(flow_data.get('volume', 0)),
                        open_interest=int(flow_data.get('open_interest', 0)),
                        premium=float(flow_data.get('premium', 0)),
                        trade_type=flow_data.get('trade_type', 'unknown'),
                        timestamp=datetime.fromisoformat(
                            flow_data.get('timestamp', datetime.now().isoformat())
                        ),
                        spot_price=float(flow_data.get('underlying_price', 0))
                    ))
                except Exception as e:
                    logger.debug(f"Error parsing flow: {e}")
                    continue

            return flows

        except Exception as e:
            logger.error(f"Error fetching Unusual Whales flow: {e}")
            return []

    def get_flow_alerts(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get real-time flow alerts."""
        try:
            url = f"{self.BASE_URL}/{self.API_VERSION}/flow/alerts"
            params = {}
            if symbol:
                params['ticker'] = symbol.upper()

            response = self._session.get(url, params=params, timeout=15)
            response.raise_for_status()

            return response.json().get('data', [])

        except Exception as e:
            logger.error(f"Error fetching flow alerts: {e}")
            return []

    def _parse_contract(
        self,
        contract_data: Dict,
        underlying: str
    ) -> Optional[OptionContract]:
        """Parse contract data into OptionContract."""
        try:
            return OptionContract(
                symbol=contract_data.get('option_symbol', ''),
                underlying=underlying,
                strike=float(contract_data.get('strike', 0)),
                expiration=datetime.strptime(
                    contract_data.get('expiration', ''),
                    '%Y-%m-%d'
                ).date(),
                option_type=contract_data.get('put_call', 'call').lower(),
                bid=float(contract_data.get('bid', 0)),
                ask=float(contract_data.get('ask', 0)),
                last_price=float(contract_data.get('last', 0)),
                volume=int(contract_data.get('volume', 0)),
                open_interest=int(contract_data.get('open_interest', 0)),
                implied_volatility=float(contract_data.get('iv', 0)),
                delta=contract_data.get('delta'),
                gamma=contract_data.get('gamma'),
                theta=contract_data.get('theta'),
                vega=contract_data.get('vega'),
                in_the_money=contract_data.get('in_the_money', False)
            )
        except Exception as e:
            logger.debug(f"Error parsing contract: {e}")
            return None


class TradingEconomicsSource(DataSource):
    """
    Trading Economics API for economic data and earnings.
    Requires API key subscription.
    """

    BASE_URL = "https://api.tradingeconomics.com"

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key)
        if not api_key:
            raise ValueError("Trading Economics requires an API key")
        self.api_key = api_key
        self._session = requests.Session()

    @property
    def name(self) -> str:
        return "Trading Economics"

    @property
    def requires_api_key(self) -> bool:
        return True

    def get_earnings_calendar(
        self,
        start_date: date,
        end_date: date
    ) -> List[EarningsData]:
        """Fetch earnings calendar from Trading Economics."""
        try:
            url = f"{self.BASE_URL}/earnings"
            params = {
                'c': self.api_key,
                'd1': start_date.strftime('%Y-%m-%d'),
                'd2': end_date.strftime('%Y-%m-%d'),
                'country': 'united states'
            }

            response = self._session.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()
            earnings_list = []

            for item in data:
                try:
                    symbol = item.get('Symbol', '')
                    if not symbol:
                        continue

                    date_str = item.get('Date', '')
                    if date_str:
                        earnings_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    else:
                        continue

                    earnings_list.append(EarningsData(
                        symbol=symbol,
                        earnings_date=earnings_date,
                        eps_estimate=item.get('Forecast'),
                        eps_actual=item.get('Actual'),
                        revenue_estimate=item.get('Revenue', {}).get('Forecast') if isinstance(item.get('Revenue'), dict) else None,
                        source="Trading Economics"
                    ))

                except Exception as e:
                    logger.debug(f"Error parsing earnings item: {e}")
                    continue

            return earnings_list

        except Exception as e:
            logger.error(f"Error fetching Trading Economics earnings: {e}")
            return []

    def get_earnings_for_symbol(self, symbol: str) -> Optional[EarningsData]:
        """Get earnings data for a specific symbol."""
        try:
            url = f"{self.BASE_URL}/earnings/symbol/{symbol}"
            params = {'c': self.api_key}

            response = self._session.get(url, params=params, timeout=15)
            response.raise_for_status()

            data = response.json()
            if data and len(data) > 0:
                item = data[0]
                date_str = item.get('Date', '')
                earnings_date = datetime.fromisoformat(date_str.replace('Z', '+00:00')) if date_str else None

                if earnings_date:
                    return EarningsData(
                        symbol=symbol,
                        earnings_date=earnings_date,
                        eps_estimate=item.get('Forecast'),
                        eps_actual=item.get('Actual'),
                        source="Trading Economics"
                    )

            return None

        except Exception as e:
            logger.error(f"Error getting Trading Economics earnings for {symbol}: {e}")
            return None


class PolygonSource(OptionsDataSource):
    """
    Polygon.io API for options data.
    Requires API key (has free tier with limitations).
    """

    BASE_URL = "https://api.polygon.io"

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key)
        if not api_key:
            raise ValueError("Polygon requires an API key")
        self._session = requests.Session()
        self._session.headers.update({
            'Authorization': f'Bearer {api_key}'
        })

    @property
    def name(self) -> str:
        return "Polygon.io"

    def get_option_chain(
        self,
        symbol: str,
        expiration: Optional[date] = None
    ) -> Optional[OptionChain]:
        """Fetch option chain from Polygon."""
        try:
            # Get current price first
            price_url = f"{self.BASE_URL}/v2/aggs/ticker/{symbol}/prev"
            price_response = self._session.get(price_url, timeout=15)
            price_data = price_response.json()
            current_price = price_data.get('results', [{}])[0].get('c', 0)

            # Get options contracts
            url = f"{self.BASE_URL}/v3/reference/options/contracts"
            params = {
                'underlying_ticker': symbol,
                'limit': 250
            }
            if expiration:
                params['expiration_date'] = expiration.strftime('%Y-%m-%d')

            response = self._session.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            calls = []
            puts = []

            for contract in data.get('results', []):
                option = self._parse_polygon_contract(contract, symbol, current_price)
                if option:
                    if option.option_type == 'call':
                        calls.append(option)
                    else:
                        puts.append(option)

            exp_date = expiration or (date.today() + timedelta(days=7))

            return OptionChain(
                underlying_symbol=symbol,
                underlying_price=current_price,
                expiration_date=exp_date,
                calls=calls,
                puts=puts
            )

        except Exception as e:
            logger.error(f"Error fetching Polygon chain for {symbol}: {e}")
            return None

    def get_option_expirations(self, symbol: str) -> List[date]:
        """Get available expiration dates from Polygon."""
        try:
            url = f"{self.BASE_URL}/v3/reference/options/contracts"
            params = {
                'underlying_ticker': symbol,
                'limit': 1000
            }

            response = self._session.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()
            expirations = set()

            for contract in data.get('results', []):
                exp_str = contract.get('expiration_date')
                if exp_str:
                    try:
                        expirations.add(datetime.strptime(exp_str, '%Y-%m-%d').date())
                    except:
                        continue

            return sorted(list(expirations))

        except Exception as e:
            logger.error(f"Error getting Polygon expirations for {symbol}: {e}")
            return []

    def get_option_quote(self, option_symbol: str) -> Dict[str, Any]:
        """Get real-time quote for an option."""
        try:
            url = f"{self.BASE_URL}/v3/quotes/{option_symbol}"
            response = self._session.get(url, timeout=15)
            response.raise_for_status()
            return response.json().get('results', {})
        except Exception as e:
            logger.error(f"Error getting Polygon option quote: {e}")
            return {}

    def _parse_polygon_contract(
        self,
        contract_data: Dict,
        underlying: str,
        current_price: float
    ) -> Optional[OptionContract]:
        """Parse Polygon contract data."""
        try:
            strike = float(contract_data.get('strike_price', 0))
            option_type = contract_data.get('contract_type', 'call').lower()

            return OptionContract(
                symbol=contract_data.get('ticker', ''),
                underlying=underlying,
                strike=strike,
                expiration=datetime.strptime(
                    contract_data.get('expiration_date', ''),
                    '%Y-%m-%d'
                ).date(),
                option_type=option_type,
                bid=0,  # Would need separate quote call
                ask=0,
                last_price=0,
                volume=0,
                open_interest=int(contract_data.get('open_interest', 0)),
                implied_volatility=0,
                in_the_money=(
                    (option_type == 'call' and strike < current_price) or
                    (option_type == 'put' and strike > current_price)
                )
            )
        except Exception as e:
            logger.debug(f"Error parsing Polygon contract: {e}")
            return None


class InteractiveBrokersSource(OptionsDataSource, OptionFlowSource):
    """
    Interactive Brokers API wrapper for options data.
    Requires IBKR account and TWS/Gateway running.
    """

    def __init__(self, host: str = '127.0.0.1', port: int = 7497, client_id: int = 1):
        super().__init__()
        self.host = host
        self.port = port
        self.client_id = client_id
        self._ib = None
        self._connected = False

    @property
    def name(self) -> str:
        return "Interactive Brokers"

    def connect(self) -> bool:
        """Connect to TWS/Gateway."""
        try:
            from ib_insync import IB
            self._ib = IB()
            self._ib.connect(self.host, self.port, clientId=self.client_id)
            self._connected = True
            logger.info("Connected to Interactive Brokers")
            return True
        except ImportError:
            logger.error("ib_insync not installed. Install with: pip install ib_insync")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to IB: {e}")
            return False

    def disconnect(self):
        """Disconnect from TWS/Gateway."""
        if self._ib and self._connected:
            self._ib.disconnect()
            self._connected = False

    def get_option_chain(
        self,
        symbol: str,
        expiration: Optional[date] = None
    ) -> Optional[OptionChain]:
        """Fetch option chain from Interactive Brokers."""
        if not self._connected:
            if not self.connect():
                return None

        try:
            from ib_insync import Stock, Option

            # Create stock contract
            stock = Stock(symbol, 'SMART', 'USD')
            self._ib.qualifyContracts(stock)

            # Get option chains
            chains = self._ib.reqSecDefOptParams(
                stock.symbol, '', stock.secType, stock.conId
            )

            if not chains:
                return None

            chain = chains[0]

            # Get current price
            ticker = self._ib.reqMktData(stock)
            self._ib.sleep(1)
            current_price = ticker.marketPrice()

            calls = []
            puts = []

            # Get expiration (use closest if not specified)
            expirations = sorted(chain.expirations)
            if expiration:
                exp_str = expiration.strftime('%Y%m%d')
            else:
                exp_str = expirations[0] if expirations else None

            if not exp_str:
                return None

            # Get strikes around current price
            strikes = sorted(chain.strikes)
            atm_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - current_price))
            relevant_strikes = strikes[max(0, atm_idx-10):atm_idx+11]

            for strike in relevant_strikes:
                for right in ['C', 'P']:
                    opt = Option(symbol, exp_str, strike, right, chain.exchange)

                    try:
                        self._ib.qualifyContracts(opt)
                        opt_ticker = self._ib.reqMktData(opt)
                        self._ib.sleep(0.1)

                        contract = OptionContract(
                            symbol=opt.localSymbol,
                            underlying=symbol,
                            strike=strike,
                            expiration=datetime.strptime(exp_str, '%Y%m%d').date(),
                            option_type='call' if right == 'C' else 'put',
                            bid=opt_ticker.bid or 0,
                            ask=opt_ticker.ask or 0,
                            last_price=opt_ticker.last or 0,
                            volume=opt_ticker.volume or 0,
                            open_interest=0,
                            implied_volatility=0,
                            in_the_money=(
                                (right == 'C' and strike < current_price) or
                                (right == 'P' and strike > current_price)
                            )
                        )

                        if right == 'C':
                            calls.append(contract)
                        else:
                            puts.append(contract)

                    except Exception as e:
                        logger.debug(f"Error getting contract {strike}{right}: {e}")
                        continue

            return OptionChain(
                underlying_symbol=symbol,
                underlying_price=current_price,
                expiration_date=datetime.strptime(exp_str, '%Y%m%d').date(),
                calls=calls,
                puts=puts
            )

        except Exception as e:
            logger.error(f"Error getting IB chain for {symbol}: {e}")
            return None

    def get_option_expirations(self, symbol: str) -> List[date]:
        """Get available expiration dates from IB."""
        if not self._connected:
            if not self.connect():
                return []

        try:
            from ib_insync import Stock

            stock = Stock(symbol, 'SMART', 'USD')
            self._ib.qualifyContracts(stock)

            chains = self._ib.reqSecDefOptParams(
                stock.symbol, '', stock.secType, stock.conId
            )

            if chains:
                expirations = []
                for exp_str in chains[0].expirations:
                    try:
                        expirations.append(datetime.strptime(exp_str, '%Y%m%d').date())
                    except:
                        continue
                return sorted(expirations)

            return []

        except Exception as e:
            logger.error(f"Error getting IB expirations for {symbol}: {e}")
            return []

    def get_unusual_activity(
        self,
        symbol: Optional[str] = None,
        min_premium: float = 100000
    ) -> List[OptionFlow]:
        """
        IB doesn't provide direct unusual activity feed.
        This would need to be implemented via scanner or third-party data.
        """
        logger.warning("IB doesn't provide direct unusual activity feed")
        return []


class EconDBSource(DataSource):
    """
    EconDB API for economic data and earnings forecasts.
    Has free tier with limited requests.
    """

    BASE_URL = "https://www.econdb.com/api"

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key)
        self._session = requests.Session()
        if api_key:
            self._session.headers.update({
                'Authorization': f'Token {api_key}'
            })

    @property
    def name(self) -> str:
        return "EconDB"

    @property
    def requires_api_key(self) -> bool:
        return False  # Has free tier

    def get_earnings_calendar(
        self,
        start_date: date,
        end_date: date
    ) -> List[EarningsData]:
        """EconDB focuses on macro data, limited earnings data."""
        logger.info("EconDB focuses on macroeconomic data, limited earnings support")
        return []

    def get_earnings_for_symbol(self, symbol: str) -> Optional[EarningsData]:
        """EconDB focuses on macro data, limited earnings data."""
        return None

    def get_economic_calendar(
        self,
        start_date: date,
        end_date: date,
        country: str = 'US'
    ) -> List[Dict]:
        """Get economic events that may impact earnings."""
        try:
            url = f"{self.BASE_URL}/calendar/"
            params = {
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d'),
                'country': country
            }

            response = self._session.get(url, params=params, timeout=30)
            response.raise_for_status()

            return response.json().get('data', [])

        except Exception as e:
            logger.error(f"Error fetching EconDB calendar: {e}")
            return []


class AlphaVantageSource(DataSource):
    """
    Alpha Vantage API for earnings data.
    Free tier available with limited requests.
    """

    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key)
        if not api_key:
            raise ValueError("Alpha Vantage requires an API key (free tier available)")
        self._session = requests.Session()

    @property
    def name(self) -> str:
        return "Alpha Vantage"

    @property
    def requires_api_key(self) -> bool:
        return True

    def get_earnings_calendar(
        self,
        start_date: date,
        end_date: date
    ) -> List[EarningsData]:
        """Fetch earnings calendar from Alpha Vantage."""
        try:
            params = {
                'function': 'EARNINGS_CALENDAR',
                'horizon': '3month',
                'apikey': self.api_key
            }

            response = self._session.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()

            # Alpha Vantage returns CSV for earnings calendar
            import csv
            from io import StringIO

            reader = csv.DictReader(StringIO(response.text))
            earnings_list = []

            for row in reader:
                try:
                    report_date = datetime.strptime(row.get('reportDate', ''), '%Y-%m-%d')

                    if start_date <= report_date.date() <= end_date:
                        earnings_list.append(EarningsData(
                            symbol=row.get('symbol', ''),
                            earnings_date=report_date,
                            eps_estimate=float(row.get('estimate', 0)) if row.get('estimate') else None,
                            source="Alpha Vantage"
                        ))

                except Exception as e:
                    logger.debug(f"Error parsing Alpha Vantage row: {e}")
                    continue

            return earnings_list

        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage earnings: {e}")
            return []

    def get_earnings_for_symbol(self, symbol: str) -> Optional[EarningsData]:
        """Get earnings data for a specific symbol."""
        try:
            params = {
                'function': 'EARNINGS',
                'symbol': symbol,
                'apikey': self.api_key
            }

            response = self._session.get(self.BASE_URL, params=params, timeout=15)
            response.raise_for_status()

            data = response.json()

            # Get upcoming earnings
            upcoming = data.get('quarterlyEarnings', [])
            if upcoming:
                latest = upcoming[0]
                report_date = datetime.strptime(latest.get('reportedDate', ''), '%Y-%m-%d')

                return EarningsData(
                    symbol=symbol,
                    earnings_date=report_date,
                    eps_estimate=float(latest.get('estimatedEPS', 0)) if latest.get('estimatedEPS') else None,
                    eps_actual=float(latest.get('reportedEPS', 0)) if latest.get('reportedEPS') else None,
                    surprise_percent=float(latest.get('surprisePercentage', 0)) if latest.get('surprisePercentage') else None,
                    source="Alpha Vantage"
                )

            return None

        except Exception as e:
            logger.error(f"Error getting Alpha Vantage earnings for {symbol}: {e}")
            return None

    def get_historical_earnings(self, symbol: str) -> List[Dict]:
        """Get historical earnings for analysis."""
        try:
            params = {
                'function': 'EARNINGS',
                'symbol': symbol,
                'apikey': self.api_key
            }

            response = self._session.get(self.BASE_URL, params=params, timeout=15)
            response.raise_for_status()

            data = response.json()
            return data.get('quarterlyEarnings', [])

        except Exception as e:
            logger.error(f"Error getting Alpha Vantage historical earnings: {e}")
            return []
