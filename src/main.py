"""
Main application for earnings move prediction.
Provides both free and premium modes with CLI interface.
"""

import logging
import json
import os
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from tabulate import tabulate

from .analysis.earnings_predictor import EarningsPredictor, EarningsPrediction
from .analysis.expected_move import ExpectedMoveCalculator, get_expected_move_summary
from .analysis.options_analyzer import OptionsAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

console = Console()


class EarningsMovePredictor:
    """
    Main application class for earnings move prediction.

    Supports two modes:
    - Free: Uses Yahoo Finance, web scraping, and free APIs
    - Premium: Uses paid APIs for faster, more comprehensive data
    """

    def __init__(self, mode: str = 'free', config_path: Optional[str] = None):
        """
        Initialize the predictor.

        Args:
            mode: 'free' or 'premium'
            config_path: Optional path to configuration file
        """
        self.mode = mode
        self.config = self._load_config(config_path)

        # Initialize predictor based on mode
        if mode == 'premium':
            self.predictor = EarningsPredictor(
                use_free_sources=True,
                use_premium_sources=True
            )
        else:
            self.predictor = EarningsPredictor(
                use_free_sources=True,
                use_premium_sources=False
            )

        logger.info(f"Initialized in {mode} mode")

    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from file or environment."""
        config = {
            'output_format': 'text',
            'cache_enabled': True,
            'cache_ttl': 300,  # 5 minutes
        }

        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                file_config = json.load(f)
                config.update(file_config)

        # Load API keys from environment
        config['api_keys'] = {
            'unusual_whales': os.getenv('UNUSUAL_WHALES_API_KEY'),
            'trading_economics': os.getenv('TRADING_ECONOMICS_API_KEY'),
            'polygon': os.getenv('POLYGON_API_KEY'),
            'alpha_vantage': os.getenv('ALPHA_VANTAGE_API_KEY'),
        }

        return config

    def predict_single(
        self,
        symbol: str,
        output_format: str = 'text'
    ) -> Optional[EarningsPrediction]:
        """
        Generate prediction for a single symbol.

        Args:
            symbol: Stock symbol
            output_format: 'text', 'json', or 'table'

        Returns:
            EarningsPrediction or None if failed
        """
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task(f"Analyzing {symbol}...", total=None)

                prediction = self.predictor.predict(symbol)

                progress.update(task, completed=True)

            self._output_prediction(prediction, output_format)
            return prediction

        except Exception as e:
            console.print(f"[red]Error analyzing {symbol}: {e}[/red]")
            logger.error(f"Error predicting {symbol}: {e}")
            return None

    def predict_batch(
        self,
        symbols: List[str],
        output_format: str = 'table'
    ) -> List[EarningsPrediction]:
        """
        Generate predictions for multiple symbols.

        Args:
            symbols: List of stock symbols
            output_format: 'text', 'json', or 'table'

        Returns:
            List of EarningsPrediction objects
        """
        predictions = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(
                f"Analyzing {len(symbols)} symbols...",
                total=len(symbols)
            )

            for symbol in symbols:
                try:
                    prediction = self.predictor.predict(symbol)
                    predictions.append(prediction)
                except Exception as e:
                    logger.error(f"Error predicting {symbol}: {e}")

                progress.advance(task)

        self._output_batch(predictions, output_format)
        return predictions

    def get_earnings_calendar(
        self,
        days_ahead: int = 7,
        output_format: str = 'table'
    ):
        """
        Get upcoming earnings calendar.

        Args:
            days_ahead: Number of days to look ahead
            output_format: Output format
        """
        start = date.today()
        end = start + timedelta(days=days_ahead)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Fetching earnings calendar...", total=None)

            df = self.predictor.get_earnings_calendar(start, end)

            progress.update(task, completed=True)

        if df.empty:
            console.print("[yellow]No earnings found in the specified period[/yellow]")
            return

        if output_format == 'json':
            console.print(df.to_json(orient='records', indent=2))
        else:
            table = Table(title=f"Earnings Calendar ({start} to {end})")
            table.add_column("Symbol", style="cyan")
            table.add_column("Date", style="green")
            table.add_column("Time", style="yellow")
            table.add_column("EPS Est.", style="magenta")

            for _, row in df.iterrows():
                table.add_row(
                    row['symbol'],
                    row['earnings_date'].strftime('%Y-%m-%d'),
                    row['time'],
                    f"${row['eps_estimate']:.2f}" if row['eps_estimate'] else "N/A"
                )

            console.print(table)

    def analyze_options(self, symbol: str, expiration: Optional[str] = None):
        """
        Detailed options analysis for a symbol.

        Args:
            symbol: Stock symbol
            expiration: Optional expiration date (YYYY-MM-DD)
        """
        from .data_sources.free_sources import YahooFinanceSource

        yahoo = YahooFinanceSource()

        exp_date = None
        if expiration:
            exp_date = datetime.strptime(expiration, '%Y-%m-%d').date()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"Fetching options for {symbol}...", total=None)

            chain = yahoo.get_option_chain(symbol, exp_date)

            progress.update(task, completed=True)

        if not chain:
            console.print(f"[red]Could not get option chain for {symbol}[/red]")
            return

        analyzer = OptionsAnalyzer()
        metrics = analyzer.analyze_chain(chain)
        calc = ExpectedMoveCalculator(analyzer)
        expected_move = calc.calculate_expected_move(chain)

        # Display results
        console.print(Panel.fit(
            f"[bold]{symbol}[/bold] Options Analysis\n"
            f"Price: ${chain.underlying_price:.2f}\n"
            f"Expiration: {chain.expiration_date}",
            title="Overview"
        ))

        table = Table(title="Key Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("ATM Strike", f"${metrics.atm_strike:.2f}")
        table.add_row("ATM Straddle", f"${metrics.atm_straddle_price:.2f}")
        table.add_row("Expected Move", f"±{expected_move.expected_move_percent:.2f}%")
        table.add_row("ATM IV", f"{metrics.atm_iv*100:.1f}%")
        table.add_row("Put/Call Ratio", f"{metrics.put_call_ratio:.2f}")
        table.add_row("Max Pain", f"${metrics.max_pain_strike:.2f}")
        table.add_row("IV Skew", f"{metrics.iv_skew:.4f}")

        console.print(table)

        # Strike analysis
        strike_df = analyzer.get_strike_analysis(chain)
        console.print("\n[bold]Strike Analysis (around ATM):[/bold]")
        console.print(tabulate(
            strike_df[['strike', 'call_volume', 'call_oi', 'put_volume', 'put_oi', 'straddle_price']].head(10),
            headers='keys',
            tablefmt='simple',
            floatfmt='.2f'
        ))

    def _output_prediction(self, prediction: EarningsPrediction, format: str):
        """Output a single prediction."""
        if format == 'json':
            console.print(json.dumps(prediction.to_dict(), indent=2))
        elif format == 'table':
            self._output_batch([prediction], 'table')
        else:
            console.print(prediction.summary())

    def _output_batch(self, predictions: List[EarningsPrediction], format: str):
        """Output batch predictions."""
        if format == 'json':
            data = [p.to_dict() for p in predictions]
            console.print(json.dumps(data, indent=2))
        elif format == 'table':
            table = Table(title="Earnings Move Predictions")
            table.add_column("Symbol", style="cyan")
            table.add_column("Price", style="green")
            table.add_column("Earnings", style="yellow")
            table.add_column("Expected Move", style="magenta")
            table.add_column("Direction", style="blue")
            table.add_column("Confidence", style="red")

            for p in predictions:
                direction_color = "green" if p.predicted_direction == "up" else "red" if p.predicted_direction == "down" else "yellow"
                table.add_row(
                    p.symbol,
                    f"${p.current_price:.2f}",
                    p.earnings_date.strftime('%m/%d') if p.earnings_date else "TBD",
                    f"±{p.expected_move_percent:.1f}%",
                    f"[{direction_color}]{p.predicted_direction.upper()}[/{direction_color}]",
                    f"{p.overall_confidence*100:.0f}%"
                )

            console.print(table)
        else:
            for p in predictions:
                console.print(p.summary())
                console.print()


# CLI Interface
@click.group()
@click.option('--mode', type=click.Choice(['free', 'premium']), default='free',
              help='Data source mode (free or premium)')
@click.option('--config', type=click.Path(), default=None,
              help='Path to configuration file')
@click.pass_context
def cli(ctx, mode, config):
    """Earnings Move Predictor - Predict stock moves after earnings."""
    ctx.ensure_object(dict)
    ctx.obj['app'] = EarningsMovePredictor(mode=mode, config_path=config)


@cli.command()
@click.argument('symbol')
@click.option('--format', '-f', type=click.Choice(['text', 'json', 'table']),
              default='text', help='Output format')
@click.pass_context
def predict(ctx, symbol, format):
    """Predict expected move for a single symbol."""
    app = ctx.obj['app']
    app.predict_single(symbol.upper(), format)


@cli.command()
@click.argument('symbols', nargs=-1, required=True)
@click.option('--format', '-f', type=click.Choice(['text', 'json', 'table']),
              default='table', help='Output format')
@click.pass_context
def batch(ctx, symbols, format):
    """Predict expected moves for multiple symbols."""
    app = ctx.obj['app']
    app.predict_batch([s.upper() for s in symbols], format)


@cli.command()
@click.option('--days', '-d', default=7, help='Days to look ahead')
@click.option('--format', '-f', type=click.Choice(['text', 'json', 'table']),
              default='table', help='Output format')
@click.pass_context
def calendar(ctx, days, format):
    """Get upcoming earnings calendar."""
    app = ctx.obj['app']
    app.get_earnings_calendar(days, format)


@cli.command()
@click.argument('symbol')
@click.option('--expiration', '-e', default=None,
              help='Expiration date (YYYY-MM-DD)')
@click.pass_context
def options(ctx, symbol, expiration):
    """Detailed options analysis for a symbol."""
    app = ctx.obj['app']
    app.analyze_options(symbol.upper(), expiration)


@cli.command()
@click.pass_context
def status(ctx):
    """Show current configuration and available data sources."""
    app = ctx.obj['app']

    console.print(Panel.fit(
        f"Mode: {app.mode}\n"
        f"Earnings Sources: {len(app.predictor.earnings_sources)}\n"
        f"Options Sources: {len(app.predictor.options_sources)}\n"
        f"Flow Sources: {len(app.predictor.flow_sources)}",
        title="Configuration"
    ))

    # Show data sources
    table = Table(title="Data Sources")
    table.add_column("Type", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Status", style="yellow")

    for src in app.predictor.earnings_sources:
        table.add_row("Earnings", src.name, "Active")

    for src in app.predictor.options_sources:
        table.add_row("Options", src.name, "Active")

    for src in app.predictor.flow_sources:
        table.add_row("Flow", src.name, "Active")

    console.print(table)

    # Show API key status
    console.print("\n[bold]API Keys:[/bold]")
    for key_name, key_value in app.config.get('api_keys', {}).items():
        status = "[green]Set[/green]" if key_value else "[red]Not Set[/red]"
        console.print(f"  {key_name}: {status}")


def main():
    """Main entry point."""
    cli(obj={})


if __name__ == '__main__':
    main()
