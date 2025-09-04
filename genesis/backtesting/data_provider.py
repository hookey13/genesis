"""
Historical Data Provider for Backtesting

Handles loading and streaming of historical market data.
"""

import asyncio
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import List, Dict, Optional, AsyncIterator, Any

import pandas as pd
import structlog

logger = structlog.get_logger()


@dataclass
class DataPoint:
    """Single point of market data."""
    timestamp: datetime
    symbol: str
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    bid_price: Optional[Decimal] = None
    ask_price: Optional[Decimal] = None
    bid_volume: Optional[Decimal] = None
    ask_volume: Optional[Decimal] = None


class HistoricalDataProvider:
    """
    Provider for historical market data.
    
    Supports multiple data sources and caching for performance.
    """
    
    def __init__(self, data_path: Optional[Path] = None):
        """Initialize the data provider.
        
        Args:
            data_path: Path to historical data storage
        """
        self.data_path = data_path or Path(".genesis/data/historical")
        self.cache = {}
        self._db_connection = None
        
    async def load_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        resolution: str = "1m"
    ) -> AsyncIterator[DataPoint]:
        """
        Load historical data for specified symbols and time range.
        
        Args:
            symbols: List of trading symbols
            start_date: Start of historical period
            end_date: End of historical period
            resolution: Data resolution (1m, 5m, 15m, 1h, 1d)
            
        Yields:
            DataPoint objects in chronological order
        """
        logger.info(
            "loading_historical_data",
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            resolution=resolution
        )
        
        # Check cache first
        cache_key = self._get_cache_key(symbols, start_date, end_date, resolution)
        if cache_key in self.cache:
            logger.debug("using_cached_data", cache_key=cache_key)
            for data_point in self.cache[cache_key]:
                yield data_point
            return
        
        # Load from appropriate source
        data_points = []
        
        try:
            # Try loading from database first
            if await self._database_exists():
                async for data_point in self._load_from_database(
                    symbols, start_date, end_date, resolution
                ):
                    data_points.append(data_point)
                    yield data_point
            
            # Fall back to CSV files
            elif await self._csv_files_exist(symbols):
                async for data_point in self._load_from_csv(
                    symbols, start_date, end_date, resolution
                ):
                    data_points.append(data_point)
                    yield data_point
            
            # Generate synthetic data for testing
            else:
                logger.warning(
                    "no_historical_data_found",
                    generating_synthetic=True
                )
                async for data_point in self._generate_synthetic_data(
                    symbols, start_date, end_date, resolution
                ):
                    data_points.append(data_point)
                    yield data_point
            
            # Cache the loaded data
            if data_points:
                self.cache[cache_key] = data_points
                logger.debug(
                    "data_cached",
                    cache_key=cache_key,
                    points=len(data_points)
                )
                
        except Exception as e:
            logger.error(
                "data_load_failed",
                error=str(e),
                symbols=symbols
            )
            raise
    
    async def _database_exists(self) -> bool:
        """Check if historical database exists."""
        db_path = self.data_path / "historical.db"
        return db_path.exists()
    
    async def _csv_files_exist(self, symbols: List[str]) -> bool:
        """Check if CSV files exist for symbols."""
        for symbol in symbols:
            csv_path = self.data_path / f"{symbol}.csv"
            if csv_path.exists():
                return True
        return False
    
    async def _load_from_database(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        resolution: str
    ) -> AsyncIterator[DataPoint]:
        """Load data from SQLite database."""
        db_path = self.data_path / "historical.db"
        
        # Convert resolution to seconds
        resolution_seconds = self._parse_resolution(resolution)
        
        query = """
            SELECT timestamp, symbol, open, high, low, close, volume,
                   bid_price, ask_price, bid_volume, ask_volume
            FROM market_data
            WHERE symbol IN ({}) 
            AND timestamp >= ?
            AND timestamp <= ?
            ORDER BY timestamp
        """.format(','.join('?' * len(symbols)))
        
        async with aiosqlite.connect(str(db_path)) as db:
            async with db.execute(
                query, 
                symbols + [start_date.timestamp(), end_date.timestamp()]
            ) as cursor:
                async for row in cursor:
                    yield DataPoint(
                        timestamp=datetime.fromtimestamp(row[0]),
                        symbol=row[1],
                        open=Decimal(str(row[2])),
                        high=Decimal(str(row[3])),
                        low=Decimal(str(row[4])),
                        close=Decimal(str(row[5])),
                        volume=Decimal(str(row[6])),
                        bid_price=Decimal(str(row[7])) if row[7] else None,
                        ask_price=Decimal(str(row[8])) if row[8] else None,
                        bid_volume=Decimal(str(row[9])) if row[9] else None,
                        ask_volume=Decimal(str(row[10])) if row[10] else None,
                    )
    
    async def _load_from_csv(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        resolution: str
    ) -> AsyncIterator[DataPoint]:
        """Load data from CSV files."""
        all_data = []
        
        for symbol in symbols:
            csv_path = self.data_path / f"{symbol}.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Filter by date range
                mask = (df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)
                df = df[mask]
                
                for _, row in df.iterrows():
                    data_point = DataPoint(
                        timestamp=row['timestamp'].to_pydatetime(),
                        symbol=symbol,
                        open=Decimal(str(row['open'])),
                        high=Decimal(str(row['high'])),
                        low=Decimal(str(row['low'])),
                        close=Decimal(str(row['close'])),
                        volume=Decimal(str(row['volume'])),
                        bid_price=Decimal(str(row.get('bid_price'))) if 'bid_price' in row else None,
                        ask_price=Decimal(str(row.get('ask_price'))) if 'ask_price' in row else None,
                    )
                    all_data.append(data_point)
        
        # Sort by timestamp and yield
        all_data.sort(key=lambda x: x.timestamp)
        for data_point in all_data:
            yield data_point
    
    async def _generate_synthetic_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        resolution: str
    ) -> AsyncIterator[DataPoint]:
        """
        Generate synthetic market data for testing.
        
        Creates realistic-looking price movements with trends and volatility.
        """
        import random
        import math
        
        resolution_seconds = self._parse_resolution(resolution)
        current_time = start_date
        
        # Initialize prices for each symbol
        prices = {}
        for symbol in symbols:
            # Random starting price between 10 and 1000
            base_price = Decimal(str(random.uniform(10, 1000)))
            prices[symbol] = {
                'price': base_price,
                'trend': random.choice([-1, 1]) * random.uniform(0.0001, 0.001),
                'volatility': random.uniform(0.001, 0.01)
            }
        
        while current_time <= end_date:
            for symbol in symbols:
                price_info = prices[symbol]
                
                # Apply trend and random walk
                trend_component = price_info['trend']
                random_component = random.gauss(0, float(price_info['volatility']))
                price_change = 1 + trend_component + random_component
                
                # Update price
                new_price = price_info['price'] * Decimal(str(price_change))
                new_price = max(Decimal('0.01'), new_price)  # Prevent negative prices
                
                # Generate OHLC from base price
                volatility = float(price_info['volatility'])
                high = new_price * Decimal(str(1 + random.uniform(0, volatility)))
                low = new_price * Decimal(str(1 - random.uniform(0, volatility)))
                open_price = price_info['price']
                close = new_price
                
                # Generate volume
                base_volume = Decimal(str(random.uniform(1000, 100000)))
                volume = base_volume * Decimal(str(1 + random.gauss(0, 0.3)))
                
                # Generate bid/ask
                spread = new_price * Decimal('0.001')  # 0.1% spread
                bid_price = new_price - spread / 2
                ask_price = new_price + spread / 2
                
                yield DataPoint(
                    timestamp=current_time,
                    symbol=symbol,
                    open=open_price,
                    high=high,
                    low=low,
                    close=close,
                    volume=abs(volume),
                    bid_price=bid_price,
                    ask_price=ask_price,
                    bid_volume=abs(volume) / 2,
                    ask_volume=abs(volume) / 2
                )
                
                # Update price for next iteration
                price_info['price'] = new_price
                
                # Occasionally change trend
                if random.random() < 0.01:  # 1% chance per tick
                    price_info['trend'] = random.choice([-1, 1]) * random.uniform(0.0001, 0.001)
            
            # Move to next time period
            current_time += timedelta(seconds=resolution_seconds)
    
    def _parse_resolution(self, resolution: str) -> int:
        """Parse resolution string to seconds.
        
        Args:
            resolution: Resolution string (1m, 5m, 15m, 1h, 1d)
            
        Returns:
            Number of seconds
        """
        unit = resolution[-1]
        value = int(resolution[:-1])
        
        if unit == 'm':
            return value * 60
        elif unit == 'h':
            return value * 3600
        elif unit == 'd':
            return value * 86400
        else:
            raise ValueError(f"Invalid resolution: {resolution}")
    
    def _get_cache_key(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        resolution: str
    ) -> str:
        """Generate cache key for data request."""
        symbols_str = "_".join(sorted(symbols))
        return f"{symbols_str}_{start_date.isoformat()}_{end_date.isoformat()}_{resolution}"
    
    async def check_data_availability(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        resolution: str
    ) -> bool:
        """
        Check if data is available for the requested period.
        
        Args:
            symbols: Trading symbols to check
            start_date: Start of period
            end_date: End of period
            resolution: Data resolution
            
        Returns:
            True if data is available
        """
        # Check cache
        cache_key = self._get_cache_key(symbols, start_date, end_date, resolution)
        if cache_key in self.cache:
            return True
        
        # Check database
        if await self._database_exists():
            return True
        
        # Check CSV files
        if await self._csv_files_exist(symbols):
            return True
        
        # Synthetic data is always available
        return True
    
    async def preload_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        resolution: str = "1m"
    ) -> None:
        """
        Preload data into cache for faster access.
        
        Args:
            symbols: Symbols to preload
            start_date: Start of period
            end_date: End of period
            resolution: Data resolution
        """
        data_points = []
        async for point in self.load_data(symbols, start_date, end_date, resolution):
            data_points.append(point)
        
        logger.info(
            "data_preloaded",
            symbols=symbols,
            points=len(data_points),
            period_days=(end_date - start_date).days
        )