"""Spread calculation and analysis module for pairs trading."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd
import structlog
from scipy import stats
from statsmodels.api import add_constant
from statsmodels.regression.linear_model import OLS

logger = structlog.get_logger(__name__)


@dataclass
class SpreadMetrics:
    """Metrics for a spread between two assets."""
    
    spread_values: np.ndarray
    mean: Decimal
    std_dev: Decimal
    current_value: Decimal
    current_zscore: Decimal
    half_life: int | None
    hurst_exponent: float
    adf_pvalue: Decimal
    max_zscore: Decimal
    min_zscore: Decimal
    spread_type: str  # 'log', 'ratio', 'dollar'
    metadata: dict[str, Any] | None = None


@dataclass
class CorrelationMetrics:
    """Correlation metrics between two assets."""
    
    pearson_correlation: Decimal
    spearman_correlation: Decimal
    kendall_correlation: Decimal
    rolling_correlation: np.ndarray
    correlation_stability: Decimal  # Std dev of rolling correlation
    correlation_trend: Decimal  # Slope of correlation over time
    is_stable: bool
    metadata: dict[str, Any] | None = None


@dataclass
class HedgeRatio:
    """Hedge ratio calculation results."""
    
    ratio: Decimal
    method: str  # 'ols', 'tls', 'kalman'
    r_squared: Decimal
    confidence_interval: tuple[Decimal, Decimal]
    stability_score: Decimal  # 0-1, higher is more stable
    last_updated: datetime
    metadata: dict[str, Any] | None = None


class SpreadCalculator:
    """Calculate and analyze spreads for pairs trading."""
    
    def __init__(self, lookback_window: int = 100):
        """Initialize spread calculator.
        
        Args:
            lookback_window: Default window for rolling calculations.
        """
        self.lookback_window = lookback_window
        self.min_periods = max(20, lookback_window // 5)
    
    def calculate_correlation(
        self,
        series1: pd.Series | np.ndarray,
        series2: pd.Series | np.ndarray,
        window: int | None = None
    ) -> CorrelationMetrics:
        """Calculate comprehensive correlation metrics.
        
        Args:
            series1: First price series.
            series2: Second price series.
            window: Rolling window size (default: lookback_window).
            
        Returns:
            CorrelationMetrics with all correlation measures.
        """
        try:
            # Convert to pandas Series for easier manipulation
            if isinstance(series1, np.ndarray):
                series1 = pd.Series(series1)
            if isinstance(series2, np.ndarray):
                series2 = pd.Series(series2)
            
            # Ensure same length
            min_len = min(len(series1), len(series2))
            series1 = series1.iloc[:min_len] if isinstance(series1, pd.Series) else series1[:min_len]
            series2 = series2.iloc[:min_len] if isinstance(series2, pd.Series) else series2[:min_len]
            
            # Calculate different correlation measures
            pearson_corr = series1.corr(series2, method='pearson')
            spearman_corr = series1.corr(series2, method='spearman')
            kendall_corr = series1.corr(series2, method='kendall')
            
            # Calculate rolling correlation
            window = window or self.lookback_window
            rolling_corr = series1.rolling(window=window, min_periods=self.min_periods).corr(series2)
            rolling_corr_values = rolling_corr.dropna().values
            
            # Calculate correlation stability (std dev of rolling correlation)
            correlation_stability = np.std(rolling_corr_values) if len(rolling_corr_values) > 0 else 1.0
            
            # Calculate correlation trend (is correlation increasing or decreasing?)
            if len(rolling_corr_values) > 1:
                x = np.arange(len(rolling_corr_values))
                slope, _, _, _, _ = stats.linregress(x, rolling_corr_values)
                correlation_trend = slope
            else:
                correlation_trend = 0.0
            
            # Determine if correlation is stable
            # Stable if: high correlation, low variability, not trending down
            is_stable = (
                pearson_corr > 0.8 and
                correlation_stability < 0.1 and
                correlation_trend >= -0.001
            )
            
            return CorrelationMetrics(
                pearson_correlation=Decimal(str(pearson_corr)),
                spearman_correlation=Decimal(str(spearman_corr)),
                kendall_correlation=Decimal(str(kendall_corr)),
                rolling_correlation=rolling_corr_values,
                correlation_stability=Decimal(str(correlation_stability)),
                correlation_trend=Decimal(str(correlation_trend)),
                is_stable=is_stable,
                metadata={
                    "n_observations": len(series1),
                    "window_size": window,
                    "min_correlation": float(np.min(rolling_corr_values)) if len(rolling_corr_values) > 0 else 0,
                    "max_correlation": float(np.max(rolling_corr_values)) if len(rolling_corr_values) > 0 else 0
                }
            )
            
        except Exception as e:
            logger.error(f"Correlation calculation failed: {e}")
            return CorrelationMetrics(
                pearson_correlation=Decimal("0"),
                spearman_correlation=Decimal("0"),
                kendall_correlation=Decimal("0"),
                rolling_correlation=np.array([]),
                correlation_stability=Decimal("1"),
                correlation_trend=Decimal("0"),
                is_stable=False
            )
    
    def calculate_hedge_ratio(
        self,
        series1: pd.Series | np.ndarray,
        series2: pd.Series | np.ndarray,
        method: str = "ols"
    ) -> HedgeRatio:
        """Calculate optimal hedge ratio between two series.
        
        Args:
            series1: Dependent variable (y).
            series2: Independent variable (x).
            method: Method to use ('ols', 'tls', 'rolling').
            
        Returns:
            HedgeRatio with calculated ratio and metrics.
        """
        try:
            # Convert to numpy arrays
            if isinstance(series1, pd.Series):
                y = series1.values
            else:
                y = series1
            
            if isinstance(series2, pd.Series):
                x = series2.values
            else:
                x = series2
            
            # Ensure same length
            min_len = min(len(y), len(x))
            y = y[:min_len]
            x = x[:min_len]
            
            if method == "ols":
                # Ordinary Least Squares
                X = add_constant(x)
                model = OLS(y, X)
                results = model.fit()
                
                hedge_ratio = results.params[1]
                r_squared = results.rsquared
                
                # Calculate confidence interval
                conf_int = results.conf_int(alpha=0.05)
                ci_lower = conf_int[1, 0]
                ci_upper = conf_int[1, 1]
                
                # Calculate stability score based on R-squared and CI width
                ci_width = ci_upper - ci_lower
                stability_score = r_squared * (1 / (1 + ci_width))
                
            elif method == "tls":
                # Total Least Squares (orthogonal regression)
                # More robust when both variables have measurement error
                
                # Center the data
                x_mean = np.mean(x)
                y_mean = np.mean(y)
                x_centered = x - x_mean
                y_centered = y - y_mean
                
                # Create data matrix
                data_matrix = np.column_stack([x_centered, y_centered])
                
                # Perform SVD
                _, _, V = np.linalg.svd(data_matrix)
                
                # The hedge ratio is the ratio of the components of the first principal component
                hedge_ratio = V[1, 1] / V[0, 1]
                
                # Calculate R-squared equivalent
                y_pred = hedge_ratio * x
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)
                
                # Simplified confidence interval for TLS
                std_error = np.sqrt(ss_res / (len(y) - 2))
                ci_width = 1.96 * std_error
                ci_lower = hedge_ratio - ci_width
                ci_upper = hedge_ratio + ci_width
                
                stability_score = r_squared * 0.9  # Slightly lower than OLS
                
            elif method == "rolling":
                # Rolling window OLS for dynamic hedge ratio
                window = self.lookback_window
                
                # Calculate rolling hedge ratios
                rolling_ratios = []
                for i in range(window, len(x)):
                    x_window = x[i-window:i]
                    y_window = y[i-window:i]
                    
                    X_window = add_constant(x_window)
                    model = OLS(y_window, X_window)
                    results = model.fit()
                    rolling_ratios.append(results.params[1])
                
                # Use the most recent ratio
                hedge_ratio = rolling_ratios[-1] if rolling_ratios else 1.0
                
                # Calculate stability based on variance of rolling ratios
                ratio_std = np.std(rolling_ratios) if len(rolling_ratios) > 1 else 1.0
                stability_score = 1 / (1 + ratio_std)
                
                # Simple R-squared for the most recent window
                y_pred = hedge_ratio * x[-window:]
                y_actual = y[-window:]
                ss_res = np.sum((y_actual - y_pred) ** 2)
                ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                # Confidence interval based on rolling ratios
                if len(rolling_ratios) > 1:
                    ci_lower = np.percentile(rolling_ratios, 2.5)
                    ci_upper = np.percentile(rolling_ratios, 97.5)
                else:
                    ci_lower = hedge_ratio * 0.9
                    ci_upper = hedge_ratio * 1.1
            
            else:
                raise ValueError(f"Unknown method: {method}")
            
            return HedgeRatio(
                ratio=Decimal(str(hedge_ratio)),
                method=method,
                r_squared=Decimal(str(r_squared)),
                confidence_interval=(Decimal(str(ci_lower)), Decimal(str(ci_upper))),
                stability_score=Decimal(str(stability_score)),
                last_updated=datetime.now(),
                metadata={
                    "n_observations": len(x),
                    "method_details": method
                }
            )
            
        except Exception as e:
            logger.error(f"Hedge ratio calculation failed: {e}")
            return HedgeRatio(
                ratio=Decimal("1"),
                method=method,
                r_squared=Decimal("0"),
                confidence_interval=(Decimal("0.9"), Decimal("1.1")),
                stability_score=Decimal("0"),
                last_updated=datetime.now()
            )
    
    def calculate_spread(
        self,
        series1: pd.Series | np.ndarray,
        series2: pd.Series | np.ndarray,
        hedge_ratio: Decimal | float = 1.0,
        spread_type: str = "log"
    ) -> SpreadMetrics:
        """Calculate spread between two series.
        
        Args:
            series1: First price series.
            series2: Second price series.
            hedge_ratio: Hedge ratio to use.
            spread_type: Type of spread ('log', 'ratio', 'dollar').
            
        Returns:
            SpreadMetrics with spread analysis.
        """
        try:
            # Convert to numpy arrays
            if isinstance(series1, pd.Series):
                s1 = series1.values
            else:
                s1 = series1.copy()
            
            if isinstance(series2, pd.Series):
                s2 = series2.values
            else:
                s2 = series2.copy()
            
            # Ensure same length
            min_len = min(len(s1), len(s2))
            s1 = s1[:min_len]
            s2 = s2[:min_len]
            
            hedge_ratio = float(hedge_ratio)
            
            # Calculate spread based on type
            if spread_type == "log":
                # Log spread (most common for pairs trading)
                spread = np.log(s1) - hedge_ratio * np.log(s2)
            elif spread_type == "ratio":
                # Ratio spread
                spread = s1 / (hedge_ratio * s2)
            elif spread_type == "dollar":
                # Dollar neutral spread
                spread = s1 - hedge_ratio * s2
            else:
                raise ValueError(f"Unknown spread type: {spread_type}")
            
            # Remove any NaN or inf values
            spread = spread[np.isfinite(spread)]
            
            if len(spread) == 0:
                raise ValueError("No valid spread values after cleaning")
            
            # Calculate spread statistics
            spread_mean = np.mean(spread)
            spread_std = np.std(spread)
            current_value = spread[-1]
            
            # Calculate z-score
            if spread_std > 0:
                current_zscore = (current_value - spread_mean) / spread_std
            else:
                current_zscore = 0.0
            
            # Calculate half-life using Ornstein-Uhlenbeck process
            half_life = self._calculate_half_life(spread)
            
            # Calculate Hurst exponent
            hurst = self._calculate_hurst_exponent(spread)
            
            # Test spread for stationarity
            from statsmodels.tsa.stattools import adfuller
            adf_result = adfuller(spread, autolag='AIC')
            adf_pvalue = adf_result[1]
            
            # Calculate z-score extremes
            z_scores = (spread - spread_mean) / spread_std if spread_std > 0 else np.zeros_like(spread)
            max_zscore = np.max(z_scores)
            min_zscore = np.min(z_scores)
            
            return SpreadMetrics(
                spread_values=spread,
                mean=Decimal(str(spread_mean)),
                std_dev=Decimal(str(spread_std)),
                current_value=Decimal(str(current_value)),
                current_zscore=Decimal(str(current_zscore)),
                half_life=half_life,
                hurst_exponent=hurst,
                adf_pvalue=Decimal(str(adf_pvalue)),
                max_zscore=Decimal(str(max_zscore)),
                min_zscore=Decimal(str(min_zscore)),
                spread_type=spread_type,
                metadata={
                    "n_observations": len(spread),
                    "is_stationary": adf_pvalue < 0.05,
                    "is_mean_reverting": hurst < 0.5,
                    "spread_range": float(np.max(spread) - np.min(spread))
                }
            )
            
        except Exception as e:
            logger.error(f"Spread calculation failed: {e}")
            return SpreadMetrics(
                spread_values=np.array([]),
                mean=Decimal("0"),
                std_dev=Decimal("1"),
                current_value=Decimal("0"),
                current_zscore=Decimal("0"),
                half_life=None,
                hurst_exponent=0.5,
                adf_pvalue=Decimal("1"),
                max_zscore=Decimal("0"),
                min_zscore=Decimal("0"),
                spread_type=spread_type
            )
    
    def calculate_zscore(
        self,
        spread: np.ndarray | pd.Series,
        window: int | None = None
    ) -> np.ndarray:
        """Calculate rolling z-score of spread.
        
        Args:
            spread: Spread values.
            window: Rolling window size (default: lookback_window).
            
        Returns:
            Array of z-scores.
        """
        try:
            # Convert to pandas Series for rolling calculations
            if isinstance(spread, np.ndarray):
                spread = pd.Series(spread)
            
            window = window or self.lookback_window
            
            # Calculate rolling mean and std
            rolling_mean = spread.rolling(window=window, min_periods=self.min_periods).mean()
            rolling_std = spread.rolling(window=window, min_periods=self.min_periods).std()
            
            # Calculate z-score
            z_scores = (spread - rolling_mean) / rolling_std
            
            # Replace NaN and inf with 0
            z_scores = z_scores.fillna(0)
            z_scores = z_scores.replace([np.inf, -np.inf], 0)
            
            return z_scores.values
            
        except Exception as e:
            logger.error(f"Z-score calculation failed: {e}")
            return np.zeros(len(spread))
    
    def _calculate_half_life(self, spread: np.ndarray) -> int | None:
        """Calculate half-life of mean reversion.
        
        Args:
            spread: Spread values.
            
        Returns:
            Half-life in periods or None if not mean-reverting.
        """
        try:
            # Use Ornstein-Uhlenbeck process
            spread_lag = spread[:-1]
            spread_diff = spread[1:] - spread_lag
            
            # OLS regression
            X = add_constant(spread_lag)
            model = OLS(spread_diff, X)
            results = model.fit()
            
            # Get mean reversion coefficient
            theta = -results.params[1]
            
            # Calculate half-life
            if theta > 0:
                half_life = int(np.log(2) / theta)
                # Cap half-life at reasonable values
                half_life = min(half_life, 252)  # Max 1 trading year
            else:
                half_life = None  # No mean reversion
            
            return half_life
            
        except Exception as e:
            logger.error(f"Half-life calculation failed: {e}")
            return None
    
    def _calculate_hurst_exponent(self, series: np.ndarray) -> float:
        """Calculate Hurst exponent.
        
        Args:
            series: Time series.
            
        Returns:
            Hurst exponent (< 0.5 = mean reverting).
        """
        try:
            # Ensure we have enough data
            if len(series) < 20:
                return 0.5
            
            # Calculate the range of cumulative deviations
            lags = range(2, min(100, len(series) // 2))
            
            # Calculate the variance of the lagged differences
            tau = []
            for lag in lags:
                diff = series[lag:] - series[:-lag]
                tau.append(np.sqrt(np.mean(diff ** 2)))
            
            # Perform linear regression on log-log plot
            log_lags = np.log(list(lags))
            log_tau = np.log(tau)
            
            # Use polyfit for linear regression
            poly = np.polyfit(log_lags, log_tau, 1)
            hurst = poly[0]
            
            # Ensure Hurst is in valid range [0, 1]
            hurst = max(0, min(1, hurst))
            
            return hurst
            
        except Exception as e:
            logger.error(f"Hurst exponent calculation failed: {e}")
            return 0.5
    
    def analyze_spread_quality(
        self,
        spread_metrics: SpreadMetrics,
        min_half_life: int = 5,
        max_half_life: int = 50
    ) -> dict[str, Any]:
        """Analyze the quality of a spread for trading.
        
        Args:
            spread_metrics: Spread metrics to analyze.
            min_half_life: Minimum acceptable half-life.
            max_half_life: Maximum acceptable half-life.
            
        Returns:
            Dictionary with quality analysis.
        """
        quality_score = 0.0
        issues = []
        strengths = []
        
        # Check stationarity
        if spread_metrics.adf_pvalue < Decimal("0.05"):
            quality_score += 0.25
            strengths.append("Spread is stationary")
        else:
            issues.append("Spread is not stationary")
        
        # Check mean reversion (Hurst < 0.5)
        if spread_metrics.hurst_exponent < 0.5:
            quality_score += 0.25
            strengths.append(f"Mean reverting (Hurst={spread_metrics.hurst_exponent:.3f})")
        else:
            issues.append(f"Not mean reverting (Hurst={spread_metrics.hurst_exponent:.3f})")
        
        # Check half-life
        if spread_metrics.half_life:
            if min_half_life <= spread_metrics.half_life <= max_half_life:
                quality_score += 0.25
                strengths.append(f"Good half-life ({spread_metrics.half_life} periods)")
            else:
                issues.append(f"Half-life out of range ({spread_metrics.half_life} periods)")
        else:
            issues.append("No measurable half-life")
        
        # Check z-score range
        z_range = abs(float(spread_metrics.max_zscore - spread_metrics.min_zscore))
        if z_range > 4:
            quality_score += 0.25
            strengths.append(f"Good z-score range ({z_range:.2f})")
        else:
            issues.append(f"Limited z-score range ({z_range:.2f})")
        
        # Calculate trading opportunities
        z_scores = (spread_metrics.spread_values - float(spread_metrics.mean)) / float(spread_metrics.std_dev)
        entry_opportunities = np.sum(np.abs(z_scores) >= 2.0)
        exit_opportunities = np.sum(np.abs(z_scores) <= 0.5)
        
        return {
            "quality_score": quality_score,
            "is_tradeable": quality_score >= 0.5,
            "strengths": strengths,
            "issues": issues,
            "entry_opportunities": int(entry_opportunities),
            "exit_opportunities": int(exit_opportunities),
            "metrics_summary": {
                "stationarity_pvalue": float(spread_metrics.adf_pvalue),
                "hurst_exponent": spread_metrics.hurst_exponent,
                "half_life": spread_metrics.half_life,
                "current_zscore": float(spread_metrics.current_zscore),
                "z_score_range": z_range
            }
        }