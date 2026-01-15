"""
Long-Term Marketing Effects: Detection and Alternative Approaches

This module implements methods from:
Cain, P.M. (2025). "Long-term advertising effects: The Adstock illusion."
Applied Marketing Analytics, 11(1), 23-42.

Includes:
1. Dual-adstock diagnostics (detect spurious regression)
2. Unobserved Components Model (UCM) - with caveats
3. Vector Autoregression (VAR) with IRFs - with caveats
4. Combined UCM+VAR approach - conceptual

WARNING: None of these approaches are perfect. Long-term effects are
fundamentally hard to measure. Use with appropriate skepticism.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf
import warnings
warnings.filterwarnings('ignore')


def create_adstock(x, retention_rate):
    """
    Create geometric adstock transformation.
    
    Parameters
    ----------
    x : array-like
        Input series (e.g., TV GRPs)
    retention_rate : float
        Retention rate between 0 and 1
        
    Returns
    -------
    adstock : np.ndarray
        Adstocked series
        
    Notes
    -----
    Half-life = ln(0.5) / ln(retention_rate)
    - λ=0.30 → half-life ≈ 1.4 weeks
    - λ=0.70 → half-life ≈ 2.0 weeks
    - λ=0.99 → half-life ≈ 69 weeks (problematic!)
    """
    adstock = np.zeros_like(x, dtype=float)
    adstock[0] = x[0]
    
    for t in range(1, len(x)):
        adstock[t] = x[t] + retention_rate * adstock[t-1]
    
    return adstock


def fit_dual_adstock_model(sales, tv, short_retention=0.3, long_retention=0.99, 
                           include_controls=None):
    """
    Fit dual-adstock model and return full diagnostics.
    
    Parameters
    ----------
    sales : array-like
        Sales data (recommend log-transformed)
    tv : array-like
        TV advertising data (GRPs, impressions, etc.)
    short_retention : float, default=0.3
        Retention rate for short-term adstock
    long_retention : float, default=0.99
        Retention rate for long-term adstock
    include_controls : array-like, optional
        Additional control variables (e.g., price, seasonality)
        
    Returns
    -------
    results : dict
        Dictionary containing:
        - 'coefficients': Model coefficients
        - 'r2': R-squared
        - 'dw_statistic': Durbin-Watson statistic
        - 'ljung_box_pvalue': Ljung-Box test p-value
        - 'residuals': Model residuals
        - 'fitted': Fitted values
        - 'is_spurious': Boolean flag if diagnostics indicate spurious regression
        - 'warning_message': Human-readable warning if spurious
    """
    # Create adstock transformations
    tv_short = create_adstock(tv, short_retention)
    tv_long = create_adstock(tv, long_retention)
    
    # Build design matrix
    X = np.column_stack([np.ones(len(sales)), tv_short, tv_long])
    
    if include_controls is not None:
        X = np.column_stack([X, include_controls])
    
    # Fit OLS
    model = LinearRegression(fit_intercept=False)
    model.fit(X, sales)
    
    fitted = model.predict(X)
    residuals = sales - fitted
    
    # Calculate diagnostics
    r2 = 1 - np.var(residuals) / np.var(sales)
    
    # Durbin-Watson statistic
    dw_stat = np.sum(np.diff(residuals)**2) / np.sum(residuals**2)
    
    # Ljung-Box test for autocorrelation
    lb_test = acorr_ljungbox(residuals, lags=min(10, len(residuals)//5), return_df=True)
    lb_pvalue = lb_test['lb_pvalue'].iloc[-1]
    
    # Check for spurious regression
    is_spurious = (dw_stat < 1.5) or (lb_pvalue < 0.05)
    
    warning_msg = None
    if is_spurious:
        warning_msg = (
            f"⚠️  SPURIOUS REGRESSION DETECTED:\n"
            f"   Durbin-Watson = {dw_stat:.3f} (should be ~2.0)\n"
            f"   Ljung-Box p-value = {lb_pvalue:.4f} (should be >0.05)\n"
            f"   \n"
            f"   The 'long-term effect' may be a statistical artifact,\n"
            f"   not a real brand-building impact."
        )
    
    # Extract coefficients
    coef_dict = {
        'intercept': model.coef_[0],
        'tv_short': model.coef_[1],
        'tv_long': model.coef_[2]
    }
    
    if include_controls is not None:
        for i in range(include_controls.shape[1]):
            coef_dict[f'control_{i+1}'] = model.coef_[3+i]
    
    return {
        'coefficients': coef_dict,
        'r2': r2,
        'dw_statistic': dw_stat,
        'ljung_box_pvalue': lb_pvalue,
        'residuals': residuals,
        'fitted': fitted,
        'tv_short_adstock': tv_short,
        'tv_long_adstock': tv_long,
        'is_spurious': is_spurious,
        'warning_message': warning_msg
    }


def plot_diagnostic_dashboard(sales, tv, results, title="Dual-Adstock Diagnostics"):
    """
    Create comprehensive diagnostic dashboard.
    
    Parameters
    ----------
    sales : array-like
        Original sales data
    tv : array-like
        Original TV data
    results : dict
        Results from fit_dual_adstock_model()
    title : str
        Plot title
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The diagnostic figure
    """
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Row 1: Data and fit
    # Panel 1: Actual vs Fitted
    axes[0, 0].scatter(sales, results['fitted'], alpha=0.5, s=30)
    axes[0, 0].plot([sales.min(), sales.max()], 
                    [sales.min(), sales.max()], 
                    'r--', linewidth=2, alpha=0.7)
    axes[0, 0].set_xlabel('Actual Sales', fontsize=10)
    axes[0, 0].set_ylabel('Fitted Sales', fontsize=10)
    axes[0, 0].set_title(f'Actual vs Fitted (R² = {results["r2"]:.3f})', fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Panel 2: Time series with fit
    axes[0, 1].plot(sales, label='Actual', linewidth=2, alpha=0.7)
    axes[0, 1].plot(results['fitted'], label='Fitted', linewidth=2, alpha=0.7)
    axes[0, 1].set_xlabel('Time Period', fontsize=10)
    axes[0, 1].set_ylabel('Sales', fontsize=10)
    axes[0, 1].set_title('Sales: Actual vs Fitted', fontsize=11)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Panel 3: TV campaigns
    axes[0, 2].bar(range(len(tv)), tv, alpha=0.6, color='orange')
    axes[0, 2].set_xlabel('Time Period', fontsize=10)
    axes[0, 2].set_ylabel('TV GRPs', fontsize=10)
    axes[0, 2].set_title('TV Advertising Schedule', fontsize=11)
    axes[0, 2].grid(True, alpha=0.3)
    
    # Row 2: Adstock transformations
    # Panel 4: Short-term adstock
    axes[1, 0].plot(results['tv_short_adstock'], color='green', linewidth=2)
    axes[1, 0].set_xlabel('Time Period', fontsize=10)
    axes[1, 0].set_ylabel('Adstock Value', fontsize=10)
    axes[1, 0].set_title('Short-Term Adstock (λ=0.30)', fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Panel 5: Long-term adstock
    axes[1, 1].plot(results['tv_long_adstock'], color='red', linewidth=2)
    axes[1, 1].set_xlabel('Time Period', fontsize=10)
    axes[1, 1].set_ylabel('Adstock Value', fontsize=10)
    axes[1, 1].set_title('Long-Term Adstock (λ=0.99)', fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].text(0.5, 0.95, 'Note: Accumulates\nlike a trend', 
                   transform=axes[1, 1].transAxes, ha='center', va='top',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6),
                   fontsize=9)
    
    # Panel 6: Coefficients
    coef_names = ['Short-term\n(λ=0.30)', 'Long-term\n(λ=0.99)']
    coef_values = [results['coefficients']['tv_short'], 
                   results['coefficients']['tv_long']]
    colors = ['green', 'red']
    bars = axes[1, 2].bar(coef_names, coef_values, color=colors, alpha=0.7, 
                          edgecolor='black', linewidth=2)
    axes[1, 2].set_ylabel('Coefficient', fontsize=10)
    axes[1, 2].set_title('Estimated Coefficients', fontsize=11)
    axes[1, 2].axhline(0, color='black', linewidth=0.5)
    axes[1, 2].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, coef_values):
        height = bar.get_height()
        axes[1, 2].text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.4f}', ha='center', 
                       va='bottom' if val > 0 else 'top',
                       fontsize=9, fontweight='bold')
    
    # Row 3: DIAGNOSTIC TESTS (THE CRITICAL PART)
    # Panel 7: Durbin-Watson
    dw_stat = results['dw_statistic']
    dw_color = 'red' if dw_stat < 1.5 else ('orange' if dw_stat < 1.8 else 'green')
    
    axes[2, 0].bar(['Model DW', 'Ideal DW'], [dw_stat, 2.0], 
                   color=[dw_color, 'green'], alpha=0.7, 
                   edgecolor='black', linewidth=2)
    axes[2, 0].axhline(2.0, color='green', linestyle='--', linewidth=2, alpha=0.5)
    axes[2, 0].axhline(1.5, color='orange', linestyle='--', linewidth=1, alpha=0.5)
    axes[2, 0].set_ylim([0, 2.5])
    axes[2, 0].set_ylabel('Durbin-Watson Statistic', fontsize=10)
    
    # Add judgment
    if dw_stat < 1.5:
        axes[2, 0].set_title(f'❌ FAIL: DW = {dw_stat:.3f}', 
                            fontsize=11, fontweight='bold', color='red')
        axes[2, 0].text(0.5, 0.6, 'SPURIOUS\nREGRESSION\nLIKELY', 
                       transform=axes[2, 0].transAxes, ha='center', va='center',
                       fontsize=10, fontweight='bold', color='red',
                       bbox=dict(boxstyle='round', facecolor='white', 
                                edgecolor='red', linewidth=2))
    else:
        axes[2, 0].set_title(f'✓ PASS: DW = {dw_stat:.3f}', 
                            fontsize=11, fontweight='bold', color='green')
    
    axes[2, 0].grid(True, alpha=0.3, axis='y')
    
    # Panel 8: Autocorrelation function
    plot_acf(results['residuals'], lags=min(20, len(results['residuals'])//5), 
             ax=axes[2, 1], alpha=0.05)
    lb_pval = results['ljung_box_pvalue']
    lb_color = 'red' if lb_pval < 0.05 else 'green'
    
    if lb_pval < 0.05:
        axes[2, 1].set_title(f'❌ FAIL: Ljung-Box p={lb_pval:.4f}', 
                           fontsize=11, fontweight='bold', color='red')
    else:
        axes[2, 1].set_title(f'✓ PASS: Ljung-Box p={lb_pval:.4f}', 
                           fontsize=11, fontweight='bold', color='green')
    
    axes[2, 1].set_xlabel('Lag', fontsize=10)
    axes[2, 1].set_ylabel('ACF', fontsize=10)
    
    # Panel 9: Residual plot
    axes[2, 2].plot(results['residuals'], linewidth=1, alpha=0.7, color=dw_color)
    axes[2, 2].axhline(0, color='black', linestyle='--', linewidth=1)
    axes[2, 2].fill_between(range(len(results['residuals'])), 
                           results['residuals'], 0, alpha=0.3, color=dw_color)
    axes[2, 2].set_xlabel('Time Period', fontsize=10)
    axes[2, 2].set_ylabel('Residual', fontsize=10)
    axes[2, 2].set_title('Residuals Over Time', fontsize=11)
    axes[2, 2].grid(True, alpha=0.3)
    
    # Add verdict text box
    if results['is_spurious']:
        axes[2, 2].text(0.5, 0.95, 'Residuals show\nsystematic pattern\n(should be noise)', 
                       transform=axes[2, 2].transAxes, ha='center', va='top',
                       fontsize=9, color='red', fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='white', 
                                edgecolor='red', linewidth=2))
    
    plt.tight_layout()
    return fig


def create_comparison_table(results):
    """
    Create a formatted comparison table of results.
    
    Parameters
    ----------
    results : dict
        Results from fit_dual_adstock_model()
        
    Returns
    -------
    df : pd.DataFrame
        Formatted results table
    """
    data = {
        'Metric': [
            'R²',
            'Short-term coefficient',
            'Long-term coefficient',
            'Durbin-Watson',
            'Ljung-Box p-value',
            'Spurious regression?'
        ],
        'Value': [
            f"{results['r2']:.4f}",
            f"{results['coefficients']['tv_short']:.4f}",
            f"{results['coefficients']['tv_long']:.4f}",
            f"{results['dw_statistic']:.4f}",
            f"{results['ljung_box_pvalue']:.4f}",
            '⚠️ YES' if results['is_spurious'] else '✓ NO'
        ],
        'Interpretation': [
            'Higher is better (but can be misleading)',
            'Short-term activation effect',
            'Claimed long-term effect',
            'Should be ~2.0 (below 1.5 = problem)',
            'Should be >0.05 (below = autocorrelation)',
            'Diagnostics indicate spurious correlation'
        ]
    }
    
    df = pd.DataFrame(data)
    return df


def simulate_marketing_data(n_periods=156, has_true_long_term=False, seed=42):
    """
    Simulate marketing data for testing (ground truth known).
    
    Parameters
    ----------
    n_periods : int
        Number of time periods
    has_true_long_term : bool
        Whether to include true long-term TV effect
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    data : pd.DataFrame
        Simulated data with columns: sales, tv, base, activation
    """
    np.random.seed(seed)
    
    # TV campaigns (sporadic bursts)
    tv = np.zeros(n_periods)
    n_campaigns = max(8, n_periods // 20)
    campaign_weeks = np.random.choice(n_periods, n_campaigns, replace=False)
    
    for start in sorted(campaign_weeks):
        duration = np.random.randint(3, 6)
        base_grp = np.random.uniform(80, 150)
        for i in range(duration):
            if start + i < n_periods:
                tv[start + i] = base_grp * np.random.uniform(0.9, 1.1)
    
    # Base sales evolution
    base = np.zeros(n_periods)
    base[0] = 1000
    
    if has_true_long_term:
        # TV builds the base (true long-term effect)
        cumulative_tv = 0
        for t in range(1, n_periods):
            cumulative_tv = 0.995 * cumulative_tv + 0.01 * tv[t]
            base[t] = base[t-1] + cumulative_tv + np.random.normal(0, 10)
    else:
        # Base just drifts (NO TV effect)
        for t in range(1, n_periods):
            base[t] = base[t-1] + np.random.normal(0, 10)
    
    # Short-term activation
    activation = np.zeros(n_periods)
    for t in range(n_periods):
        for lag in range(5):
            if t - lag >= 0:
                activation[t] += 0.5 * tv[t-lag] * (0.7 ** lag)
    
    # Total sales
    seasonality = 80 * np.sin(2 * np.pi * np.arange(n_periods) / 52)
    sales = base + activation + seasonality + np.random.normal(0, 15, n_periods)
    
    return pd.DataFrame({
        'sales': sales,
        'tv': tv,
        'base': base,
        'activation': activation
    })


def check_your_model(sales, tv, short_retention=0.3, long_retention=0.99):
    """
    Quick check: Is your dual-adstock model spurious?
    
    Parameters
    ----------
    sales : array-like
        Your sales data (preferably log-transformed)
    tv : array-like
        Your TV advertising data
    short_retention : float
        Short-term retention rate (typically 0.3-0.5)
    long_retention : float
        Long-term retention rate (typically 0.95-0.99)
        
    Returns
    -------
    verdict : str
        Plain English verdict
    """
    results = fit_dual_adstock_model(sales, tv, short_retention, long_retention)
    
    print("=" * 70)
    print("DUAL-ADSTOCK MODEL DIAGNOSTIC CHECK")
    print("=" * 70)
    print(f"\nR² = {results['r2']:.4f}")
    print(f"Short-term coefficient = {results['coefficients']['tv_short']:.4f}")
    print(f"Long-term coefficient = {results['coefficients']['tv_long']:.4f}")
    print(f"\nDurbin-Watson statistic = {results['dw_statistic']:.4f}")
    print(f"Ljung-Box p-value = {results['ljung_box_pvalue']:.6f}")
    
    print("\n" + "=" * 70)
    print("VERDICT:")
    print("=" * 70)
    
    if results['is_spurious']:
        print("❌ SPURIOUS REGRESSION DETECTED")
        print("\nYour model shows signs of spurious correlation:")
        if results['dw_statistic'] < 1.5:
            print(f"  • Durbin-Watson ({results['dw_statistic']:.3f}) is below 1.5")
        if results['ljung_box_pvalue'] < 0.05:
            print(f"  • Ljung-Box test rejects white noise (p={results['ljung_box_pvalue']:.4f})")
        print("\nThe claimed 'long-term effect' may be a statistical artifact.")
        print("The high-retention adstock is likely just correlating with trend.")
    else:
        print("✓ NO OBVIOUS SPURIOUS REGRESSION")
        print("\nDiagnostics look acceptable:")
        print(f"  • Durbin-Watson ({results['dw_statistic']:.3f}) is reasonable")
        print(f"  • Ljung-Box test passes (p={results['ljung_box_pvalue']:.4f})")
        print("\nHowever, consider other validation methods as well.")
    
    print("=" * 70)
    
    return results


# ============================================================================
# ALTERNATIVE APPROACH 1: UNOBSERVED COMPONENTS MODEL (UCM)
# ============================================================================

def fit_ucm_model(sales, tv, short_retention=0.3, include_controls=None):
    """
    Fit Unobserved Components Model (Cain's alternative to dual-adstock).
    
    CAVEAT: This approach is better than dual-adstock but still has limitations:
    - Requires expertise with state-space models
    - Sensitive to model specification
    - No consensus on how advertising should enter the trend
    - Assumes trend can be cleanly separated
    
    Parameters
    ----------
    sales : array-like
        Sales data (log-transformed recommended)
    tv : array-like
        TV advertising data
    short_retention : float
        Retention rate for short-term adstock (typically 0.3-0.5)
    include_controls : array-like, optional
        Additional control variables
        
    Returns
    -------
    results : dict
        Dictionary with trend, coefficients, diagnostics
    """
    from statsmodels.tsa.statespace.structural import UnobservedComponents
    from scipy.signal import savgol_filter
    
    # Create short-term adstock only
    tv_short = create_adstock(tv, short_retention)
    
    # Prepare exogenous variables
    exog_data = pd.DataFrame({'tv': tv_short})
    if include_controls is not None:
        for i in range(include_controls.shape[1]):
            exog_data[f'control_{i}'] = include_controls[:, i]
    
    try:
        # Fit UCM with local level trend
        model = UnobservedComponents(
            sales,
            level='local level',  # Stochastic trend
            exog=exog_data
        )
        
        fitted_model = model.fit(disp=False, maxiter=1000, method='lbfgs')
        
        # Extract components
        trend = fitted_model.level.smoothed.values if hasattr(fitted_model.level.smoothed, 'values') else np.array(fitted_model.level.smoothed)
        fitted_values = fitted_model.fittedvalues.values if hasattr(fitted_model.fittedvalues, 'values') else np.array(fitted_model.fittedvalues)
        residuals = sales - fitted_values
        
        # Get coefficients
        params = fitted_model.params
        param_names = list(params.index)
        
        coefficients = {}
        for name in param_names:
            if 'beta' in name:
                coefficients[name] = params[name]
        
        # Calculate R-squared
        r2 = 1 - np.var(residuals) / np.var(sales)
        
        # Ljung-Box test
        lb_test = acorr_ljungbox(residuals, lags=min(10, len(residuals)//5), return_df=True)
        lb_pvalue = lb_test['lb_pvalue'].iloc[-1]
        
        return {
            'trend': trend,
            'coefficients': coefficients,
            'r2': r2,
            'ljung_box_pvalue': lb_pvalue,
            'residuals': residuals,
            'fitted': fitted_values,
            'tv_short_adstock': tv_short,
            'model': fitted_model,
            'success': True,
            'method': 'Kalman filter'
        }
        
    except Exception as e:
        print(f"⚠️  UCM Kalman filter failed: {e}")
        print(f"   Falling back to smoothing filter (less reliable)...")
        
        # Fallback: use Savitzky-Golay filter
        window_length = min(51, len(sales)//3*2+1)
        if window_length % 2 == 0:
            window_length += 1
        
        trend = savgol_filter(sales, window_length=window_length, polyorder=3)
        
        # Estimate TV effect on detrended sales
        detrended = sales - trend
        X = np.column_stack([np.ones(len(tv_short)), tv_short])
        from sklearn.linear_model import LinearRegression
        model = LinearRegression(fit_intercept=False)
        model.fit(X, detrended)
        
        fitted_values = trend + model.predict(X)
        residuals = sales - fitted_values
        r2 = 1 - np.var(residuals) / np.var(sales)
        
        lb_test = acorr_ljungbox(residuals, lags=min(10, len(residuals)//5), return_df=True)
        lb_pvalue = lb_test['lb_pvalue'].iloc[-1]
        
        return {
            'trend': trend,
            'coefficients': {'tv_short': model.coef_[1]},
            'r2': r2,
            'ljung_box_pvalue': lb_pvalue,
            'residuals': residuals,
            'fitted': fitted_values,
            'tv_short_adstock': tv_short,
            'model': None,
            'success': False,
            'method': 'Savitzky-Golay filter (fallback)'
        }


def plot_ucm_decomposition(sales, tv, ucm_results, title="UCM Decomposition"):
    """
    Plot UCM decomposition into trend and transitory components.
    
    Parameters
    ----------
    sales : array-like
        Original sales data
    tv : array-like
        Original TV data
    ucm_results : dict
        Results from fit_ucm_model()
    title : str
        Plot title
        
    Returns
    -------
    fig : matplotlib figure
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    
    time = np.arange(len(sales))
    
    # Panel 1: TV campaigns
    axes[0].bar(time, tv, alpha=0.6, color='orange')
    axes[0].set_title('TV Advertising Schedule', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('TV GRPs', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Panel 2: Sales vs Fitted
    axes[1].plot(time, sales, label='Actual Sales', linewidth=2, color='black')
    axes[1].plot(time, ucm_results['fitted'], label='Fitted', linewidth=2, color='red', alpha=0.7)
    axes[1].set_title(f'Sales vs Fitted (R² = {ucm_results["r2"]:.3f})', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Sales', fontsize=10)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Panel 3: Trend component (long-term)
    axes[2].plot(time, ucm_results['trend'], linewidth=2, color='blue')
    axes[2].set_title(f'Trend Component (Base Sales) - Method: {ucm_results["method"]}', 
                     fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Log Sales', fontsize=10)
    axes[2].grid(True, alpha=0.3)
    axes[2].text(0.02, 0.95, 'This captures long-term evolution\n(NOT necessarily driven by TV)',
                transform=axes[2].transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6))
    
    # Panel 4: Residuals
    axes[3].plot(time, ucm_results['residuals'], linewidth=1, color='red', alpha=0.7)
    axes[3].axhline(0, color='black', linestyle='--', linewidth=1)
    axes[3].fill_between(time, ucm_results['residuals'], 0, alpha=0.3, color='red')
    axes[3].set_title(f'Residuals (Ljung-Box p = {ucm_results["ljung_box_pvalue"]:.4f})', 
                     fontsize=12, fontweight='bold')
    axes[3].set_ylabel('Residual', fontsize=10)
    axes[3].set_xlabel('Time Period', fontsize=10)
    axes[3].grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


# ============================================================================
# ALTERNATIVE APPROACH 2: VECTOR AUTOREGRESSION (VAR)
# ============================================================================

def fit_var_model(sales, tv, maxlags=4):
    """
    Fit Vector Autoregression model with IRF analysis.
    
    CAVEAT: This approach is theoretically sound but has limitations:
    - Requires substantial data (loses degrees of freedom)
    - Sensitive to lag length choice
    - IRFs can be unstable with small samples
    - Cointegration testing can be ambiguous
    
    Parameters
    ----------
    sales : array-like
        Sales data (will test for stationarity)
    tv : array-like
        TV advertising data
    maxlags : int
        Maximum number of lags to include
        
    Returns
    -------
    results : dict
        Dictionary with VAR results and IRFs
    """
    from statsmodels.tsa.vector_ar.var_model import VAR
    from statsmodels.tsa.stattools import adfuller
    
    # Test for stationarity
    sales_adf = adfuller(sales, maxlag=4, regression='c')
    tv_adf = adfuller(tv, maxlag=4, regression='c')
    
    sales_stationary = sales_adf[1] < 0.05
    tv_stationary = tv_adf[1] < 0.05
    
    # If non-stationary, difference
    if not sales_stationary or not tv_stationary:
        print(f"⚠️  Non-stationary series detected:")
        print(f"   Sales ADF p-value: {sales_adf[1]:.4f}")
        print(f"   TV ADF p-value: {tv_adf[1]:.4f}")
        print(f"   Differencing the data...")
        
        data_for_var = pd.DataFrame({
            'sales': np.diff(sales),
            'tv': np.diff(tv)
        })
        is_differenced = True
    else:
        data_for_var = pd.DataFrame({
            'sales': sales,
            'tv': tv
        })
        is_differenced = False
    
    # Fit VAR
    model = VAR(data_for_var)
    var_results = model.fit(maxlags=maxlags)
    
    # Compute impulse response function
    irf = var_results.irf(periods=50)
    
    # Extract IRF of sales to TV shock
    irf_sales_to_tv = irf.irfs[:, 0, 1]  # Sales response to TV
    irf_stderr = irf.stderr()[:, 0, 1] if hasattr(irf, 'stderr') else None
    
    # Cumulative IRF
    cumulative_irf = np.cumsum(irf_sales_to_tv)
    
    return {
        'var_results': var_results,
        'irf': irf,
        'irf_sales_to_tv': irf_sales_to_tv,
        'irf_stderr': irf_stderr,
        'cumulative_irf': cumulative_irf,
        'is_differenced': is_differenced,
        'sales_adf_pvalue': sales_adf[1],
        'tv_adf_pvalue': tv_adf[1],
        'sales_stationary': sales_stationary,
        'tv_stationary': tv_stationary
    }


def plot_var_irf(var_results, title="VAR Impulse Response Functions"):
    """
    Plot VAR impulse response functions.
    
    Parameters
    ----------
    var_results : dict
        Results from fit_var_model()
    title : str
        Plot title
        
    Returns
    -------
    fig : matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    periods = np.arange(len(var_results['irf_sales_to_tv']))
    irf_sales = var_results['irf_sales_to_tv']
    cumulative = var_results['cumulative_irf']
    
    # Add confidence bands if available
    if var_results['irf_stderr'] is not None:
        irf_lower = irf_sales - 2 * var_results['irf_stderr']
        irf_upper = irf_sales + 2 * var_results['irf_stderr']
    else:
        irf_lower = irf_upper = None
    
    # Panel 1: IRF
    axes[0].plot(periods, irf_sales, linewidth=2, color='blue', label='IRF')
    if irf_lower is not None:
        axes[0].fill_between(periods, irf_lower, irf_upper, alpha=0.2, color='blue')
    axes[0].axhline(0, color='red', linestyle='--', alpha=0.5)
    axes[0].set_title('Response of Sales to TV Shock', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Periods After Shock', fontsize=10)
    axes[0].set_ylabel('Response', fontsize=10)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Panel 2: Cumulative IRF
    axes[1].plot(periods, cumulative, linewidth=2, color='darkblue', label='Cumulative IRF')
    axes[1].axhline(0, color='red', linestyle='--', alpha=0.5)
    axes[1].set_title('Cumulative Response (Long-Term Effect)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Periods After Shock', fontsize=10)
    axes[1].set_ylabel('Cumulative Response', fontsize=10)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Add interpretation
    long_term_effect = cumulative[-1]
    if var_results['is_differenced']:
        interpretation = f"Differenced data\nLong-term effect: {long_term_effect:.4f}"
    else:
        interpretation = f"Levels data\nLong-term effect: {long_term_effect:.4f}"
    
    axes[1].text(0.05, 0.95, interpretation,
                transform=axes[1].transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


# ============================================================================
# ALTERNATIVE APPROACH 3: COMBINED UCM + VAR (CONCEPTUAL)
# ============================================================================

def fit_combined_approach(sales, tv, brand_metric=None, short_retention=0.3, maxlags=4):
    """
    Combined UCM + VAR approach (Cain's most sophisticated method).
    
    MAJOR CAVEAT: This is the most theoretically sound but also:
    - Most complex to implement
    - Requires brand survey data (often unavailable)
    - Two-step estimation compounds uncertainty
    - Very few practitioners have done this successfully
    
    This implementation is CONCEPTUAL - not production-ready.
    
    Parameters
    ----------
    sales : array-like
        Sales data
    tv : array-like
        TV advertising
    brand_metric : array-like, optional
        Brand perception survey data (if available)
    short_retention : float
        Short-term adstock retention
    maxlags : int
        VAR lag length
        
    Returns
    -------
    results : dict
        Conceptual results (not fully validated)
    """
    print("=" * 70)
    print("COMBINED UCM + VAR APPROACH (CONCEPTUAL)")
    print("=" * 70)
    print()
    print("⚠️  WARNING: This is a conceptual implementation.")
    print("   Requires:")
    print("   1. Brand survey data (awareness, consideration, preference)")
    print("   2. Cointegration between base sales and brand metrics")
    print("   3. Sufficient data for two-step estimation")
    print()
    
    # Step 1: Extract trend with UCM
    print("Step 1: Extracting base sales trend with UCM...")
    ucm_results = fit_ucm_model(sales, tv, short_retention)
    trend = ucm_results['trend']
    print(f"   ✓ Trend extracted (method: {ucm_results['method']})")
    print()
    
    # Step 2: VAR with trend and brand metrics
    print("Step 2: Modeling trend with VAR...")
    
    if brand_metric is not None:
        print("   Using provided brand metric...")
        var_data = pd.DataFrame({
            'trend': trend,
            'brand_metric': brand_metric,
            'tv': tv
        })
    else:
        print("   ⚠️  No brand metric provided. Using trend and TV only.")
        print("      (This defeats the purpose of the combined approach!)")
        var_data = pd.DataFrame({
            'trend': trend,
            'tv': tv
        })
    
    # Check stationarity
    from statsmodels.tsa.stattools import adfuller
    trend_adf = adfuller(trend, maxlag=4)
    
    if trend_adf[1] > 0.05:
        print(f"   Trend is non-stationary (p={trend_adf[1]:.4f})")
        print("   Should test for cointegration with brand metrics...")
        print("   (Not implemented in this conceptual version)")
    
    # Fit VAR (simplified)
    from statsmodels.tsa.vector_ar.var_model import VAR
    model = VAR(var_data)
    
    try:
        var_results = model.fit(maxlags=maxlags)
        print(f"   ✓ VAR fitted with {maxlags} lags")
    except Exception as e:
        print(f"   ✗ VAR fitting failed: {e}")
        var_results = None
    
    print()
    print("=" * 70)
    print("CAVEAT: This combined approach is HIGHLY EXPERIMENTAL")
    print("=" * 70)
    print("In practice, you would need:")
    print("1. Regular brand tracking surveys (expensive)")
    print("2. Johansen cointegration tests")
    print("3. Structural identification of the VAR")
    print("4. Validation against holdout data")
    print()
    print("Very few organizations have successfully implemented this.")
    print("=" * 70)
    
    return {
        'ucm_results': ucm_results,
        'var_results': var_results,
        'trend': trend,
        'has_brand_metric': brand_metric is not None,
        'note': 'Conceptual implementation - requires substantial additional work'
    }


# ============================================================================
# CONVENIENCE FUNCTION: CHECK YOUR MODEL
# ============================================================================

def check_dual_adstock_diagnostics(sales, tv, short_retention=0.3, long_retention=0.99):
    """
    Quick diagnostic check for dual-adstock spurious regression.
    
    This is an alias for check_your_model() with clearer naming.
    """
    return check_your_model(sales, tv, short_retention, long_retention)
