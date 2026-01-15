# Replicating Cain's "The Adstock Illusion" Methodology

This implementation replicates the key findings from:
**P.M. Cain (2025): "Long-term advertising effects: The Adstock illusion"**
*Applied Marketing Analytics, Vol. 11, No. 1, pp. 23-42*

## The Problem

Marketing Mix Models (MMMs) commonly use "dual-adstock" approaches to measure both:
- **Short-term activation** (low retention rate, e.g., 0.30)
- **Long-term brand-building** (high retention rate, e.g., 0.99)

**Cain demonstrates this approach is fundamentally flawed:**
1. It treats brand-building as just a stretched version of activation (same mechanism)
2. High retention adstock creates a deterministic trend, leading to spurious regression
3. It cannot distinguish true brand-building from statistical artifacts

## The Solution

Cain proposes three proper approaches:

### 1. Unobserved Components Model (UCM)
- Decomposes sales into: stochastic trend + short-term effects + seasonality
- Brand-building = advertising driving the trend component
- Activation = temporary deviations from trend

### 2. Vector Autoregression (VAR)
- Tests whether effects decay to zero (activation only) or persist (brand-building)
- Uses Impulse Response Functions (IRFs) with proper confidence intervals
- Requires testing for stationarity/cointegration first

### 3. Combined UCM + VAR
- UCM extracts the trend
- VAR explains trend evolution using brand metrics + advertising
- Provides theoretical grounding (memory structures, emotional engagement)

## Key Results from This Implementation

### Dual-Adstock Problems (Scenario 1 - With Long-Term Effects):
```
R²: 0.7493
DW Statistic: 0.0116 (should be ~2.0)  ← MAJOR RED FLAG
Ljung-Box p-value: 0.0000 (want >0.05) ← SEVERE AUTOCORRELATION
```
**Interpretation:** The model appears to fit well (R² = 0.75) but has terrible diagnostics. 
The DW statistic of 0.0116 indicates extreme positive autocorrelation - a classic sign of 
spurious regression.

### Dual-Adstock Problems (Scenario 2 - WITHOUT Long-Term Effects):
```
R²: 0.4534
DW Statistic: 0.1138  ← STILL PROBLEMATIC
Ljung-Box p-value: 0.0000
Short-term TV coefficient: 0.0002
Long-term TV coefficient: 0.0000
```
**Interpretation:** Even when there's NO true long-term effect in the data, the dual-adstock 
model shows spurious patterns and terrible diagnostics.

### VAR Results:
- Properly differenced the non-stationary series
- Impulse Response Functions show whether effects persist or decay
- Confidence bands provide proper inference

## Files Generated

1. **cain_mmm_replication.py** - Complete implementation (800+ lines)
2. **summary_comparison.png** - Side-by-side comparison of all approaches
3. **adstock_comparison.png** - How different retention rates transform data
4. **dual_adstock_decomposition.png** - The problematic "ladder chart" patterns
5. **dual_adstock_diagnostics.png** - Shows the spurious regression problem
6. **ucm_components.png** - Proper decomposition into trend + activation
7. **var_irf.png** - Impulse response functions with confidence bands

## How to Use

```python
# Basic usage
from cain_mmm_replication import MarketingDataSimulator, DualAdstockModel, \
                                   UnobservedComponentsMMM, VARModel

# 1. Generate or load your data
simulator = MarketingDataSimulator(n_periods=156)
data = simulator.generate_data(has_long_term_effect=True)

# 2. Fit dual-adstock (what Cain critiques)
dual_model = DualAdstockModel(data)
results = dual_model.fit(short_retention=0.3, long_retention=0.99)
print(f"DW Statistic: {results['dw_statistic']:.4f}")  # Should be ~2.0
if results['dw_statistic'] < 1.5:
    print("WARNING: Likely spurious regression!")

# 3. Fit UCM (Cain recommends)
ucm = UnobservedComponentsMMM(data)
ucm_results = ucm.fit(short_retention=0.3)
ucm.plot_components()  # Shows trend vs activation

# 4. Fit VAR for persistent effects
var = VARModel(data)
var_results = var.fit(maxlags=4)
var.plot_irf()  # Shows if effects decay or persist
```

## Implementation Details

### What's Fully Implemented:
✅ Dual-adstock OLS with diagnostics
✅ Geometric adstock transformations
✅ Data simulation with/without long-term effects
✅ VAR with unit root testing (ADF)
✅ Impulse Response Functions (IRFs)
✅ Durbin-Watson and Ljung-Box tests
✅ UCM with trend extraction (simplified version)
✅ Comprehensive visualizations

### What's Simplified:
⚠️ UCM uses simplified trend extraction (Savitzky-Golay filter fallback)
⚠️ Brand metrics integration not implemented (conceptual only)
⚠️ Combined UCM+VAR approach shown conceptually
⚠️ Cointegration testing present but not fully integrated

## Dependencies

```bash
pip install numpy pandas matplotlib seaborn scipy statsmodels scikit-learn
```

## Key Insights for Practitioners

1. **Check your diagnostics:** Low DW statistics (<1.5) indicate spurious regression
2. **Test for stationarity:** Use ADF tests before modeling
3. **Separate mechanisms:** Don't use the same distributed lag for activation and brand-building
4. **Use IRFs:** They show whether effects truly persist or just decay slowly
5. **Model the trend:** Explicitly model base sales evolution rather than approximating with high-retention adstock

## Academic vs. Commercial Practice

**Academic consensus:** Long-term effects require persistent changes in the data-generating process

**Commercial MMM practice:** Often uses dual-adstock with retention rates like 0.3/0.99

**The gap:** Dual-adstock is convenient but statistically flawed. This implementation shows why.

## Further Reading

- Cain (2025): "Long-term advertising effects: The Adstock illusion"
- Dekimpe & Hanssens (1995): "The persistence of marketing effects on sales"
- Harvey (1989): "Forecasting, Structural Time Series Models and the Kalman Filter"
- Sims (1980): "Macroeconomics and Reality" (VAR methodology)

## Citation

If you use this code, please cite:
```
Cain, P.M. (2025). Long-term advertising effects: The Adstock illusion. 
Applied Marketing Analytics, 11(1), 23-42.
```

## License

This implementation is for educational purposes, replicating published academic research.

---
**Date:** January 2026  
