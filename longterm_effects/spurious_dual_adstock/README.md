# MMM Diagnostics: Detecting Spurious Regression in Dual-Adstock Models

**A practical toolkit for checking if your Marketing Mix Models are statistically valid.**

Based on: Cain, P.M. (2025). "Long-term advertising effects: The Adstock illusion." *Applied Marketing Analytics*, 11(1), 23-42.

## Quick Start

### 1. Check Your Model (30 seconds)

```python
from mmm_diagnostics import check_your_model

# Your data
sales = ...  # Your sales data (log-transform recommended)
tv = ...     # Your TV advertising data

# Quick diagnostic
results = check_your_model(sales, tv, short_retention=0.3, long_retention=0.99)
```

### 2. Full Diagnostic Dashboard

```python
from mmm_diagnostics import fit_dual_adstock_model, plot_diagnostic_dashboard

# Fit model
results = fit_dual_adstock_model(sales, tv)

# Create diagnostic plots
fig = plot_diagnostic_dashboard(sales, tv, results)
```

### 3. Work Through the Demo Notebook

```bash
jupyter notebook dual_adstock_diagnostics_demo.ipynb
```

## What This Detects

### The Problem:
High-retention adstock (λ ≥ 0.95) used for "long-term effects" can produce **spurious regression**:
- Model finds effects that don't exist
- High-retention adstock accumulates like a trend
- Correlates with any trending component in sales
- Looks good on surface metrics (R², coefficients)
- Fails diagnostic tests (Durbin-Watson, Ljung-Box)

### Critical Thresholds:

| Diagnostic | Safe Zone | Warning Zone | Danger Zone |
|------------|-----------|--------------|-------------|
| Durbin-Watson | ≥ 1.8 | 1.5-1.8 | < 1.5 ❌ |
| Ljung-Box p-value | > 0.05 | 0.01-0.05 | < 0.01 ❌ |
| ACF plot | No pattern | Some spikes | Many spikes ❌ |

**If DW < 1.5 OR Ljung-Box p < 0.05:** Your model likely suffers from spurious regression.

## Files in This Package

1. **mmm_diagnostics.py** - Core diagnostic functions
   - `fit_dual_adstock_model()` - Fit and diagnose
   - `plot_diagnostic_dashboard()` - Comprehensive visualizations
   - `check_your_model()` - Quick diagnostic check
   - `simulate_marketing_data()` - Test data generation

2. **dual_adstock_diagnostics_demo.ipynb** - Interactive walkthrough
   - Controlled experiment with known ground truth
   - Shows how dual-adstock fails
   - Explains the diagnostics
   - Ready to share with colleagues

## Example Output

```
========================================================================
DUAL-ADSTOCK MODEL DIAGNOSTIC CHECK
========================================================================

R² = 0.314
Short-term coefficient = 0.6860
Long-term coefficient = -0.0211

Durbin-Watson statistic = 0.2469
Ljung-Box p-value = 0.000000

========================================================================
VERDICT:
========================================================================
❌ SPURIOUS REGRESSION DETECTED

Your model shows signs of spurious correlation:
  • Durbin-Watson (0.247) is below 1.5
  • Ljung-Box test rejects white noise (p=0.0000)

The claimed 'long-term effect' may be a statistical artifact.
The high-retention adstock is likely just correlating with trend.
========================================================================
```

## What This Package Does NOT Do

This package:
- ❌ Does NOT claim to have "the solution" to long-term measurement
- ❌ Does NOT recommend specific alternative approaches (all have tradeoffs)
- ❌ Does NOT validate your entire MMM methodology

This package:
- ✅ DOES help you detect a specific, well-documented problem
- ✅ DOES provide clear diagnostic criteria
- ✅ DOES give you evidence to challenge questionable claims

## Use Cases

### 1. Validate Your Own Models
```python
# Check your current MMM
results = check_your_model(my_sales, my_tv)

if results['is_spurious']:
    print("Need to revisit this model")
```

### 2. Challenge Vendor Reports
```python
# Replicate vendor's dual-adstock approach
vendor_results = fit_dual_adstock_model(
    sales=vendor_sales,
    tv=vendor_tv,
    short_retention=0.3,
    long_retention=0.99
)

# Show diagnostics
fig = plot_diagnostic_dashboard(vendor_sales, vendor_tv, vendor_results)

# Ask: "Why is your Durbin-Watson so low?"
```

### 3. Team Training
```python
# Run the demo notebook with your team
jupyter notebook dual_adstock_diagnostics_demo.ipynb

# Show them:
# 1. Data where we KNOW there's no long-term effect
# 2. Dual-adstock finds one anyway
# 3. Diagnostics reveal the problem
```

## Installation

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn statsmodels
```

## Key Takeaways

1. **Always check diagnostics** - R² and coefficients aren't enough
2. **Durbin-Watson < 1.5 is a red flag** - likely spurious regression
3. **High-retention adstock (λ > 0.95) is risky** - accumulates like trend
4. **Be skeptical of large long-term effects** - especially with poor diagnostics

## Limitations

This diagnostic package helps you identify **one specific problem**: spurious regression from high-retention adstock.

It does NOT:
- Prove your model is correct if diagnostics pass
- Recommend specific alternative methodologies
- Solve the general problem of long-term measurement

**Long-term marketing effects are hard to measure. No approach is perfect.**

This package helps you avoid one well-documented pitfall.

## Questions?

This is a practical implementation of published academic research. 

For the full theoretical treatment, see:
- Cain, P.M. (2025). "Long-term advertising effects: The Adstock illusion." *Applied Marketing Analytics*, 11(1), 23-42.

For questions about using this package with your data, consider:
1. Does your sales data have a trend?
2. Is your TV data sporadic or continuous?
3. Have you log-transformed your sales?
4. What retention rates are typical in your category?

---

**Created**: January 2026  
**Purpose**: Help practitioners detect spurious regression in MMM  
**Philosophy**: Expose problems clearly; don't oversell solutions
