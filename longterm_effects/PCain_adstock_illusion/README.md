# The Long-Term Advertising Effects Problem: An Honest Assessment

## ⚠️ DISCLAIMER - Notes from Charlie

**I'll be straightforward and honest:**

There is so much noise in marketing data that, by experience, I know it is almost impossible to measure the long-term effect. Signals over time are drowned in the noise, and all channels' adstock end up being highly correlated. 

**Most of "long-term effect measurement" sold by vendors are borderline snake oil.** A lot of solutions look elegant in theory but then fall apart in practice. Be very skeptical when vendors claim they've solved long-term measurement. Ask to see their Durbin-Watson statistics, retention rates, and residual diagnostics. If they won't share or don't know what you're asking about that is a red flag

Now, it does not mean that we should not try, and give some credit to people who try. This repository presents research by P.M. Cain (2025) that exposes one specific problem and proposes alternatives. Are these alternatives perfect? No. But they're more honest about what we can and cannot measure.

---

## What This Repository Contains

Based on: **Cain, P.M. (2025). "Long-term advertising effects: The Adstock illusion." *Applied Marketing Analytics*, 11(1), 23-42.**

### The Core Problem Exposed:

**Dual-adstock models** (using λ=0.99 for "long-term effects") produce **spurious regression**:
- They find "long-term effects" even when none exist
- High-retention adstock accumulates like a trend
- Correlates with any drift in base sales  
- Diagnostics reveal the problem (Durbin-Watson < 1.5)

### Cain's Proposed Alternatives:

1. **Unobserved Components Model (UCM)** - Explicitly decomposes trend from activation
2. **Vector Autoregression (VAR)** - Tests if effects truly persist via IRFs
3. **Combined UCM+VAR** - Uses brand metrics to ground long-term measurement

**Are these perfect?** No. But they avoid the specific spurious regression trap that dual-adstock falls into.

---

## Files in This Repository

### 1. `long_term_measurement.py`
Python module with all functions:
- Dual-adstock diagnostics (detect spurious regression)
- UCM implementation (with caveats)
- VAR with Impulse Response Functions
- Combined approach (conceptual)

### 2. `long_term_measurement_demo.ipynb`
Comprehensive Jupyter notebook demonstrating:
- Why dual-adstock fails (with proof)
- How UCM works (and its limitations)
- How VAR/IRF analysis works
- Combined approach (theory)

### 3. This README
Documentation with honest assessment of limitations

---

## Quick Start

### Install Dependencies
```bash
pip install numpy pandas matplotlib seaborn scipy statsmodels scikit-learn
```

### Check Your Model for Spurious Regression (30 seconds)
```python
from long_term_measurement import check_dual_adstock_diagnostics

# Your data
sales = ...  # Log-transformed sales
tv = ...     # TV advertising

# Quick diagnostic
results = check_dual_adstock_diagnostics(sales, tv)

# If Durbin-Watson < 1.5: Your model is likely spurious
```

### Run Complete Analysis
```python
jupyter notebook long_term_measurement_demo.ipynb
```

---

## What Each Approach Does (and Doesn't Do)

### ❌ Dual-Adstock (What Most Vendors Use)

**What it claims to do:**
- Short-term adstock (λ=0.30) measures activation
- Long-term adstock (λ=0.99) measures brand-building

**What's wrong with it:**
- High retention creates deterministic drift
- Correlates with any trend in data
- Finds "effects" that don't exist
- Diagnostics: DW < 1.5, failed Ljung-Box

**Verdict: Fundamentally flawed. Don't use λ > 0.95.**

---

### 1️⃣ Unobserved Components Model (UCM)

**What it does:**
```
Sales = Stochastic Trend + Short-term effects + Seasonality + Noise
```
- Explicitly models trend (base sales evolution)
- Brand-building = advertising driving the trend
- Activation = temporary deviations from trend

**Limitations:**
- Requires expertise with state-space models
- Sensitive to model specification
- No consensus on how advertising should enter trend equation
- Still assumes trend can be cleanly separated

**Honest assessment:**
✅ Better than dual-adstock (avoids spurious regression)  
⚠️ Results depend heavily on model choices  
❌ Not a silver bullet

---

### 2️⃣ Vector Autoregression (VAR)

**What it does:**
- Tests if sales are stationary (unit root tests)
- Models sales and TV jointly with lags
- Impulse Response Functions (IRFs) show:
  - If effects decay → activation only
  - If effects persist → brand-building present

**Limitations:**
- Requires substantial data (loses degrees of freedom)
- Sensitive to lag length choice
- Cointegration testing can be ambiguous
- IRFs can be unstable with small samples

**Honest assessment:**
✅ Theoretically sound (avoids spurious regression)  
✅ Provides proper statistical inference  
⚠️ Requires expertise to implement correctly  
❌ Data hungry (may not work with limited history)

---

### 3️⃣ Combined UCM + VAR

**What it does:**
```
Step 1: UCM extracts base sales trend
Step 2: VAR models trend using brand metrics + advertising
```
- Grounds long-term effects in consumer brand perceptions
- Tests for cointegration between sales and brand metrics
- Provides theoretical justification (memory structures)

**Limitations:**
- Requires brand survey data (often unavailable)
- Two-step estimation compounds uncertainty
- Very complex to implement correctly
- Few practitioners have actually done this

**Honest assessment:**
✅ Most theoretically sound approach  
⚠️ Highly complex (borderline impractical)  
❌ Requires data most companies don't have  
❌ No guarantee it will work even with good data

---

## The Brutal Truth About Long-Term Measurement

### Why This is So Hard:

1. **Signal-to-Noise Ratio is Terrible**
   - Long-term effects are small and slow
   - Marketing data is noisy and confounded
   - Many things change simultaneously

2. **Correlation ≠ Causation**
   - Brand-building happens over 6-18 months
   - During that time: competition changes, economy shifts, seasons pass
   - What caused the base sales increase? Almost impossible to isolate

3. **All Channels Are Correlated**
   - You don't run TV in isolation
   - TV, digital, PR all happen together
   - Adstock transformations make them even more correlated

4. **Data Requirements Are Unrealistic**
   - Need 3+ years of weekly data (minimum)
   - Need consistent measurement throughout
   - Need no structural breaks (yeah, right)
   - Need brand surveys (expensive, often unavailable)

### What We CAN Do:

1. **Avoid Obvious Mistakes**
   - Don't use dual-adstock with λ > 0.95
   - Always check diagnostics (DW, Ljung-Box)
   - Be honest about uncertainty

2. **Use Multiple Forms of Evidence**
   - MMM (with caveats)
   - Brand lift studies
   - Experiments (where possible)
   - Share of voice analysis
   - Category growth patterns

3. **Be Humble**
   - Report ranges, not point estimates
   - Acknowledge what we don't know
   - Don't oversell certainty to stakeholders

---

## Critical Diagnostic Thresholds

| Test | Safe | Warning | Danger |
|------|------|---------|--------|
| **Durbin-Watson** | ≥ 1.8 | 1.5-1.8 | < 1.5 ❌ |
| **Ljung-Box p-value** | > 0.05 | 0.01-0.05 | < 0.01 ❌ |

**If DW < 1.5:** Your model almost certainly suffers from spurious regression. The "long-term effect" is a statistical artifact.

---

## Real Talk: What Should Practitioners Do?

### 1. Be Skeptical of Vendor Claims
If a vendor says:
- "We've solved long-term measurement"
- "Our proprietary method captures brand-building"
- "Trust us, it works"

**Ask them:**
- What's your Durbin-Watson statistic?
- What's your Ljung-Box test result?
- Can you show me residual diagnostics?
- What retention rate are you using?

If they can't answer or won't share → 🚩🚩🚩

### 2. Use Shorter Retention Rates
Instead of λ = 0.99:
- Try λ = 0.5-0.8
- Accept that you're measuring "medium-term" not "long-term"
- Be honest that true long-term effects are hard to isolate

### 3. Triangulate Multiple Evidence Sources
Don't rely on MMM alone:
- Run brand lift surveys
- Monitor share of voice
- Track brand search trends
- Do holdout experiments (if budget allows)
- Look at category-level patterns

### 4. Report Uncertainty Honestly
Instead of: "TV drives 30% long-term lift"  
Say: "TV shows medium-term effects of 15-45% (wide range due to measurement uncertainty)"

---

## Academic Reference

**Cain, P.M. (2025).** "Long-term advertising effects: The Adstock illusion." *Applied Marketing Analytics*, 11(1), 23-42.

**Key insight from Cain:**
> "The very high retention rates used to mimic brand-building behaviour essentially imply that sales follow a deterministic growth path. This leads to spurious regression issues, and fails to accurately reflect the theoretical brand-building process."

---

## Final Thoughts

**From Charlie:**

This repository doesn't claim to have the answer. What it does is:
1. Show you one approach that's definitely wrong (dual-adstock with high retention)
2. Show you some approaches that are less wrong (but still imperfect)
3. Give you tools to evaluate claims critically

**The honest truth:** Long-term effects are really hard to measure. Anyone who tells you otherwise is likely either naive or selling something.

But that doesn't mean we give up. It means we're honest about uncertainty, we validate carefully, and we don't oversell our confidence.

Good luck out there. 🍀
