# **Option Pricing with Monte Carlo Simulation using the Heston Model**
This project implements a Monte Carlo (MC) simulation using the **Heston stochastic volatility model** to price European call options. Additionally, it compares the results with the **closed-form option valuation scheme** from Heston and Nandi (2000), the **Black-Scholes model**, and **real option prices** obtained from Yahoo Finance.
A **Maximum Likelihood Estimation (MLE)** scheme is used to estimate the Heston Model parameters for calibration to market data of any specific stock.

# 🚀 **Project Overview**
The goal of this project is to explore and compare different methods for pricing European call options. The primary focus is on the Heston model, which incorporates stochastic volatility to better capture market dynamics compared to the constant volatility assumption in the Black-Scholes model. The project includes:

1. **Monte Carlo Simulation using the Heston Model**: A full implementation of the Heston stochastic volatility model to simulate asset price paths and estimate option prices.
2. **Closed-Form Option Valuation (Heston and Nandi, 2000)**: Implementation of the analytical solution for option pricing under the Heston and Nandi framework.
3. **Black-Scholes Model**: A benchmark comparison using the classic Black-Scholes option pricing formula.
4. **Real Option Prices**: Validation of the models using real-world option prices from Yahoo Finance.
5. **Parameter Estimation using Maximum Likelihood Estimation (MLE)**: Estimation of the Heston model parameters (e.g., mean reversion rate, long-term volatility, volatility of volatility and correlation coefficient) using historical asset price data.

# 📊 **Features**
✅ Monte Carlo simulations with the Heston model  
✅ Closed-form valuation using Heston–Nandi methodology  
✅ Benchmarking against Black–Scholes prices  
✅ Real option data retrieval and comparison from Yahoo Finance  
✅ MLE-based parameter estimation for the Heston model using historical data  
✅ Visualization of price differences, model performance, and parameter fits  

## 📦 **Dependencies**  
The project uses the following Python packages:  

- `numpy` – Numerical operations  
- `matplotlib` – Data visualization  
- `pandas` – Data manipulation and analysis  
- `yfinance` – Fetching financial data from Yahoo Finance  
- `datetime` – Date and time handling  
- `py_vollib_vectorized` – Implied volatility calculations  
- `scipy` – Numerical integration and optimization:  
  - `scipy.integrate.quad` – Integration  
  - `scipy.optimize.minimize` – Optimization  
  - `scipy.stats.norm` – Statistical functions  
- `warnings` – Warning management  
- `typing` – Type hinting support  
- **Custom modules**:  
  - `utils` – Data processing functions (`process_asset_data`, `process_treasury_data`, `calculate_rolling_volatility`)  
  - `pricing_models` – Pricing models (`HestonModel`, `OptionPricer`)

See requirements.txt file for package versions used.

## 📝 **References**  
- S. Heston and S. Nandi, “A Closed-Form GARCH Option Valuation Model.” The Review of Financial Studies (2000), 13, 585-625.