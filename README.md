# **Option Pricing with the Heston Model**
This project implements a Monte Carlo (MC) simulation using the **Heston stochastic volatility model** to price European call and put options. Additionally, it compares the results with the **closed-form option valuation scheme** from Heston and Nandi (2000), the **Black-Scholes model**, and **real option prices** obtained from Yahoo Finance.
A **Maximum Likelihood Estimation (MLE)** scheme is used to estimate the Heston Model parameters for calibration to market data of any specific stock.

📄 This repository is part of a project on option pricing with the Heston Model. The blog posts covering the theory, implementation and discussion of results can be found here: [**Part 1**](https://medium.com/@uddin.maher/option-pricing-with-the-heston-model-part-1-heston-model-theory-8b9950620a60) and [**Part 2**](https://medium.com/@uddin.maher/option-pricing-with-the-heston-model-part-2-parameter-estimation-monte-carlo-simulations-2a7be6e00cf2)

# 🚀 **Project Overview**
The goal of this project is to explore and compare different methods for pricing European call options. The primary focus is on the Heston model, which incorporates stochastic volatility to better capture market dynamics compared to the constant volatility assumption in the Black-Scholes model. The project includes:

1. **Monte Carlo Simulation using the Heston Model**: A full implementation of the Heston stochastic volatility model to simulate asset price paths and estimate option prices.
2. **Closed-Form Option Valuation (Heston and Nandi, 2000)**: Implementation of the analytical solution for option pricing under the Heston and Nandi framework.
3. **Black-Scholes Model**: A benchmark comparison using the classic Black-Scholes option pricing formula.
4. **Real Option Prices**: Validation of the models using real-world option prices from Yahoo Finance.
5. **Parameter Estimation using Maximum Likelihood Estimation (MLE)**: Estimation of the Heston model parameters:
   - Mean reversion rate (κ)
   - Long-term average volatility (θ)
   - Volatility of volatility (σ)
   - Correlation coefficient (ρ)

# 📊 **Features**
✅ Monte Carlo simulations with the Heston model  
✅ Closed-form valuation using Heston–Nandi methodology  
✅ Benchmarking against Black–Scholes prices  
✅ Real option data retrieval and comparison from Yahoo Finance  
✅ MLE-based parameter estimation for the Heston model using historical data  
✅ Visualization of price differences, model performance, and parameter fits  
✅ Volatility smile analysis for different pricing models  

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
  - `scipy.optimize.brentq` – Root finding  
- `warnings` – Warning management  
- `typing` – Type hinting support  

## 📁 **Project Structure**
- `pricing_models.py`: Contains the core pricing models including:
  - `HestonModel`: Implements the Heston stochastic volatility model
  - `OptionPricer`: Handles option pricing using different methods
  - `MLEOptimizer`: Performs parameter estimation using MLE
- `utils.py`: Utility functions for data processing and analysis
- `playground.ipynb`: Jupyter notebook for interactive exploration and testing
- `option_analysis.ipynb`: Notebook for analyzing option prices and volatility surfaces
- `requirements.txt`: List of Python package dependencies

## 📝 **References**  
- S. Heston and S. Nandi, “A Closed-Form GARCH Option Valuation Model.” The Review of Financial Studies (2000), 13, 585-625.
- R. Dunn, P. Hauser, T. Seibold and H. Gong, "Estimating Option Prices with Heston’s Stochastic Volatility Model" Valparaiso University (2014)

## 🤝 **Contributing**
Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 **License**
This project is licensed under the MIT License - see the LICENSE file for details.
