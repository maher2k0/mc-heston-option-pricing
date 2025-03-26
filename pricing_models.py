import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
import warnings
from scipy.stats import norm
from typing import List
from scipy.optimize import brentq


class HestonModel:
    def __init__(self, S0, r, kappa, theta, sigma, rho, v0):
        self.S0 = S0
        self.r = r
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.v0 = v0
    
    def heston_characteristic_function(self, phi, T):
        """ Compute the Heston Model Characteristic Function"""
        # Parameters
        tau = T
        i = complex(0, 1)
        
        # Compute M and N
        M = np.sqrt((self.rho * self.sigma * i * phi - self.kappa)**2 + self.sigma**2 * (i * phi + phi**2))
        N = (self.rho * self.sigma * i * phi - self.kappa - M) / (self.rho * self.sigma * i * phi - self.kappa + M)
        
        # Compute A, B, C
        A = self.r * i * phi * tau + (self.kappa * self.theta / self.sigma**2) * (
            -(self.rho * self.sigma * i * phi - self.kappa - M) * tau - 2 * np.log((1 - N * np.exp(M * tau)) / (1 - N))
        )
        B = 0
        C = ((np.exp(M * tau) - 1) * (self.rho * self.sigma * i * phi - self.kappa - M)) / (self.sigma**2 * (1 - N * np.exp(M * tau)))
        
        # Characteristic function
        f = np.exp(A + B * np.log(self.S0) + C * self.v0 + i * phi * np.log(self.S0))
        return f
    
    def integrand(self, phi, K, T, flag):
        """Integrand for the option price formula"""
        i = complex(0, 1)
        if flag == 1:
            f = self.heston_characteristic_function(phi - i, T)
            return np.real((K**(-i * phi) * f) / (i * phi))
        else:
            f = self.heston_characteristic_function(phi, T)
            return np.real((K**(-i * phi) * f) / (i * phi))
        
    def heston_option_price(self, K, T):
        """Compute the Heston Model option price"""
        # Compute the two integrals using numerical integration
        integral1, _ = quad(self.integrand, 0, 100, args=(K, T, 1))
        integral2, _ = quad(self.integrand, 0, 100, args=(K, T, 2))
        
        # Option price formula
        C = 0.5 * self.S0 + (np.exp(-self.r * T) / np.pi) * integral1 - K * np.exp(-self.r * T) * (0.5 + (1 / np.pi) * integral2)
        return C
    
    def heston_monte_carlo(self, T, N, mu, num_sims=500):
        """Monte Carlo simulation of the Heston Model"""
        dt = T/N # Time step size
        # Generate correlated Brownian motions for all simulations at once
        dW2 = np.random.normal(0, np.sqrt(dt), size=(num_sims, N))
        dW1 = self.rho * dW2 + np.sqrt(1 - self.rho**2) * np.random.normal(0, np.sqrt(dt), size=(num_sims, N))

        # Initialize arrays for stock prices and volatilities
        S = np.zeros((num_sims, N+1))
        V = np.zeros((num_sims, N+1))
        S[:, 0] = self.S0
        V[:, 0] = self.v0

        # Vectorized Euler-Maruyama method
        for i in range(N):
            V[:, i+1] = V[:, i] + self.kappa * (self.theta - np.maximum(V[:, i], 0)) * dt + self.sigma * np.sqrt(np.maximum(V[:, i], 0)) * dW2[:, i]
            # Use full truncation scheme to ensure volatility stays positive
            V[:, i+1] = np.maximum(V[:, i+1], 0)
            S[:, i+1] = S[:, i] * (1 + mu * dt + np.sqrt(V[:, i]) * dW1[:, i])
            #S[:, i+1] = S[:, i] * np.exp((mu - 0.5 * V[:, i]) * dt + np.sqrt(V[:, i]) * np.sqrt(dt) * dW1[:, i])

        return S, V
    

class OptionPricer(HestonModel):
    def __init__(self, S0, r, kappa, theta, sigma, rho, v0, S, V, option_type = 'call', pricer='heston'):
        super().__init__(S0, r, kappa, theta, sigma, rho, v0)
        self.S = S
        self.V = V
        self.option_type = option_type
        self.pricer = pricer

    def MC_call_pricing(self, trading_days, strike, T):
        """Heston Monte Carlo pricing of European call options"""
        S_T = np.array([x[trading_days-1] for x in self.S])
        call_payoffs = np.maximum(S_T - strike, 0)
        call_price = np.mean(call_payoffs)

        # Calculate discounted expected payoff
        call_price = call_price * np.exp(-self.r*T)
        return call_price
    
    def MC_put_prices(self, trading_days, strike, T):
        """Heston Monte Carlo pricing of European put options"""
        S_T = np.array([x[trading_days-1] for x in self.S])
        put_payoffs = np.maximum(strike - S_T, 0)
        put_price = np.mean(put_payoffs)

        # Calculate discounted expected payoff
        put_price = put_price * np.exp(-self.r*T)
        return put_price
    
    def BS_CALL(self, T, K):
        """Black-Scholes pricing of European call options"""
        N = norm.cdf
        d1 = (np.log(self.S0/K) + (self.r + self.sigma**2/2)*T) / (self.sigma*np.sqrt(T))
        d2 = d1 - self.sigma * np.sqrt(T)
        return self.S0 * N(d1) - K * np.exp(-self.r*T)* N(d2)

    def BS_PUT(self, T, K):
        """Black-Scholes pricing of European put options"""
        N = norm.cdf
        d1 = (np.log(self.S0/K) + (self.r + self.sigma**2/2)*T) / (self.sigma*np.sqrt(T))
        d2 = d1 - self.sigma* np.sqrt(T)
        return K*np.exp(-self.r*T)*N(-d2) - self.S0*N(-d1)


    def price_options(self, strikes, trading_days, T=1.0):
        """Price options using the specified pricer"""
        if self.pricer == 'heston' and self.option_type == 'call':
            option_prices = [self.heston_option_price(K, T) for K in strikes]

        elif self.pricer == 'monte_carlo' and self.option_type == 'call':
            option_prices = [self.MC_call_pricing(trading_days, K, T) for K in strikes]
        
        elif self.pricer == 'monte_carlo' and self.option_type == 'put':
            option_prices = [self.MC_put_prices(trading_days, K, T) for K in strikes]
        
        elif self.pricer == 'black_scholes' and self.option_type == 'call':
            option_prices = [self.BS_CALL(T, K) for K in strikes]
        
        elif self.pricer == 'black_scholes' and self.option_types == 'put':
            option_prices = [self.BS_PUT(T, K) for K in strikes]

        return option_prices
    
            

class MLEOptimizer:

    def __init__(self, Q: List[float], V: List[float], r:float, n_guesses=10):
        self.Q = np.array(Q)
        self.V = np.array(V)
        self.r = r
        self.n_guesses = n_guesses

    def transform_parameters(self, params_transformed: np.ndarray) -> np.ndarray:
        """
        Transform parameters from unconstrained to constrained space with specific bounds:
        0.5 < k < 5
        0 < theta < 1
        0 < sigma < 5
        -1 < rho < 0
        """
        x1, x2, x3, x4 = params_transformed
        
        k = 0.5 + 4.5 * (1 / (1 + np.exp(-x1)))
        theta = 1 / (1 + np.exp(-x2))
        sigma = 5 / (1 + np.exp(-x3))
        rho = -1 / (1 + np.exp(-x4))
        
        return np.array([k, theta, sigma, rho])
    
    def transform_parameters_inverse(self, params: np.ndarray):
        """Transform parameters from constrained to unconstrained space"""
        k, theta, sigma, rho = params
        
        # Inverse transforms
        x1 = -np.log((4.5/(k - 0.5)) - 1)
        x2 = -np.log((1/theta) - 1)
        x3 = -np.log((5/sigma) - 1)
        x4 = -np.log((-1/rho) - 1)
        
        return np.array([x1, x2, x3, x4])


    def log_likelihood_transformed(self, params_transformed):
        """Log-likelihood function with parameter transformation"""
        params = self.transform_parameters(params_transformed)
        k, theta, sigma, rho = params
        
        Q = np.array(self.Q) #Array of changes in asset returns S[t]/S[t-1]
        V = np.array(self.V) #Array of variance of asset returns
        n = len(V) - 1
        
        ll = 0
        
        try:
            for t in range(n):
                term1 = -np.log(2 * np.pi) - np.log(sigma) - np.log(V[t])
                term2 = -0.5 * np.log(1 - rho**2)
                
                frac1 = -(Q[t+1] - 1 - self.r)**2 / (2 * V[t] * (1 - rho**2))
                frac2 = rho * (Q[t+1] - 1 - self.r) * (V[t+1] - V[t] - theta * k + k * V[t]) / (V[t] * sigma * (1 - rho**2))
                frac3 = -((V[t+1] - V[t] - theta * k + k * V[t])**2) / (2 * sigma**2 * V[t] * (1 - rho**2))
                
                ll += term1 + term2 + frac1 + frac2 + frac3
        except:
            return -np.inf
        
        # Add penalty for extreme values
        penalty = 0
        for param in params:
            if np.abs(param) > 1e3:
                penalty += np.abs(param)
        
        return -ll - penalty
    
    def generate_initial_guesses(self):
        """Generate multiple random initial guesses"""
        guesses = []
        for _ in range(self.n_guesses):
            #r = np.random.uniform(0.01, 1)
            k = np.random.uniform(0.5, 5)
            theta = np.random.uniform(0, 1)
            sigma = np.random.uniform(0, 5)
            rho = np.random.uniform(-1, 0)
            guesses.append([k, theta, sigma, rho])
        return guesses
        
    def estimate_parameters_robust(self):
        """Estimate parameters using multiple starting points and optimization methods"""
        initial_guesses = self.generate_initial_guesses()
        best_result = None
        best_likelihood = -np.inf
        
        methods = ['L-BFGS-B', 'SLSQP', 'TNC']
        count=0

        for guess in initial_guesses:
            guess_transformed = self.transform_parameters_inverse(guess)
            
            for method in methods:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        
                        result = minimize(
                            self.log_likelihood_transformed,
                            guess_transformed,
                            method=method,
                            options={
                                'maxiter': 1000,
                                'ftol': 1e-8,
                                'gtol': 1e-8
                            }
                        )
                    count+=1
                    print('SEARCH {}'.format(count))
                    
                    if result.success and -result.fun > best_likelihood:
                        best_likelihood = -result.fun
                        best_result = result
                        best_method = method
                        best_initial = guess
                except:
                    continue
        
        if best_result is None:
            raise ValueError("Optimization failed for all attempts")
        
        # Transform parameters back to constrained space
        final_params = self.transform_parameters(best_result.x)
        
        return {
            'parameters': {
                #'r': final_params[0],
                'k': final_params[0],
                'theta': final_params[1],
                'sigma': final_params[2],
                'rho': final_params[3]
            },
            'log_likelihood': best_likelihood,
            'convergence': best_result.success,
            'method': best_method,
            'initial_guess': best_initial,
            'message': best_result.message
        }