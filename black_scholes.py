import numpy as np
from scipy.stats import norm

# Black-Scholes-Merton call price
def bs_call_price(S, K, T, r, sigma, q=0.0):
    """
    Computes the theoretical price of a European call option using the
    Black-Scholes-Merton formula, i.e. Black-Scholes extended to a continuous
    dividend yield q. Setting q=0 recovers the plain Black-Scholes formula.

    Parameters
    ----------
    S : float
        Current stock price.
    K : float
        Strike price.
    T : float
        Time to maturity in years.
    r : float
        Risk-free rate, expressed in annual terms.
    sigma : float
        Volatility, expressed in annual terms.
    q : float
        Continuous dividend yield, expressed in annual terms.

    Returns
    -------
    float
        Black-Scholes-Merton theoretical call price.
    """

    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return np.nan
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price


# Vega of a European call option
def bs_call_vega(S, K, T, r, sigma, q=0.0):
    """
    Computes the Vega of a European call option under the Black-Scholes-Merton framework.

    Vega measures the sensitivity of the option price to a change in volatility.
    It is required for the Newton-Raphson inversion method.
    """

    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return np.nan
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
    return vega


# Implied volatility inversion using Newton-Raphson
def implied_volatility_call(C_market, S, K, T, r, q=0.0, initial_sigma=0.20, tol=1e-6, max_iter=100):
    """
    Computes the implied volatility of a European call option by inverting
    the Black-Scholes-Merton formula using the Newton-Raphson method.

    Parameters
    ----------
    C_market : float
        Observed market price of the call option.
    S : float
        Current stock price.
    K : float
        Strike price.
    T : float
        Time to maturity in years.
    r : float
        Risk-free rate, expressed in annual terms.
    q : float
        Continuous dividend yield, expressed in annual terms. Ignoring it
        (q=0) overstates the model's forward price for dividend-paying
        stocks, which biases the extracted implied volatility downward -
        see the discussion in the write-up.
    initial_sigma : float
        Initial volatility guess for the algorithm.
    tol : float
        Convergence tolerance.
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    float
        Implied volatility if the algorithm converges, otherwise np.nan.
    """

    if C_market <= 0 or S <= 0 or K <= 0 or T <= 0:
        return np.nan
    sigma = initial_sigma

    for _ in range(max_iter):
        price = bs_call_price(S, K, T, r, sigma, q=q)
        if np.isnan(price):
            return np.nan
        vega = bs_call_vega(S, K, T, r, sigma, q=q)
        if np.isnan(vega) or vega < 1e-8:
            return np.nan
        diff = price - C_market
        if abs(diff) < tol:
            return sigma
        sigma = sigma - diff / vega
        if sigma < 0.01 or sigma > 2.00:
            return np.nan
    return np.nan
