# Case1: Gaming industry optimal portfolio

### Overview
The purpose of the project is to create a portfolio with gaming industry stocks.

### Stock Set (currency: USD)
* Nintendo (NTDOY)
* Electronic Arts (EA)
* Activision Blizzard (ATVI)
* Sony (SNE)
* Ubisoft (UBSFY)

### Data cleaning
```
import ffn
import numpy as np
import pandas as pd
import matplotlib.pyplot as py
%matplotlib inline
import seaborn as sn
import cvxopt as opt
from cvxopt import blas, solvers
import scipy as sci
from scipy import stats

stock_price = ffn.get('NTDOY, EA, ATVI, SNE, UBSFY', start = '2020-01-01')
py.figure(figsize = (12, 6))
py.plot(stock_price['ntdoy'], label = 'Nintendo')
py.plot(stock_price['ea'], label = 'Electronic Arts')
py.plot(stock_price['atvi'], label = 'Activision Blizzard')
py.plot(stock_price['sne'], label = 'Sony')
py.plot(stock_price['ubsfy'], label = 'Ubisoft')
py.legend(loc = 'best')
py.grid(True)
```
![](https://i.imgur.com/71l0aTe.png)

```
# Rebase stock price change
py.figure(figsize = (12, 6))
py.plot(stock_price['ntdoy'].rebase(), label = 'Nintendo')
py.plot(stock_price['ea'].rebase(), label = 'Electronic Arts')
py.plot(stock_price['atvi'].rebase(), label = 'Activision Blizzard')
py.plot(stock_price['sne'].rebase(), label = 'Sony')
py.plot(stock_price['ubsfy'].rebase(), label = 'Ubisoft')
py.legend(loc = 'upper left')
py.grid(True)
```
![](https://i.imgur.com/MKbM4u4.png)

### Descriptive analysis

```
# Observe the correlation between stocks
stock_price.plot_corr_heatmap()
```
![](https://i.imgur.com/CTBPaGu.png)


```
# Aquire daily price variation
pct_change = stock_price.pct_change().dropna()

# Observe the distribution of price variation
py.figure(figsize = (12,6))
pct_change['ntdoy'].plot.hist(alpha = 0.5, bins = 100, legend = True)
pct_change['ea'].plot.hist(alpha = 0.5, bins = 100, legend = True)
pct_change['atvi'].plot.hist(alpha = 0.5, bins = 100, legend = True)
pct_change['sne'].plot.hist(alpha = 0.5, bins = 100, legend = True)
pct_change['ubsfy'].plot.hist(alpha = 0.5, bins = 100, legend = True)
```
![](https://i.imgur.com/YVuABah.png)

```
# Observe the volatility of  sigle stock
# Acquire standard deviation of stocks
volatility = pct_change.std()
print(volatility)

# Plot the variaton through time line
graph = pct_change.plot(figsize = (12,6), grid = True)
graph.set_ylabel('Return Rate')

# Annualize price variation
days = stock_price.shape[0]
assets = stock_price.shape[1]
print(pct_change)
annualized_return_rate = ((1 + pct_change).cumprod() ** (1 / days))[-1:] ** 252 - 1
```

### Model building
```
# Apply Mean-Variance Analysis to portfolio
def generate_one_portfolio(pct_change):
  # randomly get a asset allocation
  weight = np.random.random(pct_change.shape[1])
  weight /= np.sum(weight)
  
  # calculate the annualized return rate
  annualized_return_rate = ((1 + pct_change).cumprod() ** (1 / days)-1)[-1:]
  annualized_return_rate = annualized_return_rate.iloc[0, :]
  portfolio_return = weight.T.dot(annualized_return_rate) * 252
  
  # calaulate the volatility of the asset allocation
  stock_cov = pct_change.cov()
  portfolio_volatility = np.sqrt(weight.T.dot(stock_cov.dot(weight)) * 252)
  return weight, portfolio_return, portfolio_volatility

n = 8000
weight, portfolio_return, portfolio_volatility = np.column_stack([generate_one_portfolio(pct_change) for _ in range(n)])
py.figure(figsize = (12,8))
py.scatter(portfolio_volatility , portfolio_return, c = (portfolio_return - 0.0046)/(portfolio_volatility**0.5), marker = 'o')
py.colorbar(label = 'Sharpe ratio')
py.xlabel('Volatility (Standard Deviation)')
py.ylabel('Return Rate')
py.title('Mean-Variance Analysis of Portfolios')
py.grid()
```
![](https://i.imgur.com/3xadlNT.png)


```
# Calculate the optimal asset allocation
def opt_portfolio(pct_change):
    
    n = pct_change.shape[1]
    solvers.options['show_progress'] = False
     
    N = 50
    risk_levels = [10 ** (2 * t / N - 2) for t in range(N)]
    
    p = (((1 + pct_change).cumprod() ** (1 / days)) - 1)[-1:]
    p = opt.matrix(p.values * 252).T
    S = opt.matrix(pct_change.cov().values * 252)
    
    G = opt.matrix(-np.eye(n))
    h = opt.matrix(0.0, (n, 1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)
    
    weights = [solvers.qp(2 * S, -level * p, G, h, A, b)['x'] for level in risk_levels]
    
    returns = np.array([blas.dot(p, x) for x in weights])
    vols = np.array([np.sqrt(blas.dot(x, S * x)) for x in weights])
    
    idx = (returns / vols).argmax()
    wt =  weights[idx]
    
    return idx, wt, returns, vols

opt_idx, opt_weight, opt_returns, opt_risks = opt_portfolio(pct_change)
opt_returns = np.sort(opt_returns)
opt_risks = np.sort(opt_risks)

ind = np.argmin(opt_risks)
evols = opt_risks[ind:]
erets = opt_returns[ind:]
tck = sci.interpolate.splrep(evols, erets)

def f(x):
    return sci.interpolate.splev(x, tck, der = 0)

def df(x):
    return sci.interpolate.splev(x, tck, der = 1)

def equations(p, rf = 0.0046):
    eq1 = rf - p[0]
    eq2 = rf + p[1] * p[2] - f(p[2])
    eq3 = p[1] - df(p[2])
    return eq1, eq2, eq3
    
opt = sci.optimize.fsolve(equations, [0.01, 0.5, 0.15])
print(opt) # opt[0]: risk-free rate, opt[1]: maximum Sharpe ratio, opt[2]: tangent of EF

py.figure(figsize = (12, 6))
py.scatter(portfolio_volatility, portfolio_return, c = (portfolio_return - 0.0046)/(portfolio_volatility **0.5), marker = 'o')
py.scatter(opt_risks[opt_idx], opt_returns[opt_idx], c = "r", marker = 'x', s = 100, label = 'Optimal Portfolio')
py.legend(loc = "best")
py.colorbar(label = 'Sharpe ratio')
py.plot(opt_risks, opt_returns, 'y-o')

py.plot([0, opt_risks[opt_idx]], [0.0046, opt_returns[opt_idx]], "r--")
py.grid()
```
![](https://i.imgur.com/2SEngSD.png)


```
# Based on the risk-free rate, acquire the optimal asset allocation based on CAPM
py.figure(figsize = (5, 5))
py.pie(list(opt_weight), labels = pct_change.columns.values, autopct = "%1.2f%%"), 
py.title("Ingredient of Portfolio")
```
![](https://i.imgur.com/5TQVJqa.png)

### Validation
```
# Observe the portfolio value under buy-and-hold
pct_change['Equally Weighted'] = pct_change.dot(np.ones((assets, 1)) / assets)
pct_change['Opt Portfolio'] = pct_change.iloc[:, 0:assets].dot(opt_weight)

py.figure(figsize = (12,6))
py.plot(np.cumprod(pct_change['Equally Weighted'] + 1), label = 'Equally Weighted')
py.plot(np.cumprod(pct_change['Opt Portfolio'] + 1), label = 'Opt Portfolio')
py.legend(loc = 'best')
py.ylabel('Portfolio Value')
py.grid(True)
```
![](https://i.imgur.com/40AcsDj.png)
