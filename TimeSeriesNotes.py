# -*- coding: utf-8 -*-
"""
Created on Wed May 13 12:12:50 2020

@author: dimit
"""
import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight') 
# Above is a special style template for matplotlib, highly useful for visualizing time series data
%matplotlib inline
from pylab import rcParams
import statsmodels.api as sm
import statsmodels.tsa.api as smt
from numpy.random import normal, seed
from scipy.stats import norm
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.arima_model import ARIMA
import math
from sklearn.metrics import mean_squared_error
print(os.listdir("../input"))

cd "C:\Users\dimit\Documents\GitHub\Time-Series-2---Predicting-air-quality-"
data = pd.read_csv("energydata_complete.csv", index_col="date", parse_dates=['date'])

# if it needs cleaning
#pressure = pressure.iloc[1:]
#pressure = pressure.fillna(method='ffill')

# initial plot
data["T3"].asfreq('W').plot()

# if you need to slice the times
dr2 = pd.date_range(start='1/1/18', end='1/1/19', freq='M')

data = data.resample('3H').mean()

rcParams['figure.figsize'] = 11, 9

decomposed = sm.tsa.seasonal_decompose(data['T3'],freq=1)
figure = decomposed.plot()
plt.show()

def plot_moving_average(series, window, plot_intervals=False, scale=1.96):

    rolling_mean = series.rolling(window=window).mean()

    plt.figure(figsize=(17,8))
    plt.title('Moving average\n window size = {}'.format(window))
    plt.plot(rolling_mean, 'g', label='Rolling mean trend')

    #Plot confidence intervals for smoothed values
    if plot_intervals:
        mae = mean_absolute_error(series[window:], rolling_mean[window:])
        deviation = np.std(series[window:] - rolling_mean[window:])
        lower_bound = rolling_mean - (mae + scale * deviation)
        upper_bound = rolling_mean + (mae + scale * deviation)
        plt.plot(upper_bound, 'r--', label='Upper bound / Lower bound')
        plt.plot(lower_bound, 'r--')

    plt.plot(series[window:], label='Actual values')
    plt.legend(loc='best')
    plt.grid(False)
    
def exponential_smoothing(series, alpha):

    result = [series[0]] # first value is same as series
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n-1])
    return result
  
def plot_exponential_smoothing(series, alphas):
 
    plt.figure(figsize=(17, 8))
    for alpha in alphas:
        plt.plot(exponential_smoothing(series, alpha), label="Alpha {}".format(alpha))
    plt.plot(series.values, "c", label = "Actual")
    plt.legend(loc="best")
    plt.axis('tight')
    plt.title("Exponential Smoothing")
    plt.grid(True);

plot_exponential_smoothing(edata.T2, [0.05, 0.3])

from sklearn.metrics import mean_absolute_error
plot_moving_average(data.T3, 10, plot_intervals=True)   

data.T3.plot()
data.T4.plot()
plt.legend(['T3','T4'])
plt.show()

rolling_10_t3 = data.T3.rolling('10H').mean()
data.T3.plot()
rolling_10_t3.plot()
plt.legend(['T3', 'Rolling'])
plt.show()

rolling_10_t3 = data.T3.rolling('5D').mean()
data.T3.plot()
rolling_10_t3.plot()
plt.legend(['T3', 'Rolling'])
plt.show()

#------------------------------------------------------#
# Manual Autocorrelation
#------------------------------------------------------#

plot_acf(data['T3'], lags=30,title='T3')
plt.show()

#------------------------------------------------------#
# Manual Partial Autocorrelation
#------------------------------------------------------#

plot_pacf(data['T3'], lags=30,title='T3')
plt.show()

#------------------------------------------------------#
#Perform Dickey-Fuller test:
#------------------------------------------------------#

# Augmented Dickey-Fuller test OPTION 1 (MANUAL)
adf = adfuller(data.T2)
print("p-value of microsoft: {}".format(float(adf[1])))
adf = adfuller(data.T3)
print("p-value of google: {}".format(float(adf[1])))

# Take the first difference to remove to make the process stationary
data_diff = edata.T2 - edata.T2.shift(1)

# Augmented Dickey-Fuller test OPTION 2 (NEAT WITH FUNCTION)

def test_stationarity(timeseries):
    
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)

test_stationarity(data.T3)

#------------------------------------------------------#
#--------------- Summary of ADF and Corr plots --------#
#------------------------------------------------------#

def tsplot(y, lags=None, figsize=(12, 7), syle='bmh'):
    
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
        
    with plt.style.context(style='bmh'):
        fig = plt.figure(figsize=figsize)
        layout = (2,2)
        ts_ax = plt.subplot2grid(layout, (0,0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1,0))
        pacf_ax = plt.subplot2grid(layout, (1,1))
        
        y.plot(ax=ts_ax)
        p_value = sm.tsa.stattools.adfuller(y)[1]
        ts_ax.set_title('Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(p_value))
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
        plt.tight_layout()
        
tsplot(data.T3, lags=30)

#------------------------------------------------------#
#--------------- MODELLING HERE -----------------------#
#------------------------------------------------------#
#--------------- START WITH AR  -----------------------#
#------------------------------------------------------#

# Focus on  AR, MA and ASRA

# AR(1) MA(1) model:AR parameter = +0.9
rcParams['figure.figsize'] = 16, 12
plt.subplot(4,1,1)
ar1 = np.array([1, -0.9]) # We choose -0.9 as AR parameter is +0.9
ma1 = np.array([1])
AR1 = ArmaProcess(ar1, ma1)
sim1 = AR1.generate_sample(nsample=1000)
plt.title('AR(1) model: AR parameter = +0.9')
plt.plot(sim1)

# AR(1) MA(1) AR parameter = -0.9
plt.subplot(4,1,2)
ar2 = np.array([1, 0.9]) # We choose +0.9 as AR parameter is -0.9
ma2 = np.array([1])
AR2 = ArmaProcess(ar2, ma2)
sim2 = AR2.generate_sample(nsample=1000)
plt.title('AR(1) model: AR parameter = -0.9')
plt.plot(sim2)

# Activate the model in a simulation

model = ARMA(sim1, order=(1,0))
result = model.fit()
print(result.summary())
print("μ={} ,ϕ={}".format(result.params[0],result.params[1]))

# results from sim 1 μ=-0.059368646525892355 ,ϕ=0.8892422524373136
# results from sim 2 μ=0.02355623994485676 ,ϕ=-0.864264763960079

# --- Plot the simulations -----
result.plot_predict(start=900, end=1010)
plt.show()

rmse = math.sqrt(mean_squared_error(sim1[900:1011], result.predict(start=900,end=999)))
print("The root mean squared error is {}.".format(rmse))

# ---- Apply on real data - T3 reading

AR1t3predict = ARMA(data.T3.diff().iloc[1:].values, order=(1,0))
t3 = t3predict.fit()
t3.plot_predict(start=1000, end=1100)
plt.show()

rmse = math.sqrt(mean_squared_error(data.T3.diff().iloc[900:1000], result.predict(start=900,end=999)))
print("The root mean squared error is {}.".format(rmse))

#------------------------------------------------------#
#--------------- MOVING ON TO MA ----------------------#
#------------------------------------------------------#

# simulate MA model

AR1rcParams['figure.figsize'] = 16, 6
ar1 = np.array([1])
ma1 = np.array([1, -0.5])
MA1 = ArmaProcess(ar1, ma1)
sim1 = MA1.generate_sample(nsample=1000)
plt.plot(sim1)

# forecast the simulation with MA

simulation_ma_model = ARMA(sim1, order=(0,1))
result = simulation_ma_model.fit()
print(result.summary())
print("μ={} ,θ={}".format(result.params[0],result.params[1]))

# prediction on actual data

real_data_ma_model = ARMA(data['T3'].diff().iloc[1:].values, order=(0,3))
result = real_data_ma_model.fit()
print(result.summary())
print("μ={} ,θ={}".format(result.params[0],result.params[1]))
result.plot_predict(start=1000, end=1100)
plt.show()

rmse = math.sqrt(mean_squared_error(data.T3.diff().iloc[900:1000], result.predict(start=900,end=999)))
print("The root mean squared error is {}.".format(rmse))
# The root mean squared error is 0.36690362745507066.


#------------------------------------------------------#
#--------------- MOVING ON TO ARMA --------------------#
#------------------------------------------------------#

real_data_arma_model = ARMA(data['T3'].diff().iloc[1:].values, order=(3,3))
result = real_data_arma_model.fit()
print(result.summary())
print("μ={}, ϕ={}, θ={}".format(result.params[0],result.params[1],result.params[2]))
result.plot_predict(start=1000, end=1100)
plt.show()

rmse = math.sqrt(mean_squared_error(data.T3.diff().iloc[900:1000], result.predict(start=900,end=999)))
print("The root mean squared error is {}.".format(rmse))

# The root mean squared error is 0.36455704283985324.





