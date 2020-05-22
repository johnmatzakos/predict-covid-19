# Author: Ioannis Matzakos | Date: 01/05/2020

# import python modules needed
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import statsmodels.api as sm
import itertools
import warnings
#from fbprophet import Prophet

# import project's classes and scripts
from Utilities import Log

# Configure logger
log = Log.setup_logger("data_mining")

# Configure pyplot
plt.style.use('fivethirtyeight')
warnings.filterwarnings("ignore")

# Configure matplotlib
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'

class DataMining:


# TODO: make more methods from the execute_process() method
    def execute_process(self, time_series, data, plot_label):
        # Time Series Forecasting and Analysis
        # Autoregressive Integrated Moving Average
        # Parameter Selection for ARIMA model
        log.info("Parameter Selection for ARIMA model")
        '''
        ARIMA Parameters
        p : seasonality
        d : trend
        q : noise
        '''
        p = d = q = range(0, 2)
        pdq = list(itertools.product(p, d, q))
        seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

        for param in pdq:
            for param_seasonal in seasonal_pdq:
                try:
                    mod = sm.tsa.statespace.SARIMAX(data, order=param, seasonal_order=param_seasonal,
                                                    enforce_stationarity=False, enforce_invertibility=False)
                    results = mod.fit()
                except:
                    continue
            log.info('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))

        # Fit the ARIMA model
        log.info("Fitting the ARIMA model")
        mod = sm.tsa.statespace.SARIMAX(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), enforce_stationarity=False,
                                        enforce_invertibility=False)
        results = mod.fit()
        # log.info(f"ARIMA Model Results: \n{results.get_forecast()}")
        log.info(f"\n{results.summary().tables[1]}")

        # Model Diagnostics
        log.info("Model Diagnostics")
        results.plot_diagnostics(figsize=(16, 8))
        plt.show()

        # Validating forecasts
        pred = results.get_prediction(start=pd.to_datetime(str(time_series.reset_index()['Date'].iloc[0])),
                                      dynamic=False)
        log.info(f"ARIMA Model Results: \n{pred}")
        pred_ci = pred.conf_int()
        log.info(f"ARIMA Model Results: \n{pred_ci}")
        ax = data['2020-02-25':].plot(label='observed')
        pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
        ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='k', alpha=.2)
        ax.set_xlabel('Date')
        ax.set_ylabel('New Confirmed Cases')
        plt.legend()
        plt.show()

        y_forecasted = pred.predicted_mean
        y_truth = data['2020-02-25':]
        mse = ((y_forecasted - y_truth) ** 2).mean()
        log.info('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

        log.info('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))

        pred_uc = results.get_forecast(steps=100)
        pred_ci = pred_uc.conf_int()
        ax = data.plot(label='observed', figsize=(14, 7))
        pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
        ax.fill_between(pred_ci.index,
                        pred_ci.iloc[:, 0],
                        pred_ci.iloc[:, 1], color='k', alpha=.25)
        ax.set_xlabel('Date')
        ax.set_ylabel(str(plot_label))
        plt.legend()
        plt.show()
'''
    def execute_mining_with_prophet(self, time_series, plot_title):
        log.info("Start time series modelling with Facebook's Prophet...")
        # time_series = time_series.rename(columns={'Order Date': 'ds', 'Sales': 'y'})
        # Model fitting
        time_series_model = Prophet(interval_width=0.95)
        time_series_model.fit(time_series)
        # Forecasting
        time_series_forecast = time_series_model.make_future_dataframe(periods=36, freq='MS')
        time_series_forecast = time_series_model.predict(time_series_forecast)
        # Plotting
        plt.figure(figsize=(18, 6))
        time_series_model.plot(time_series_forecast, xlabel='Date', ylabel=str(plot_title))
        plt.title(f'Forecast for Covid-19 {str(plot_title)} in Greece')
'''
