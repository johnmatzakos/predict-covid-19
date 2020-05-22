# Author: Ioannis Matzakos | Date: 18/05/2020

import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from Utilities import Log

# Configure logger
log = Log.setup_logger("preliminary_analysis")


class PreliminaryAnalysis:
    def is_stationary(self, time_series, column_name):
        """
        Checks if a time series is stationary or not.
        :param time_series: pandas dataframe with datetime index
        :param column_name: string
        :return: true or false
        """
        is_stationary = False
        significance_level = 0.05

        log.info("Augmented Dickey Fuller Test (ADH Test)")

        # ADF Test
        result = adfuller(time_series[str(column_name)], autolag='AIC')
        log.info(f'ADF Statistic: {result[0]}')
        p_value = result[1]
        log.info(f'p-value: {p_value}')

        # Print Critical Values
        for key, value in result[4].items():
            log.info('Critial Values:')
            log.info(f'   {key}, {value}')

        # Stationarity check
        log.info("The null hypothesis is that the time series has a unit root and is non-stationary.")
        if p_value <= significance_level:
            is_stationary = True
            log.info(f"The p-value ({p_value}) is less than the significance level ({significance_level}).")
            log.info(f"The time series of {column_name} are stationary.")
            log.info(f"Therefore, the null hypothesis is rejected.")
        else:
            log.info(f"The p-value ({p_value}) is greater than the significance level ({significance_level}).")
            log.info(f"The time series of {column_name} are non-stationary.")
            log.info(f"Therefore, the null hypothesis is accepted.")
        return is_stationary

    def is_trend_stationary(self, time_series, column_name):
        """
        Checks if a time series is stationary or not.
        :param time_series: pandas dataframe with datetime index
        :param column_name: string
        :return: true or false
        """
        is_trend_stationary = False
        significance_level = 0.05

        log.info("Kwiatkowski-Phillips-Schmidt-Shin (KPSS test)")

        # KPSS Test
        result = kpss(time_series[str(column_name)], regression='c')
        log.info('\nKPSS Statistic: %f' % result[0])
        p_value = result[1]
        log.info(f'p-value: {p_value}')

        # Print Critical Values
        for key, value in result[3].items():
            log.info('Critial Values:')
            log.info(f'   {key}, {value}')

        # Trend Stationarity check
        log.info("The null hypothesis is that the time series has a unit root and is trend-stationary.")
        if p_value <= significance_level:
            is_trend_stationary = True
            log.info(f"The p-value ({p_value}) is greater than the significance level ({significance_level}).")
            log.info(f"The time series of {column_name} are non-trend-stationary.")
            log.info(f"Therefore, the null hypothesis is accepted.")
        else:
            log.info(f"The p-value ({p_value}) is less than the significance level ({significance_level}).")
            log.info(f"The time series of {column_name} are trend-stationary.")
            log.info(f"Therefore, the null hypothesis is rejected.")
        return is_trend_stationary

    def execute_preliminary_analysis(self, time_series):
        # New Confirmed Cases
        log.info("For New Confirmed Cases...")
        is_stationary = self.is_stationary(time_series, "New Confirmed Cases")
        log.info(f"Is time series data stationary: {is_stationary}")

        is_trend_stationary = self.is_trend_stationary(time_series, "New Confirmed Cases")
        log.info(f"Is time series data trend-stationary: {is_trend_stationary} \n")

        # New Deaths
        log.info("For New Deaths...")
        is_stationary = self.is_stationary(time_series, "New Deaths")
        log.info(f"Is time series data stationary: {is_stationary}")

        is_trend_stationary = self.is_trend_stationary(time_series, "New Deaths")
        log.info(f"Is time series data trend-stationary: {is_trend_stationary} \n")

        # New Recovered Cases
        log.info("For New Recovered Cases...")
        is_stationary = self.is_stationary(time_series, "New Recovered Cases")
        log.info(f"Is time series data stationary: {is_stationary}")

        is_trend_stationary = self.is_trend_stationary(time_series, "New Recovered Cases")
        log.info(f"Is time series data trend-stationary: {is_trend_stationary} \n")

        # New Active Cases
        log.info("For New Active Cases...")
        is_stationary = self.is_stationary(time_series, "New Active Cases")
        log.info(f"Is time series data stationary: {is_stationary}")

        is_trend_stationary = self.is_trend_stationary(time_series, "New Active Cases")
        log.info(f"Is time series data trend-stationary: {is_trend_stationary} \n")
