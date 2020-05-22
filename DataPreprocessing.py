# Author: Ioannis Matzakos | Date: 01/05/2020

import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from Utilities import Log

# Configure logger
log = Log.setup_logger("data_preprocessing")


class DataPreprocessing:

    def __init__(self):
        self.file = ""

    def set_file(self, file):
        self.file = file

    def get_file(self):
        return self.file

    def set_dataframe_name(self, df, name):
        """
        Sets the name of a pandas dataframe
        :param df: pandas dataframe
        :param name: string
        :return: pandas dataframe
        """
        log.info(f"Setting name: {name} for data frame")
        df.name = name
        return df

    def load_csv(self, filepath, df_name):
        """
        Reads a csv file and imports the data into a pandas dataframe.
        :param filepath: string
        :param df_name: string
        :return: pandas dataframe
        """
        log.info(f"Loading file: {filepath}")
        csv_data = pd.read_csv(filepath)
        csv_data = self.set_dataframe_name(csv_data, df_name)
        return csv_data

    def transpose_dataframe(self, df):
        """
        Turns rows into columns and vice versa.
        :param df: pandas dataframe
        :return: pandas dataframe
        """
        df_transposed = df.transpose()
        df_transposed.index.names = ['Date']
        df_transposed.rename(columns={0: 'Confirmed Cases'}, inplace=True)
        log.info(f"Transposed data frame {df.name} column names \n{df_transposed.columns}")
        log.info(f"{df.name}: \n{df_transposed}")
        return df_transposed

    def delete_column(self, time_series, columns_list):
        """
        Deletes a column from pandas dataframe.
        :param time_series:
        :param columns_list:
        :return: pandas dataframe
        """
        new_time_series = time_series
        new_time_series = new_time_series.drop(columns_list, 1)
        log.info(f"New Time Series Data Frame: \n{new_time_series}")
        return new_time_series

    def truncate_time_series(self, time_series, start_date):
        time_series = time_series[start_date:]
        log.info(f"time series data after date {start_date}: {time_series}")
        return time_series

    def stationarize(self, time_series):
        """
        Converts non-stationary time series into stationary.
        :param time_series: pandas dataframe with datetime index
        :return: the transformed pandas dataframe with datetime index
        """

# TODO: make more methods from the preprocessing() method
    def preprocessing(self):
        """
        This is the driver method of Data Preprocessing. Makes all the necessary transformations.
        :return: a pandas dataframe with datetime index
        """
        # Load time series data
        log.info("Loading Time Series Data")
        confirmed = pd.read_csv("Data/time_series_covid19_confirmed_greece.csv")
        log.info(f"Covid-19 confirmed cases in Greece \n{confirmed}")

        deaths = pd.read_csv("Data/time_series_covid19_deaths_greece.csv")
        log.info(f"Deaths due to Covid-19 in Greece \n{deaths}")

        recovered = pd.read_csv("Data/time_series_covid19_recovered_greece.csv")
        log.info(f"Covid-19 recovered cases in Greece \n{recovered}")

        # TODO: Perform Preliminary Analysis

        # TODO: Remove unwanted columns from initial time series dataset

        # Transpose the three pandas data frames in order columns to become rows
        # and establish two columns: Date and Number of Cases, in the next phase
        log.info("Transposing the Data Frames")

        confirmed_cases = confirmed.transpose()
        confirmed_cases.index.names = ['Date']
        confirmed_cases.rename(columns={0: 'Confirmed Cases'}, inplace=True)
        log.info(f"Covid-19 confirmed cases in Greece column names \n{confirmed_cases.columns}")
        log.info(f"Covid-19 confirmed cases in Greece \n{confirmed_cases}")

        death_cases = deaths.transpose()
        death_cases.index.names = ['Date']
        death_cases.rename(columns={0: 'Deaths'}, inplace=True)
        log.info(f"Deaths due to Covid-19 in Greece \n{death_cases}")

        recovered_cases = recovered.transpose()
        recovered_cases.index.names = ['Date']
        recovered_cases.rename(columns={0: 'Recovered Cases'}, inplace=True)
        log.info(f"Covid-19 recovered cases in Greece \n{recovered_cases}")

        # Merge the three transposed data frames into one in order to have the following
        # columns: Date, Confirmed Cases, Deaths and Recovered Cases
        log.info("Merging Data Frames")

        df = confirmed_cases.merge(death_cases, left_index=True, right_index=True)
        df = df.merge(recovered_cases, left_index=True, right_index=True)
        # pd.set_option("display.max_rows", None, "display.max_columns", None)
        log.info(f"Covid-19 Cases in Greece \n{df}")

        # Convert data frame index to datetime index
        log.info(f"data frame index: {df.index}")
        time_series = df.reset_index()
        log.info(f"data frame: \n{time_series.head()}")
        date_rng = pd.date_range(start=str(time_series['Date'].iloc[0]),
                                 end=str(time_series['Date'].iloc[len(time_series) - 1]))
        time_series['DateTime'] = date_rng
        log.info(f"data frame: \n{time_series.head()}")
        time_series['DateTime'] = pd.to_datetime(time_series['DateTime'])
        time_series = time_series.set_index('DateTime')
        time_series = time_series.drop('Date', 1)
        time_series.index.names = ['Date']
        log.info(f"data frame: \n{time_series.head()}")
        log.info(f"data frame index: {time_series.index}")

        return time_series
