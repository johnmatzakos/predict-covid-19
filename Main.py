# Author: Ioannis Matzakos | Date: 01/05/2020

# import python modules needed
import time

# import project's classes and scripts
from Utilities import Log
from Utilities import Constants
from PreliminaryAnalysis import PreliminaryAnalysis
from DataPreprocessing import DataPreprocessing
from FeatureEngineering import FeatureEngineering
from DataVisualization import DataVisualization
from DataMining import DataMining

# Configure logger
log = Log.setup_logger("main")

log.info(Constants.INITIAL_MSG)

# Start calculating execution time
start_time = time.time()

log.info(Constants.START_MSG)

# Data Preprocessing Phase
log.info(Constants.DATA_PREPROCESSING_MSG)
dp = DataPreprocessing()
time_series = dp.preprocessing()

# Feature Engineering Phase
log.info(Constants.FEATURE_ENGINEERING_MSG)
fe = FeatureEngineering()

new_time_series = fe.execute_feature_engineering(time_series)

# new_time_series = dp.delete_column(time_series, ['Confirmed Cases', 'Deaths', 'Recovered Cases', 'Active Cases'])

# Truncate zero values from the time series
# new_time_series = dp.truncate_time_series(new_time_series, '26/02/2020')

# Preliminary Analysis: Stationarity Check
pa = PreliminaryAnalysis()

pa.execute_preliminary_analysis(new_time_series)

# Data Visualization Phase
log.info(Constants.DATA_VISUALIZATION_MSG)
dv = DataVisualization()

# original time series data
dv.plot_data_structure(time_series)
# new time series data
dv.plot_data_structure(new_time_series)

# plots of all cases
dv.plot_data_structure(new_time_series['New Confirmed Cases'])
dv.plot_data_structure(new_time_series['New Deaths'])
dv.plot_data_structure(new_time_series['New Recovered Cases'])
dv.plot_data_structure(new_time_series['New Active Cases'])

# Decomposition plots for:
# Confirmed Cases
x = new_time_series['New Confirmed Cases'].resample('D').mean()
dv.get_decomposition_plot(x)

# Deaths
y = new_time_series['New Deaths'].resample('D').mean()
dv.get_decomposition_plot(y)

# Recovered Cases
z = new_time_series['New Recovered Cases'].resample('D').mean()
dv.get_decomposition_plot(z)

# Active Cases
k = new_time_series['New Active Cases'].resample('D').mean()
dv.get_decomposition_plot(k)

# Data Mining Phase (Time Series Forecasting)
log.info(Constants.DATA_MINING_MSG)
dm = DataMining()

# dm.execute_process(new_time_series, x, 'Confirmed Cases')
# dm.execute_process(new_time_series, y, 'Deaths')
# dm.execute_process(new_time_series, z, 'Recovered Cases')
dm.execute_process(new_time_series, k, 'Active Cases')

# dm.execute_mining_with_prophet(new_time_series, 'Active Cases')

log.info(Constants.END_MSG)

# stop calculating execution time
end_time = time.time()
total_time = end_time - start_time
log.info(f"Execution Time: {total_time} seconds")

log.info(Constants.INITIAL_MSG)
