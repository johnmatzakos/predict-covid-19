# Author: Ioannis Matzakos | Date: 01/05/2020

# import python modules needed
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import statsmodels.api as sm
import itertools
import warnings
from pylab import rcParams

# import project's classes and scripts
from Utilities import Log

# Configure logger
log = Log.setup_logger("data_visualization")

# Configure pyplot
plt.style.use('fivethirtyeight')
warnings.filterwarnings("ignore")

# Configure matplotlib
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'


class DataVisualization:

    def plot_data_structure(self, dataframe):
        dataframe.plot(figsize=(15, 6))
        plt.show()

    def get_decomposition_plot(self, x):
        rcParams['figure.figsize'] = 18, 8
        decomposition = sm.tsa.seasonal_decompose(x, model='additive')
        fig = decomposition.plot()
        plt.show()
