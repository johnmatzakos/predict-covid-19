# Author: Ioannis Matzakos | Date: 01/05/2020

# import python modules needed

# import project's classes and scripts
from Utilities import Log

# Configure logger
log = Log.setup_logger("feature_engineering")


class FeatureEngineering:

    def calculate_cases(self, time_series, column_name):
        """Calculates the daily new cases, new deaths and new recovered cases
        :param time_series: pandas dataframe
        :param column_name: string
        :return: list
        """
        # Initialize the list for the new calculated feature
        new_cases_list = list()
        log.debug(f"new_cases_list list size before {len(new_cases_list)}")

        # create a list of the data source (a data frame column)
        sum_cases_list = time_series[str(column_name)].tolist()
        log.info(f"Source list size: {len(sum_cases_list)}")

        # Create features
        new_cases_list.append(0)
        i = 0
        for i in range(len(sum_cases_list)-1):
            new_cases = sum_cases_list[i+1] - sum_cases_list[i]
            new_cases_list.append(new_cases)

        log.debug(f"new_cases_list size after {len(new_cases_list)}")
        return new_cases_list

    def calculate_active_cases(self, time_series):
        """Calculates the daily active cases by subtracting deaths and recovered cases from total confirmed cases
        :param time_series: pandas dataframe
        :return: list
        """
        log.info("Caculating the daily active cases...")
        active_cases_list = list()

        # Converting time series dataframe columns to individual lists
        cases_list = time_series['Confirmed Cases'].tolist()
        deaths_list = time_series['Deaths'].tolist()
        recovered_list = time_series['Recovered Cases'].tolist()

        # Subtract deaths and recovered cases from total confirmed cases
        active_cases_list.append(0)
        i = 0
        for i in range(len(cases_list) - 1):
            active_cases = cases_list[i] - deaths_list[i] - recovered_list[i]
            active_cases_list.append(active_cases)

        log.info(f"Active Cases: {active_cases_list}")
        return active_cases_list

    def execute_feature_engineering(self, time_series):
        # Calculate active cases and add them into the dataframe
        active_cases_list = self.calculate_active_cases(time_series)
        log.debug(f"active_cases_list length {len(active_cases_list)}")
        log.debug(f"time_series length {len(time_series)}")
        time_series['Active Cases'] = active_cases_list

        # Calculate new cases
        new_confirmed_cases_list = self.calculate_cases(time_series, 'Confirmed Cases')
        new_deaths_list = self.calculate_cases(time_series, 'Deaths')
        new_recovered_cases_list = self.calculate_cases(time_series, 'Recovered Cases')
        new_active_cases_list = self.calculate_cases(time_series, 'Active Cases')

        # Add the new features in a new dataframe
        new_time_series = time_series
        time_series['New Confirmed Cases'] = new_confirmed_cases_list
        time_series['New Deaths'] = new_deaths_list
        time_series['New Recovered Cases'] = new_recovered_cases_list
        time_series['New Active Cases'] = new_active_cases_list

        new_time_series = new_time_series.drop(['Confirmed Cases', 'Deaths', 'Recovered Cases', 'Active Cases'], 1)
        log.info(f"New Time Series Data Frame: \n{new_time_series.head()}")

        return new_time_series
