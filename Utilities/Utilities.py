# Author: Ioannis Matzakos | Date: 20/12/2019

import csv


def write_csv(data, filename):
    csv_file = open(f'{filename}.csv', 'w')
    with csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(data)
