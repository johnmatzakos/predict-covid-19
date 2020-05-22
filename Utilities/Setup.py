# Author: Ioannis Matzakos | Date: 05/02/2020

from Utilities import Log
import time

# configure logger
log = Log.setup_logger("setup")

log.info("Start of Execution\n\n\t\t\t\t---------- PROPAGANDA RECOGNITION: SETUP ----------\n")

# start calculating execution time
start_time = time.time()

# extract dependencies
# pip freeze > requirements.txt

# install the required python modules
# pip install dependencies.txt

# stop calculating execution time
end_time = time.time()
total_time = end_time - start_time
log.info(f"Execution Time: {total_time} seconds")

log.info("Start of Execution\n\n\t\t\t\t---------- SETUP COMPLETED SUCCESSFULLY ----------\n")