##############################
# Churn library constants
##############################
'''
This modul delivers the constants values for 
data and model handling of the Churn Customer ML Project.

It includes an additional data string functionality for file labelling
and an additional function for getting the appropriate log file.

Date: 2021-12-26
Author: I. Brinkmeier
'''

#
# imports
#
from datetime import datetime


# Create current date string
current_date = datetime.today()
date_str = current_date.strftime('%Y-%m-%d')

#
# Constants
#

# directories
DATA_DIR = './data/'
LOG_DIR = './logs/'
IMAGES_DIR = './images/'
EDA_DIR = './images/eda/'
EDA_TEST_DIR = './test/eda/'

# data file
BANK_DATA = 'bank_data.csv'

# EDA profiling report and diagrams of datasets
EDA_REPORT_ORIG = 'EDA_churndata_profile_' + date_str + '.html'
BOXPLOT_FILENAME = 'Boxplot_DataTendency.png'
SPEARMAN_CORR_FILENAME = 'Spearman_FeatCorrelation.png'
SCATTERPLOT_FILENAME = 'ScatterplotByChurn.png'
DIST_TOTALTRANS_FILENAME = 'DistributionTotalTransactions.png'
DIST_CHURNERS_FEWPROPS_FILENAME = 'DistributionChurnersByFewProps.png'
DIST_MARITALSTAT_FILENAME = 'DistributionCustomersMaritalStatus.png'
DIST_AGE_FILENAME = 'DistributionCustomersAge.png'
AMOUNT_SEX_FILENAME = 'AmountOfChurnersBySex.png'

# model storage
MODELS_DIR = './models/'

# model evaluation png storage 
RESULTS_DIR = './images/results/'
RESULTS_TEST_DIR = './test/results/'


#
# Set Logging
#

# log file
LOG_FILE_PROJ = LOG_DIR + 'churn_library_proj_' + date_str + '.log'
LOG_FILE_TEST = LOG_DIR + 'churn_library_test_' + date_str + '.log'


def get_log_file(workflow=""):
    '''
    Returns the log file string.
    
    input:
        workflow: (str) the workflow case for test or project
    output:
        file name of log file, specific one for test cases
    '''
    print(f'workflow string: {workflow}')
    if workflow == 'test':
        return LOG_FILE_TEST

    return LOG_FILE_PROJ
