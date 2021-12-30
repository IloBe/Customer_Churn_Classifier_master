##################################
# Churn test library doc string
##################################
'''
This modul delivers the test functions for the main coding activities
regarding data and model handling of the Churn Customer ML Project.

Notes:
In general, regarding the concept of test-driven-development,
I would use some test data which are similar to the real one
together with a test environment and its specific constants
being different compared to the project/product ones.
Because of project time I am using the same data as delivered for the project,
focus of the project is a general workflow concepts, not content.
Furthermore, I would prefer the unittest library and its assert statements.
see: https://docs.python.org/3/library/unittest.html#assert-methods
This activities can be handled as future toDo.

Date: 2021-12-26
Author: I. Brinkmeier
'''

##################################
# import libraries
##################################

# general
import os
import sys
import shutil
import logging
from datetime import datetime
import pandas as pd
import numpy as np

# project ones
import churn_library as cl
from config import churn_config as cfg

##################################
# Administrative
##################################

# Create current date string
current_date = datetime.today()
date_str = current_date.strftime('%Y-%m-%d')

#
# Set Logging
#
logging.basicConfig(
    filename=cfg.get_log_file(''),  # future toDo: shall be its own test log file
    level=logging.INFO,
    filemode='w',
    format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')

# Tell us about versions of main libraries, clearly visible in log file
# note: using f'' concept is not working with pylint
logging.info('Project Customer Churn - Main Library Versioning of Test File:')
logging.info('- Python: %s', sys.version)
logging.info('- Pandas: %s', pd.__version__)
logging.info('===============================')


##################################
# Test Coding
##################################

def test_import_data(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions;
    my note: nevertheless, the code is partly modified
    '''
    print("-----------")
    filename = os.path.join(cfg.DATA_DIR, cfg.BANK_DATA)
    logging.info("--- Start testing import_data() with file '%s'", filename)
    try:
        df = import_data(filename)
    except FileNotFoundError as err:
        logging.error(
            "Test import_data: ERROR: The file to import isn't found.")
        raise err

    # check if created dataframe is usable
    try:
        assert df is not None, "The file could not be mapped to the needed dataframe; df is None."
        assert df.empty is False, "The dataframe df includes no items, it is empty."
        assert df.shape[0] > 0, "Initial df dataframe: No data rows exist."
        assert df.shape[1] > 0, "Initial df dataframe: No data columns exist."
        logging.info("TEST import_data(): SUCCESS")
    except AssertionError as err:
        logging.error(
            "Test import_data output: ERROR: The file doesn't have rows and columns or is None.")
        raise err


def construct_dir(new_dir):
    '''
    Creates given directory.
    Delete existing folder and create it again, e.g. for new plots created in this notebook.

    input:
        new_dir: (str) directory name
    output:
        None
    '''
    logging.info("Try to create given directory: %s", new_dir)
    try:
        shutil.rmtree(new_dir)
        os.mkdir(new_dir)
    except FileNotFoundError as ferr:
        logging.error("ERROR: Directory not found, create it. ferr: %s", ferr)
        os.mkdir(new_dir)
    except PermissionError as perr:
        logging.error(
            "ERROR: You are not allowed to create directory: perr: %s", perr)
        raise perr


def get_plot_items(eda_dir):
    '''
    Returns the data diagram path and names list created during EDA activity (perform_eda()).

    input:
        eda_dir: (str) directory name of stored eda files
    output:
        list of diagram labels and path
    '''
    counter = 0
    vis_data = []
    temp = []

    # get all plots
    files = os.listdir(eda_dir)

    # iterate over all created visualisations
    for fname in files:
        if counter == 1:
            vis_data.append(temp)
            temp = []
            counter = 0

        temp.append(f'{eda_dir}/{fname}')
        counter += 1

    return [*vis_data, temp]


def test_perform_eda(perform_eda):
    '''
    test perform eda function and the creation of the overall EDA profile report
    '''
    print("-----------")
    filename = os.path.join(cfg.DATA_DIR, cfg.BANK_DATA)
    logging.info("--- Start testing perform_eda() with file '%s'", filename)
    df_bank_data = cl.import_data(filename)
    df = cl.prepare_df(df_bank_data)

    try:
        # create a eda test dir; not miss up testing with project file creation
        construct_dir(cfg.EDA_TEST_DIR)
    except (FileNotFoundError, PermissionError) as err:
        logging.error("ERROR: './test/eda/' directory creation failed.")
        raise err

    try:
        perform_eda(df, out_pth=cfg.EDA_TEST_DIR)
        # check if all 9 files of report and plot creation parts are available
        plots = get_plot_items(cfg.EDA_TEST_DIR)
        assert len(
            plots) == 9, "Not all 9 plots are created and stored in EDA_DIR by perform_eda()."
        assert os.path.isfile("./test/eda/EDA_churndata_profile_" + date_str + ".html"), \
            "EDA profile report has not been stored."
        assert os.path.isfile("./test/eda/Spearman_FeatCorrelation.png"), \
            "EDA 'Spearman_FeatCorrelation.png' has not been stored."
        assert os.path.isfile("./test/eda/ScatterplotByChurn.png"), \
            "EDA 'ScatterplotByChurn.png' has not been stored."
        assert os.path.isfile("./test/eda/DistributionTotalTransactions.png"), \
            "EDA 'DistributionTotalTransactions.png' has not been stored."
        assert os.path.isfile("./test/eda/DistributionCustomersMaritalStatus.png"), \
            "EDA 'DistributionCustomersMaritalStatus.png' has not been stored."
        assert os.path.isfile("./test/eda/DistributionCustomersAge.png"), \
            "EDA 'DistributionCustomersAge.png' has not been stored."
        assert os.path.isfile("./test/eda/DistributionChurnersByFewProps.png"), \
            "EDA 'DistributionChurnersByFewProps.png' has not been stored."
        assert os.path.isfile("./test/eda/Boxplot_DataTendency.png"), \
            "EDA 'Boxplot_DataTendency.png' has not been stored."
        assert os.path.isfile("./test/eda/AmountOfChurnersBySex.png"), \
            "EDA 'AmountOfChurnersBySex.png' has not been stored."
        logging.info("TEST perform_eda(): SUCCESS")
    except AssertionError as err:
        logging.error("ERROR: perform_eda() test failed.")
        raise err


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''
    print("-----------")
    filename = os.path.join(cfg.DATA_DIR, cfg.BANK_DATA)
    df_bank_data = cl.import_data(filename)
    df = cl.prepare_df(df_bank_data)

    cat_lst_correct = [
        'Sex',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    cat_lst_wrong = ['Education_Level', 3, 4.8]

    try:
        logging.info("--- Start testing encoder_helper() with bank dataframe.")
        # check if correct list includes categorical items
        assert len(
            cat_lst_correct) > 0, "Categorical list does not include items."
        logging.info(
            "SUCCESS: Check of amount of categorical items > 0: '%s'.",
            len(cat_lst_correct))
        assert all(isinstance(val, str) for val in cat_lst_correct),\
            "Not all items are categorical strings."
        assert all(val != "" for val in cat_lst_correct),\
            "Some categorical items are empty strings."
        logging.info(
            "SUCCESS: All list elements are categorical, non-empty items.")

        assert df.empty is False,\
            "Dataframe df for encoding_helper test is empty."
        logging.info("SUCCESS: Dataframe is not empty.")

        # with appropriate params call function shall work
        df_test = encoder_helper(df, cat_lst_correct, 'Churn')

        # check created dataframe
        assert df_test.empty is False,\
            "Dataframe df_test created by encoder helper test is empty."

        # check if original cat columns don't exist anymore in the test df columns list
        assert (list(df_test.columns) != cat_lst_correct) is True,\
            "Original categorical columns df_test.columns are the same."
        assert any(item in list(df_test.columns) for item in cat_lst_correct) is False,\
            "Any original categorical columns still exist in df_test."
        logging.info(
            "SUCCESS: Original & encoded df have same shape, datatypes no, correct labels.")
        # in new df_test no object, i.e. string datatype shall exist
        dtype_object_test = df_test.dtypes[df_test.dtypes == np.object]
        assert len(list(dtype_object_test.index)) == 0,\
            "Not all categorical features encoded."
        logging.info("TEST encoder_helper(): SUCCESS with correct categorical list.")
    except AssertionError as err:
        logging.error("ERROR: encoder_helper() test fails.")
        raise err

    try:
        # check if wrong list includes categorical and none-categorical items
        assert len(cat_lst_wrong) > 0
        logging.info(
            "SUCCESS: Check of amount of categorical items > 0: '%s'.",
            len(cat_lst_wrong))
        assert any(isinstance(val, str) for val in cat_lst_wrong),\
            "No item is a categorical string."
        assert any(isinstance(val, int)
                   for val in cat_lst_wrong), "No item is an int."
        assert any(isinstance(val, float)
                   for val in cat_lst_wrong), "No item is a float."
        logging.info(
            "SUCCESS: List elements are a mixture of categorical & non- categorical items.")

        # encoder_helper() function must fail with assertion error
        df_test = encoder_helper(df, cat_lst_wrong, 'Churn')
        logging.error("ERROR: TEST encoder_helper(): Wrong categorical list test fails.")
        raise AssertionError("ERROR: TEST encoder_helper(): Wrong categorical list test fails.")
    except AssertionError as err:
        logging.info(
            "SUCCESS: Wrong categorical encoder_helper test fails as expected: %s", err)


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''
    print("-----------")
    filename = os.path.join(cfg.DATA_DIR, cfg.BANK_DATA)
    df_bank_data = cl.import_data(filename)
    df = cl.prepare_df(df_bank_data)

    try:
        logging.info(
            "--- Start testing perform_feature_engineering() with bank dataframe.")
        assert df.empty is False, "The dataframe df for perform feature engineering test\
            includes no items, it is empty."

        X_train, X_test, y_train, y_test, train_label_0, train_label_1 =\
            perform_feature_engineering(df=df)

        assert X_train is not None, "Train models function param 'X_train' does not exist."
        assert X_test is not None, "Train models function param 'X_test' does not exist."
        assert y_train is not None, "Train models function param 'y_train' does not exist."
        assert y_test is not None, "Train models function param 'y_test' does not exist."
        assert train_label_0 is not None,\
            "Train models function param train_label_0' does not exist."
        assert train_label_1 is not None,\
            "Train models function param 'train_label_1' does not exist."
        assert isinstance(X_train, pd.core.frame.DataFrame) is True,\
            "'X_train' param is no dataframe type."
        assert isinstance(X_test, pd.core.frame.DataFrame) is True,\
            "'X_test' param is no dataframe type."
        assert isinstance(
            y_train, pd.core.series.Series) is True, "'y_train' param is no series type."
        assert isinstance(
            y_test, pd.core.series.Series) is True, "'y_test' param is no series type."
        assert isinstance(
            train_label_0, int) is True, "'train_label_0' param is no int type."
        assert isinstance(
            train_label_1, int) is True, "'train_label_1' param is no int type."
        logging.info("TEST perform_feature_engineering(): SUCCESS")
    except AssertionError as err:
        logging.error("ERROR: perform_feature_engineering() test fails.")
        raise err


def test_train_models(train_models):
    '''
    test train_models
    '''
    print("-----------")
    filename = os.path.join(cfg.DATA_DIR, cfg.BANK_DATA)
    df_bank_data = cl.import_data(filename)
    df = cl.prepare_df(df_bank_data)
    cat_columns = df.select_dtypes(include='object').columns
    df = cl.encoder_helper(df=df, category_lst=cat_columns, response="Churn")

    X_train, X_test, y_train, y_test, train_label_0, train_label_1 = \
        cl.perform_feature_engineering(df=df)

    try:
        logging.info("--- Start testing train_models() with bank dataframe.")
        dict_pred_results = train_models(X_train, X_test, y_train, y_test,
                                         train_label_0, train_label_1)

        assert dict_pred_results is not None, \
            "'dict_pred_results' is none, no classifier predictions exist."
        assert len(dict_pred_results) == 3, \
            "Wrong amount of classifiers in 'dict_pred_results', not 3."
        assert dict_pred_results.get('LogisticRegression') is not None, \
            "Key 'LogisticRegression' has no value in dict_pred_results."
        assert dict_pred_results.get('RandomForestClassifier') is not None, \
            "Key 'RandomForestClassifier' has no value in dict_pred_results."
        assert dict_pred_results.get('XGBClassifier') is not None, \
            "Key 'XGBClassifier' has no value in dict_pred_results."

        # equal items creation for each classifier exist, therefore check only LogisticRegression
        # check the dictionary value elements
        assert dict_pred_results.get('LogisticRegression')[0] is not None,\
            "LogReg y train predictions do not exist."
        assert dict_pred_results.get('LogisticRegression')[1] is not None,\
            "LogReg y test predictions do not exist."
        assert dict_pred_results.get('LogisticRegression')[2] is not None,\
            "LogReg pipe pkl model does not exist."
        assert dict_pred_results.get('LogisticRegression')[3] is not None,\
            "LogReg test report does not exist."
        assert dict_pred_results.get('LogisticRegression')[4] is not None,\
            "LogReg train report does not exist."

        # logistic regression pipe pkl model is stored in model dir
        assert os.path.isfile("./models/best_churn_clf_lr.pkl"), \
            "train_models() test lr pipe pickle model file has not been stored."
        logging.info("TEST train_models(): SUCCESS")
    except AssertionError as err:
        logging.error("ERROR: train_models() test fails.")
        raise err


##############################
# Main Call
##############################

def main():
    '''
    Workflow of churn classifier library test file started by command line interface.
    '''
    print("--- Churn test workflow of " + date_str + " ---")
    print("--- TEST start of import_data() ---")
    test_import_data(cl.import_data)
    print("--- TEST start of perform_eda() ---")
    test_perform_eda(cl.perform_eda)
    print("--- TEST start of encoder_helper() ---")
    test_encoder_helper(cl.encoder_helper)
    print("--- TEST start of perform_feature_engineering() ---")
    test_perform_feature_engineering(cl.perform_feature_engineering)
    print("--- TEST start of train_models() ---")
    test_train_models(cl.train_models)

if __name__ == "__main__":
    main()
