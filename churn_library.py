##############################
# Churn library doc string
##############################
'''
This modul delivers the helper functions to structure the main coding activities for
data and model handling of Customer Churn ML Project. It starts via command line interface.

It includes an additional state-of-the-art classifier - XGBClassifier from xgboost library -
beside the 2 proposed ones 'LogisticRegression' and 'RandomForestClassifier'.
In general, tree classifiers are better to handle the imbalanced dataset.
XGBClassifier includes a specific parameter to handle the specific target values.

Notes:
- plot_roc_curve() is deprecated with scikit-learn version 1.0 and will be removed with 1.2,
So, we use RocCurveDisplay as proposed by the scikit-learn team.
-  During training of XGBClassifier an internal source code message appeared:
'ntree_limit is deprecated, use `iteration_range` or model slicing instead.'
Such parameter is not set in this coding; this message is put as issue on GitHub already.


Date: 2021-12-25
Author: I. Brinkmeier
'''

##############################
# import libraries
##############################

# general
import os
import sys
import logging
from collections import Counter
from datetime import datetime
import pandas as pd
import pandas_profiling as pdp
import numpy as np
import joblib

# classifier model handling
import sklearn
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, make_scorer, RocCurveDisplay, classification_report
import xgboost
from xgboost import XGBClassifier

# visualisation
import shap
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

# own ones, put at the end will throw minimum of pylint import order messages
from config import churn_config as cfg

##############################
# Administration
##############################

# Create current date string
current_date = datetime.today()
date_str = current_date.strftime('%Y-%m-%d')

#
# Set Logging
#
logging.basicConfig(
    filename=cfg.get_log_file('proj'),  # LOG_FILE for project
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


##############################
# Function Coding
##############################

#
# Administrative
#
def get_library_version_info():
    '''
    Logs main libraries version information as starting point of the log file.
    '''
    # using f'' concept is not working with pylint
    logging.info('Project Customer Churn - Main Library Versioning:')
    logging.info('- Python: %s', sys.version)
    logging.info('- Pandas: %s', pd.__version__)
    logging.info('- Pandas_Profiling: %s', pdp.__version__)
    logging.info('- scikit_learn: %s', sklearn.__version__)
    logging.info('- xgboost: %s', xgboost.__version__)
    logging.info('- SHAP: %s', shap.__version__)
    logging.info('- matplotlib: %s', matplotlib.__version__)
    logging.info('- seaborn: %s', sns.__version__)
    logging.info('===============================')


def import_data(file_pth):
    '''
    Returns dataframe for the .csv file found at given path variable input or None otherwise.
    Checks the shape of the dataframe not being empty.

    input:
        file_pth: (str) a path to the .csv file
    output:
        df: (DataFrame) pandas dataframe, if file is read, None otherwise
    '''
    df_data = None  # pylint expects a return value for each part

    try:
        assert len(file_pth) > 0
        assert isinstance(
            file_pth, str) is True, "No file path string given for data import."
        assert file_pth.lower().endswith(
            '.csv') is True, "No .csc file given for data import."
        logging.info("Start to read in file '%s'.", file_pth)
    except AssertionError as err:
        logging.error(
            "ERROR: File and path string is empty or no .csv file info given.")
        raise err

    try:
        df_data = pd.read_csv(file_pth)
        assert df_data is not None, \
            "No initial df dataframe has been assigned to df_data variable, still None."
        assert df_data.empty is False, "The dataframe df includes no items, it is empty."
        logging.info("SUCCESS: the file is not empty and read in.")
    except AssertionError as err:
        logging.error(
            "ERROR: The file is not stored or doesn't have rows and columns.")
        raise err

    return df_data


def prepare_df(df):
    '''
    Performs some specific data preparation on the given dataframe and returns the modified one:
    - the target column 'Churn' with binary values 0 and 1 is build out of 'Attrition_Flag' column,
      that categorical column is removed afterwards
    - unneeded columns with no added value are removed:
        'Unnamed: 0' (same as index),
        'CLIENTNUM' (no added value for churn task, still available in origina read-in dataframe
    - rename of 'Gender' column to 'Sex',
        because regarding British English gender is a role and not associated with biological sex
    - rename 'Dependent_count' to 'Dependent_Count' having same df label structure
    - removes duplicated rows, if available
    Finally, the new dataframe structure and information about features null values sums are logged.
    (Note: no null values exists in original dataframe)

    input:
        df: pandas dataframe
    output:
        df: modified pandas dataframe, if modifications are possible, given df param otherwise
    '''
    try:
        assert df is not None, "File could not be mapped to the needed dataframe; df is None."
        assert df.empty is False, "Dataframe df includes no items, it is empty."
        assert {'Attrition_Flag', 'Gender', 'Unnamed: 0', 'CLIENTNUM'}.issubset(
            df.columns) is True, "Needed features 'Churn', 'Sex' not available in df."

        logging.info("Start to perform df preprocessing ...")
        # starting with target value y:
        # 0 and 1 values needed as classification target of customer churn ML algorithms
        # afterwards the column 'Attrition_Flag' and other ones are not needed
        # anymore
        df['Churn'] = df['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)
        df.drop(
            columns=[
                'Attrition_Flag',
                'Unnamed: 0',
                'CLIENTNUM'],
            inplace=True)

        if sum(df.duplicated()) > 0:
            df.drop_duplicates(inplace=True)
        df.rename(columns={'Gender': 'Sex',
                           'Dependent_count': 'Dependent_Count'},
                  inplace=True)

        logging.info(
            "SUCCESS: Read-in dataframe modifications happen successfully.")
        logging.info('Preprocessed df is:\n %s', df.info())

        # from original dataset: no null values exists, at least we log this
        # property
        logging.info('Preprocessed df includes no null values:')
        logging.info(df.isnull().sum())
    except AssertionError as err:
        logging.error(
            "ERROR: The given df doesn't have appropriate rows and columns.")
        raise err

    return df


#
# EDA
#
def create_eda_profile_report(
    df,
    file_pth=os.path.join(
        cfg.EDA_DIR,
        cfg.EDA_REPORT_ORIG)):
    '''
    Creates overall EDA profile .html report which is stored by default in  an eda folder.

    input:
        df: pandas dataframe
        file_pth: (string) path and name of the report file;
                default for project is "./images/eda/EDA_data_profile.html"
    output:
        profile: data profile report for display in notebook cell and
                storage by default in given file path parameter.
                None returned if the EDA profile report could not be created.
    '''
    profile = None
    try:
        assert df is not None, "File could not be mapped to the needed dataframe; df is None."
        assert df.empty is False, "Dataframe df includes no items, it is empty."
        assert df.shape[0] > 0, "Initial df dataframe: No data rows exist."
        assert df.shape[1] > 0, "Initial df dataframe: No data columns exist."
        logging.info("SUCCESS: the file has valid shape: %s", df.shape)
        profile = df.profile_report()
        profile.to_file(file_pth)
        logging.info(
            "SUCCESS: EDA profile report creation and storage happens successfully.")
    except AssertionError as err:
        logging.error(
            "ERROR: The file doesn't appear to have rows and columns or can be None. \
            No EDA profile report created.")
        raise err

    return profile


def plot_churners_by_sex(
    df,
    file_pth=os.path.join(
        cfg.EDA_DIR,
        cfg.AMOUNT_SEX_FILENAME)):
    '''
    Plots and stores the histogram about churners separated by sex.

    input:
        df: pandas dataframe
        file_pth: (str) path and file name string of diagram, default path is "./images/eda/"
    output:
        None
    '''
    try:
        assert df is not None, "File could not be mapped to the needed dataframe; df is None."
        assert df.empty is False, "Dataframe df includes no items, it is empty."
        assert {'Churn', 'Sex'}.issubset(df.columns) is True, \
            "Needed features 'Churn', 'Sex' not available in df."

        plt.figure(figsize=(5, 6))
        plt.title('Amount of Churners by Sex', fontsize=13, fontweight='bold')

        r = [0.15, 1.05]  # custom x axis, position of the bars on the x-axis
        names = ['False', 'True']
        plt.xticks(r, names, fontsize=12)
        plt.xlabel('Churn', fontsize=12, fontweight='bold')
        plt.ylabel('Count', fontsize=12, fontweight='bold')
        sns.histplot(
            data=df,
            x='Churn',
            hue="Sex",
            multiple="dodge",
            binwidth=0.3,
            shrink=2.2)

        plt.savefig(file_pth, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.clf()
        logging.info(
            "SUCCESS: Successful creation & storage of 'AmountOfChurnersBySex.png'.")
    except AssertionError as err:
        logging.error(
            "ERROR: Plot file 'AmountOfChurnersBySex.png' not created or stored.")
        raise err


def plot_customers_age(
    df,
    file_pth=os.path.join(
        cfg.EDA_DIR,
        cfg.DIST_AGE_FILENAME)):
    '''
    Plots and stores the distribution of customers age.

    input:
        df: pandas dataframe
        file_pth: (str) path and file name string of diagram, default path is "./images/eda/"
    output:
        None
    '''
    try:
        assert df is not None, "File could not be mapped to the needed dataframe; df is None."
        assert df.empty is False, "Dataframe df includes no items, it is empty."
        assert {'Customer_Age'}.issubset(df.columns) is True, \
            "Needed feature 'Customer_Age' not available in df."

        plt.figure(figsize=(8, 6))
        plt.title(
            'Distribution of Customers Age',
            fontsize=13,
            fontweight='bold')
        plt.xlabel('Age', fontsize=12, fontweight='bold')
        plt.ylabel('Count', fontsize=12, fontweight='bold')
        sns.histplot(data=df, x="Customer_Age", bins=10, kde=True)

        plt.savefig(file_pth, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.clf()
        logging.info(
            "SUCCESS: Succesful creation & storage of 'DistributionCustomersAge.png'.")
    except AssertionError as err:
        logging.error(
            "ERROR: Plot file 'DistributionCustomersAge.png' not created or stored.")
        raise err


def plot_marital_status(
    df,
    file_pth=os.path.join(
        cfg.EDA_DIR,
        cfg.DIST_MARITALSTAT_FILENAME)):
    '''
    Plots and stores the distribution of the customers marital status.

    input:
        df: pandas dataframe
        file_pth: (str) path and file name string of diagram, default path is "./images/eda/"
    output:
        None
    '''
    try:
        assert df is not None, "File could not be mapped to the needed dataframe; df is None."
        assert df.empty is False, "Dataframe df includes no items, it is empty."
        assert {'Marital_Status'}.issubset(df.columns) is True, \
            "Needed feature 'Marital_Status' not available in df."

        g = sns.catplot(
            data=df,
            x="Marital_Status",
            kind='count',
            palette='mako_r',
            alpha=0.8)
        g.fig.set_size_inches(6, 5)
        g.set(ylim=(0, 5000))
        plt.title(
            'General Amount of Customers Marital Status',
            fontsize=13,
            fontweight='bold',
        )
        plt.xlabel('Marital Status', fontsize=12, fontweight='bold')
        plt.ylabel('Count', fontsize=12, fontweight='bold')

        plt.savefig(file_pth, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.clf()
        logging.info(
            "SUCCESS: Successful creation & storage of 'DistributionCustomersMaritalStatus.png'.")
    except AssertionError as err:
        logging.error(
            "ERROR: Plot file 'DistributionCustomersMaritalStatus.png' not created or stored.")
        raise err


def plot_churners_by_fewprops(
    df,
    file_pth=os.path.join(
        cfg.EDA_DIR,
        cfg.DIST_CHURNERS_FEWPROPS_FILENAME)):
    '''
    Plots and stores the distribution of churners filtered by few feature properties.

    input:
        df: pandas dataframe
        file_pth: (str) path and file name string of diagram, default path is "./images/eda/"
    output:
        None
    '''
    try:
        assert df is not None, "File could not be mapped to the needed dataframe; df is None."
        assert df.empty is False, "Dataframe df includes no items, it is empty."
        assert {'Marital_Status', 'Customer_Age', 'Churn', 'Sex'}.issubset(df.columns) is True, \
            "Needed features 'Marital_Status', 'Sex', ... not available in df."

        g = sns.catplot(data=df, x="Marital_Status", y="Customer_Age",
                        col="Churn", hue="Sex", kind="violin")
        g.fig.set_size_inches(10, 8)
        g.fig.suptitle('Churn Distribution by few Customer Properties',
                       y=1.01, fontsize=14, fontweight='bold')
        g.set_axis_labels(
            "Marital Status",
            "Age",
            fontsize=13,
            fontweight='bold')
        g.set_titles("{col_var} {col_name}")
        g.set(ylim=(20, 80))

        plt.savefig(file_pth, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.clf()
        logging.info(
            "SUCCESS: Successful creation & storage of 'DistributionChurnersByFewProps.png'.")
    except AssertionError as err:
        logging.error(
            "ERROR: Plot file 'DistributionChurnersByFewProps.png' not created or stored.")
        raise err


def plot_total_transactions(
    df,
    file_pth=os.path.join(
        cfg.EDA_DIR,
        cfg.DIST_TOTALTRANS_FILENAME)):
    '''
    Plots and stores the distribution of all transactions.

    input:
        df: pandas dataframe
        file_pth: (str) path and file name string of diagram, default path is "./images/eda/"
    output:
        None
    '''
    try:
        assert df is not None, "File could not be mapped to the needed dataframe; df is None."
        assert df.empty is False, "Dataframe df includes no items, it is empty."
        assert {'Total_Trans_Ct'}.issubset(df.columns) is True, \
            "Needed feature 'Total_Trans_Ct' is not available in df."

        g = sns.displot(data=df, x="Total_Trans_Ct", kde=True, rug=True)
        g.fig.set_size_inches(18, 8)
        plt.title(
            'Distribution of Total Transactions',
            fontsize=14,
            fontweight='bold')
        plt.xlabel('Total Transactions Ct', fontsize=13, fontweight='bold')
        plt.ylabel('Count', fontsize=13, fontweight='bold')

        plt.savefig(file_pth, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.clf()
        logging.info(
            "SUCCESS: Successful creation & storage of 'DistributionTotalTransactions.png'.")
    except AssertionError as err:
        logging.error(
            "ERROR: Plot file 'DistributionTotalTransactions.png' not created or stored.")
        raise err


def plot_scatter_by_churn(
    df,
    file_pth=os.path.join(
        cfg.EDA_DIR,
        cfg.SCATTERPLOT_FILENAME)):
    '''
    Plots the scatterplot of the dataframe features filtered by churn.

    input:
        df: pandas dataframe
        file_pth: (str) path and file name string of diagram, default path is "./images/eda/"
    output:
        None
    '''
    try:
        assert df is not None, "File could not be mapped to the needed dataframe; df is None."
        assert df.empty is False, "Dataframe df includes no items, it is empty."
        assert {'Churn'}.issubset(
            df.columns) is True, "Needed feature 'Churn' not available in df."

        graph = sns.pairplot(data=df, hue="Churn")
        graph.fig.suptitle(
            'Feature Scatterplot: Distribution by Churn',
            y=1.00,
            fontsize=14,
            fontweight='bold')

        plt.savefig(file_pth, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.clf()
        logging.info(
            "SUCCESS: Successful creation & storage of 'ScatterplotByChurn.png'.")
    except AssertionError as err:
        logging.error(
            "ERROR: Plot file 'ScatterplotByChurn.png' has not been created or stored.")
        raise err


def plot_feat_correlation(
    df,
    file_pth=os.path.join(
        cfg.EDA_DIR,
        cfg.SPEARMAN_CORR_FILENAME)):
    '''
    Plots and stores the spearman feature correlation diagram.

    input:
        df: pandas dataframe
        file_pth: (str) path and file name string of diagram, default path is "./images/eda/"
    output:
        None
    '''
    try:
        assert df is not None, "File could not be mapped to the needed dataframe; df is None."
        assert df.empty is False, "Dataframe df includes no items, it is empty."

        plt.figure(figsize=(16, 8))
        sns.heatmap(df.corr(method='spearman'), annot=False, linewidths=2,
                    cmap=sns.color_palette("YlOrBr", as_cmap=True))
        plt.title('Feature Correlations', fontsize=14, fontweight='bold')

        plt.savefig(file_pth, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.clf()
        logging.info(
            "SUCCESS: Successful creation & storage of 'Spearman_FeatCorrelation.png'.")
    except AssertionError as err:
        logging.error(
            "ERROR: Plot file 'Spearman_FeatCorrelation.png' not created or stored.")
        raise err


def plot_boxplot_data(
    df,
    file_pth=os.path.join(
        cfg.EDA_DIR,
        cfg.BOXPLOT_FILENAME)):
    '''
    Plots and stores the boxplot diagram.

    input:
        df: pandas dataframe for boxplot creation of the following features:
                feat_labels = ['Customer_Age', 'Dependent_count',
                    'Total_Relationship_Count','Months_Inactive_12_mon',
                    'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
                    'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Churn']
        file_pth: (str) path and file name of diagram, default path is "./images/eda/"
    output:
        None
    '''
    try:
        assert df is not None, "File could not be mapped to the needed dataframe; df is None."
        assert df.empty is False, "Dataframe df includes no items, it is empty."

        plt.figure(figsize=(16, 8))
        g = sns.boxplot(data=df, palette='mako_r')
        g.set(ylim=(0, 35500))
        feat_labels = [
            'Customer_Age',
            'Dependent_count',
            'Total_Relationship_Count',
            'Months_Inactive_12_mon',
            'Contacts_Count_12_mon',
            'Credit_Limit',
            'Total_Revolving_Bal',
            'Total_Amt_Chng_Q4_Q1',
            'Total_Trans_Ct',
            'Total_Ct_Chng_Q4_Q1',
            'Churn']
        g.set_xticklabels(feat_labels, fontsize=12, rotation=75)
        plt.title(
            'Boxplots - Data Tendency of Features',
            fontsize=14,
            fontweight='bold')
        plt.xlabel('Features', fontsize=13, fontweight='bold')
        plt.ylabel('Count', fontsize=13, fontweight='bold')

        plt.savefig(file_pth, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.clf()
        logging.info(
            "SUCCESS: Successful creation & storage of 'Boxplot_DataTendency.png'.")
    except AssertionError as err:
        logging.error(
            "ERROR: Plot file 'Boxplot_DataTendency.png' not created or stored.")
        raise err


def perform_eda(df, out_pth=cfg.EDA_DIR):
    '''
    Performs some specific EDA analysis on fiven df and saves figures to images/eda folder.
    This method is used as a wrapper to call specific plot methods where the testing happen.

    input:
        df: pandas dataframe for that exploratory data analysis shall happen
        out_pth: (str) path string of created outputs
    output:
        None
    '''
    try:
        logging.info("Start creating the EDA report and .png image plots.")
        # param checks happen in the specific functions
        create_eda_profile_report(
            df=df, file_pth=os.path.join(
                out_pth, cfg.EDA_REPORT_ORIG))
        plot_churners_by_sex(
            df=df, file_pth=os.path.join(
                out_pth, cfg.AMOUNT_SEX_FILENAME))
        plot_customers_age(
            df=df, file_pth=os.path.join(
                out_pth, cfg.DIST_AGE_FILENAME))
        plot_marital_status(
            df=df, file_pth=os.path.join(
                out_pth, cfg.DIST_MARITALSTAT_FILENAME))
        plot_churners_by_fewprops(
            df=df, file_pth=os.path.join(
                out_pth, cfg.DIST_CHURNERS_FEWPROPS_FILENAME))
        plot_total_transactions(
            df=df, file_pth=os.path.join(
                out_pth, cfg.DIST_TOTALTRANS_FILENAME))
        plot_scatter_by_churn(
            df=df, file_pth=os.path.join(
                out_pth, cfg.SCATTERPLOT_FILENAME))
        plot_feat_correlation(
            df=df, file_pth=os.path.join(
                out_pth, cfg.SPEARMAN_CORR_FILENAME))

        # future toDo:
        # drop of features before boxplot is identified visually via feat. corr. diagram,
        # implementation of automatic rule needed
        df.drop(
            columns=[
                'Avg_Open_To_Buy',
                'Avg_Utilization_Ratio',
                'Total_Trans_Amt',
                'Months_on_book'],
            inplace=True)

        plot_boxplot_data(
            df=df, file_pth=os.path.join(
                out_pth, cfg.BOXPLOT_FILENAME))
        plt.clf()
        logging.info("SUCCESS: EDA part happens successfully.")
    except Exception as err:
        logging.error("ERROR: EDA part fails.")
        raise err


#
# Feature Engineering
#
def encoder_helper(df, category_lst, response=""):
    '''
    Helper function to turn each categorical column into a new column with
    proportion of churn for each category - associated with cell 15 from the original notebook.

    input:
        df: pandas dataframe
        category_lst: list of columns that contain categorical features
        response: string of response name, by default an empty string
            [optional argument that could be used for naming variables or index y column]
    output:
        df: pandas dataframe with new columns for encoded categorical features,
            original ones are deleted and
            if no match between categorical feature list and df.columns parameter exists,
            the original df is returned
    '''

    try:
        assert len(
            category_lst) > 0, "encoder_helper(): categorical list param is empty."
        logging.info(
            "SUCCESS: Check of amount of categorical items: '%s'.",
            len(category_lst))
        assert all(isinstance(val, str) for val in category_lst) is True, \
            "Not all items are categorical strings."
        assert all(val != "" for val in category_lst) is True, \
            "Some categorical items are empty strings."
        logging.info("SUCCESS: All list elements are categorical items.")
        assert df.empty is False, "Dataframe df is empty, no items included."
        logging.info("SUCCESS: dataframe is not empty, shape: %s.", df.shape)
    except AssertionError as err:
        logging.error(
            "ERROR: df and/or categorical_lst are empty or include wrong types.")
        raise err

    try:
        logging.info("Start to perform categorical feature encoding ...")
        for item in category_lst:
            if item in df.columns:
                groups = df.groupby(item).mean()['Churn']
                lst = []
                for val in df[item]:
                    lst.append(groups.loc[val])
                col_name = item + '_' + response
                df[col_name] = lst
                # after encoding remove the item column
                df.drop(columns=[item], inplace=True)
                logging.info(
                    "SUCCESS: Proportion of churn encoding for item: %s", item)
            else:
                logging.error(
                    "ERROR: %s does not exist in dataframe df.", item)
        logging.info(
            "SUCCESS: Whole encoding of categorical features happens successfully.")
    except Exception as err:
        logging.error("ERROR: Categorical proportion of churn encoding fails.")
        raise err

    return df


def perform_feature_engineering(df):
    '''
    Performs some feature engineering, necessary before doing the model handling.

    input:
          df: pandas dataframe for the feature engineering process
    output:
          X_train: X training data or None if not created
          X_test: X testing data or None if not created
          y_train: y training data or None if not created
          y_test: y testing data or None if not created
          train_label_0: amount of non-churn customers in training set; default is 0.5
          train_label_1: amount of churn customers in training set; default is 0.5
    '''
    # default settings of return params
    X_train = None
    X_test = None
    y_train = None
    y_test = None
    train_label_0 = 0.5
    train_label_1 = 0.5

    try:
        assert df is not None, "File could not be mapped to the needed dataframe; df is None."
        assert df.empty is False, "Dataframe df includes no items, it is empty."
        assert {'Churn'}.issubset(
            df.columns) is True, "Feature 'Churn' is not available in df."

        logging.info("Start to perform feature engineering ...")
        X = pd.DataFrame()   # X matrix
        y = df['Churn']      # y target column

        keep_cols = [col for col in df.columns if col != 'Churn']
        X[keep_cols] = df[keep_cols]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42)

        # We need the imbalanced information for the xgboost classifier
        # configuration ...
        counter = Counter(y_train)
        logging.info(
            "Class distribution about churn in the training set is: %s",
            counter)
        train_label_0 = counter.get(0)
        logging.info(
            "Target label class of the training includes '%s' identified non-churn customers.",
            train_label_0)
        train_label_1 = counter.get(1)
        logging.info(
            "Target label class of the training includes '%s' identified churn customers.",
            train_label_1)

        logging.info(
            "SUCCESS: perform df feature engineering for model handling happens successfully.")
    except AssertionError as err:
        logging.error(
            "ERROR: perform df feature engineering for model handling fails.")
        raise err

    return X_train, X_test, y_train, y_test, train_label_0, train_label_1


#
# Model Training & Prediction
#
def save_model_pipe(model, model_type="", model_filepath=""):
    '''
    Save model as a pickle file.

    input:
        model: the model pipe to be stored (includes scaler and model steps)
        model_type: (str) model type can be 'lr', 'rfc' or 'xgbc';
                'unknown_type' if nothing is delivered
        model_filepath: (str) storage path,
                if not available store the model in './models/' churn directory
    output:
        filename: (str) path and name of model pickle file
    '''
    filename = ""
    try:
        assert model is not None, "No model to pickle given."
        # pickeled models are of type
        # sklearn.linear_model._logistic.LogisticRegression,
        # sklearn.ensemble._forest.RandomForestClassifier,
        # xgboost.sklearn.XGBClassifier)
        logging.info("Save model: %s", model)
        if model_type == "":
            logging.info(
                "Unexpected model type, model pickeld with 'unknown_type' substring.")
            model_type = 'unknown_type'

        if model_filepath == "":
            model_filepath = cfg.MODELS_DIR

        filename = model_filepath + 'best_churn_clf_' + model_type + '.pkl'
        joblib.dump(model, filename)
        logging.info(
            "SUCCESS: storage of model '%s' as pickle file happens successfully.",
            model_type)
    except AssertionError as err:
        logging.error("ERROR: Storage of model as pickle file fails.")
        raise err

    return filename


def read_model(model_filepath):
    '''
    Read model pipeline from given pickle file path via joblib.load().

    input:
        model_filepath: (str) storage path of the model,
        if not available try to read from './models/' churn directory
    output:
        clf: classifier model, None if nothing has been read or function fails
    '''
    clf = None
    try:
        assert model_filepath is not None, "No clf model path for pickle read given."

        if model_filepath == "":
            model_filepath = cfg.MODELS_DIR

        # returns whole pipeline, 2. elem is model estimator
        clf = joblib.load(model_filepath)
        assert clf is not None, "Model pipe pickle file has not been read."
        logging.info(
            "SUCCESS: reading of classifier model pipe via pickle happens successfully.")
        logging.info(
            "SUCCESS: function returns the model estimator only 'clf.steps[1][1].")
    except AssertionError as err:
        logging.error("ERROR: Reading of classifier model via pickle fails.")
        raise err

    # we need the whole pipe, not only classifier; clf.steps[1][1] would be
    # classifier only
    return clf


def train_logregression_clf(X_train, X_test, y_train, y_test):
    '''
    Training works with pipeline transformer for scaler and classification estimators:
    first scaling  with RobustScaler, then classification via LogisticRegression.

    input:
        X_train: X training data
        X_test: X testing data
        y_train: y training data
        y_test: y testing data
    output:
        y_train_preds_logreg: calculated predictions of X_train input; None if failed
        y_test_preds_logreg: calculated predictions of X_test input; None if failed
    '''
    try:
        assert X_train is not None, "Train models function param 'X_train' does not exist."
        assert X_test is not None, "Train models function param 'X_test' does not exist."
        assert y_train is not None, "Train models function param 'y_train' does not exist."
        assert y_test is not None, "Train models function param 'y_test' does not exist."
        assert isinstance(X_train, pd.core.frame.DataFrame) is True, \
            "'X_train' param is no dataframe type."
        assert isinstance(X_test, pd.core.frame.DataFrame) is True, \
            "'X_test' param is no dataframe type."
        assert isinstance(y_train, pd.core.series.Series) is True, \
            "'y_train' param is no series type."
        assert isinstance(y_test, pd.core.series.Series) is True, \
            "'y_test' param is no series type."

        logging.info('Start to train LogisticRegression Classifier ...')
        # return predictions
        y_train_preds_logreg = None
        y_test_preds_logreg = None
        test_report = None
        train_report = None

        # pipe
        rscaler = RobustScaler(quantile_range=(25, 75))
        lrc = LogisticRegression()
        logreg_cv = None
        estimators_lg = []
        estimators_lg.append(('rscaler', rscaler))
        # note: LogisticRegression is not a transformer
        estimators_lg.append(('logreg', lrc))
        pipe = Pipeline(estimators_lg, verbose=False)

        param_grid = {
            'logreg__class_weight': [None, 'balanced'],
            'logreg__C': [0.01, 0.1, 1],
            'logreg__solver': ['lbfgs', 'liblinear'], }

        # fitting the data in the pipe, evaluation with CV, log best settings & scores
        # regarding verbose int param see info of:
        # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
        # >1 : computation time for each fold and parameter candidate is displayed;
        # >2 : score is also displayed;
        # >3 : fold and candidate parameter indexes are also displayed together with starting time
        seed = 17
        skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        logreg_cv = GridSearchCV(estimator=pipe, param_grid=param_grid,
                                 cv=skfold, verbose=0)
        # print(logreg_cv.estimator.get_params().keys())
        assert logreg_cv is not None, "CV classifier result is still None."
        logging.info(
            "--- X_train for log regression GridSearchCV fit and predict: %s",
            X_train)
        logging.info("--- X_test for CV best testimator predict: %s", X_test)
        logreg_cv.fit(X_train, y_train)

        y_train_preds_logreg = logreg_cv.best_estimator_.predict(X_train)
        y_test_preds_logreg = logreg_cv.best_estimator_.predict(X_test)

        logreg_pkl_file = save_model_pipe(
            logreg_cv.best_estimator_,
            model_type="lr",
            model_filepath=cfg.MODELS_DIR)

        logging.info(
            '---  LogisticRegression Classifier CV best estimator ---')
        logging.info(logreg_cv.best_estimator_[-1])

        logging.info('------------------------------------------------------')
        logging.info(
            '---  LogisticRegression CV report results (single model)  ---')
        logging.info('test results')
        test_report = classification_report(y_test, y_test_preds_logreg)
        logging.info(test_report)
        logging.info('\ntrain results')
        train_report = classification_report(y_train, y_train_preds_logreg)
        logging.info(train_report)
        logging.info('------------------------------------------------------')

        logging.info(
            "SUCCESS: Training of logistic regression classifier model happens successfully.")
    except AssertionError as err:
        logging.error(
            "ERROR: Training of logistic regression classifier model fails.")
        raise err

    return y_train_preds_logreg, y_test_preds_logreg, logreg_pkl_file, test_report, train_report


def train_randomforest_clf(X_train, X_test, y_train, y_test):
    '''
    Training works with pipeline transformer for scaler and classification estimators:
    first scaling  with RobustScaler, then classification via RandomForestClassifier.

    input:
        X_train: X training data
        X_test: X testing data
        y_train: y training data
        y_test: y testing data
    output:
        y_train_preds_rf: calculated predictions of X_train input; None if failed
        y_test_preds_rf: calculated predictions of X_test input; None if failed
    '''
    try:
        assert X_train is not None, "Train models function param 'X_train' does not exist."
        assert X_test is not None, "Train models function param 'X_test' does not exist."
        assert y_train is not None, "Train models function param 'y_train' does not exist."
        assert y_test is not None, "Train models function param 'y_test' does not exist."
        assert isinstance(X_train, pd.core.frame.DataFrame) is True, \
            "'X_train' param is no dataframe type."
        assert isinstance(X_test, pd.core.frame.DataFrame) is True, \
            "'X_test' param is no dataframe type."
        assert isinstance(y_train, pd.core.series.Series) is True, \
            "'y_train' param is no series type."
        assert isinstance(y_test, pd.core.series.Series) is True, \
            "'y_test' param is no series type."

        logging.info('Start to train RandomForestClassifier ...')
        # return predictions
        y_train_preds_rf = None
        y_test_preds_rf = None
        test_report = None
        train_report = None

        # pipe
        rscaler = RobustScaler(quantile_range=(25, 75))
        rfc = RandomForestClassifier(random_state=42)
        estimators_rf = []
        estimators_rf.append(('rscaler', rscaler))
        # RandomForestClassifier is not a transformer
        estimators_rf.append(('rforest', rfc))
        pipe = Pipeline(estimators_rf, verbose=False)

        param_grid = {
            'rforest__n_estimators': [200, 500, 800],
            'rforest__max_features': ['auto', 'sqrt'],
            'rforest__max_depth': [4, 6, 8, 10],
            'rforest__criterion': ['gini', 'entropy']}
        scoring_metrics = {
            'AUC': 'roc_auc',
            'Accuracy': make_scorer(accuracy_score),
            'F1': 'f1'}

        # fitting the data in the pipe, evaluation with CV, log best settings &
        # scores
        seed = 17
        skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        rfc_cv = GridSearchCV(estimator=pipe, param_grid=param_grid,
                              scoring=scoring_metrics,
                              refit='AUC', return_train_score=True,
                              cv=skfold, verbose=0)
        rfc_cv.fit(X_train, y_train)

        y_train_preds_rf = rfc_cv.best_estimator_.predict(X_train)
        y_test_preds_rf = rfc_cv.best_estimator_.predict(X_test)

        rfc_pkl_file = save_model_pipe(
            rfc_cv.best_estimator_,
            model_type="rfc",
            model_filepath=cfg.MODELS_DIR)

        logging.info('---  RandomForestClassifier CV best estimator ---')
        logging.info(rfc_cv.best_estimator_[-1])

        logging.info('------------------------------------------------------')
        logging.info(
            '---  RandomForestClassifier CV report results (ensemble model)  ---')
        logging.info('test results')
        test_report = classification_report(y_test, y_test_preds_rf)
        logging.info(test_report)
        logging.info('\ntrain results')
        train_report = classification_report(y_train, y_train_preds_rf)
        logging.info(train_report)
        logging.info('------------------------------------------------------')

        logging.info(
            "SUCCESS: Training of RandomForestClassifier model happens successfully.")
    except AssertionError as err:
        logging.error(
            "ERROR: Training of radom forest classifier model fails.")
        raise err

    return y_train_preds_rf, y_test_preds_rf, rfc_pkl_file, test_report, train_report


def train_xgboost_clf(
        X_train,
        X_test,
        y_train,
        y_test,
        train_label_0,
        train_label_1):
    '''
    Training works with pipeline transformer for scaler and classification estimators:
    first scaling  with RobustScaler, then classification via XGBClassifier.

    input:
        X_train: X training data
        X_test: X testing data
        y_train: y training data
        y_test: y testing data
    output:
        y_train_preds_xgb: calculated predictions of X_train input; None if failed
        y_test_preds_xgb: calculated predictions of X_test input; None if failed
    '''
    try:
        assert X_train is not None, "Train models function param 'X_train' does not exist."
        assert X_test is not None, "Train models function param 'X_test' does not exist."
        assert y_train is not None, "Train models function param 'y_train' does not exist."
        assert y_test is not None, "Train models function param 'y_test' does not exist."
        assert train_label_0 is not None, \
            "Train models function param train_label_0' does not exist."
        assert train_label_1 is not None, \
            "Train models function param 'train_label_1' does not exist."
        assert isinstance(X_train, pd.core.frame.DataFrame) is True, \
            "'X_train' param is no dataframe type."
        assert isinstance(X_test, pd.core.frame.DataFrame) is True, \
            "'X_test' param is no dataframe type."
        assert isinstance(y_train, pd.core.series.Series) is True, \
            "'y_train' param is no series type."
        assert isinstance(y_test, pd.core.series.Series) is True, \
            "'y_test' param is no series type."
        assert isinstance(
            train_label_0, int) is True, "'train_label_0' param is no int type."
        assert isinstance(
            train_label_1, int) is True, "'train_label_1' param is no int type."

        logging.info('Start to train XGBClassifier ...')
        # return attributes
        y_train_preds_xgb = None
        y_test_preds_xgb = None
        test_report = None
        train_report = None

        # pipe
        rscaler = RobustScaler(quantile_range=(25, 75))
        # XGBClassifier:
        # train_label_0: to handle the target imbalance, this is the majority group
        # train_label_1: to handle the target imbalance, this is the minority
        # group
        xgb_cls = XGBClassifier(
            base_score=0.5,  # for binary classification
            objective='binary:logistic',   # eval metric for binary classification
            scale_pos_weight=train_label_0 / train_label_1,   # to handle the imbalance
            use_label_encoder=False,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss',  # for classification
            random_state=24,
            verbosity=0
        )
        estimators_xgb = []
        estimators_xgb.append(('rscaler', rscaler))
        estimators_xgb.append(('xgbc', xgb_cls))
        pipe = Pipeline(estimators_xgb, verbose=False)

        xgb_param_grid = {
            'xgbc__n_estimators': [600, 800, 1000],
            'xgbc__learning_rate': [0.1, 0.05, 0.01],
            'xgbc__max_depth': [5, 6, 7, 8],
            'xgbc__min_child_weight': [2, 3, 4, 6],
            'xgbc__reg_lambda': [1.1, 1.3], }
        scoring_metrics = {
            'AUC': 'roc_auc',
            'Accuracy': make_scorer(accuracy_score),
            'F1': 'f1'}

        # Construct grid searches, pipeline evaluation with CV, log best
        # settings & scores
        seed = 17
        skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        xgb_grid = GridSearchCV(estimator=pipe, param_grid=xgb_param_grid,
                                scoring=scoring_metrics,
                                refit='AUC', return_train_score=True,
                                cv=skfold, verbose=0)
        xgb_cv = xgb_grid.fit(X_train, y_train)

        y_train_preds_xgb = xgb_cv.best_estimator_.predict(X_train)
        y_test_preds_xgb = xgb_cv.best_estimator_.predict(X_test)

        xgbc_pkl_file = save_model_pipe(
            xgb_cv.best_estimator_,
            model_type="xgbc",
            model_filepath=cfg.MODELS_DIR)

        logging.info('---  XGBClassifier CV best estimator ---')
        logging.info(xgb_cv.best_estimator_[-1])

        logging.info('------------------------------------------------------')
        logging.info(
            '---  XGBClassifier CV report results (ensemble model)  ---')
        logging.info('test results')
        test_report = classification_report(y_test, y_test_preds_xgb)
        logging.info(test_report)
        logging.info('\ntrain results')
        train_report = classification_report(y_train, y_train_preds_xgb)
        logging.info(train_report)
        logging.info('------------------------------------------------------')

        logging.info(
            "SUCCESS: Training of xgb classifier model happens successfully.")
    except AssertionError as err:
        logging.error("ERROR: Training of xgb classifier model fails.")
        raise err

    return y_train_preds_xgb, y_test_preds_xgb, xgbc_pkl_file, test_report, train_report


def train_models(
        X_train,
        X_test,
        y_train,
        y_test,
        train_label_0,
        train_label_1):
    '''
    Train model types and store models best results and scoring.
    Included classifier model types are:
    - LogisticRegression
    - RandomForest
    - XGBClassifier

    input:
        X_train: X training data
        X_test: X testing data
        y_train: y training data
        y_test: y testing data
        train_label_0: (int) amount of customers labelled as non-churner
        train_label_1: (int) amount of customers labelled as churner
    output:
        dict_pred_results: prediction results values for each model type key,
        empty dictionary if failure appeared
    '''
    try:
        # param checks happen in the specific functions
        logging.info("Start with model trainings ...")
        dict_pred_results = {}
        # log regression
        y_train_preds_lr, y_test_preds_lr, logreg_pkl_file, test_report_lr, train_report_lr = \
            train_logregression_clf(X_train, X_test, y_train, y_test)
        # random forest
        y_train_preds_rf, y_test_preds_rf, rfc_pkl_file, test_report_rfc, train_report_rfc = \
            train_randomforest_clf(X_train, X_test, y_train, y_test)
        # XGBoost
        y_train_preds_xgb, y_test_preds_xgb, xgbc_pkl_file, test_report_xgbc, train_report_xgbc = \
            train_xgboost_clf(X_train, X_test, y_train, y_test, train_label_0, train_label_1)

        # before creation of dictionary check if the values are appropriate:
        assert y_train_preds_lr is not None, \
            "'y_train_preds_lr' param is none, no classification report possible."
        assert y_train_preds_rf is not None, \
            "'y_train_preds_rf' param is none, no classification report possible."
        assert y_train_preds_xgb is not None, \
            "'y_train_preds_xgb' param is none, no classification report possible."
        assert y_test_preds_lr is not None, \
            "'y_test_preds_lr' param is none, no classification report possible."
        assert y_test_preds_rf is not None, \
            "'y_test_preds_rf' param is none, no classification report possible."
        assert y_test_preds_xgb is not None, \
            "'y_test_preds_xgb' param is none, no classification report possible."
        assert isinstance(y_train_preds_lr, np.ndarray) is True, \
            "'y_train_preds_lr' param is not a ndarray."
        assert isinstance(y_train_preds_rf, np.ndarray) is True,  \
            "'y_train_preds_rf' param is not a ndarray."
        assert isinstance(y_train_preds_xgb, np.ndarray) is True, \
            "'y_train_preds_xgb' param is not a ndarray."
        assert isinstance(y_test_preds_lr, np.ndarray) is True, \
            "'y_test_preds_lr' param is not a ndarray."
        assert isinstance(y_test_preds_rf, np.ndarray) is True, \
            "'y_test_preds_rf' param is not a ndarray."
        assert isinstance(y_test_preds_xgb, np.ndarray) is True, \
            "'y_test_preds_xgb' param is not a ndarray."

        # note: it is expected that the model prediction delivers correct 0 and 1 values,
        # but check if all items of a list are numbers
        assert all(isinstance(item, np.int64) for item in y_train_preds_lr) is True, \
            "Not all elements of 'y_train_preds_lr' are int numbers."
        assert all(isinstance(item, np.int64) for item in y_train_preds_rf) is True, \
            "Not all elements of 'y_train_preds_rf' are int numbers."
        assert all(isinstance(item, np.int32) for item in y_train_preds_xgb) is True, \
            "Not all elements of 'y_train_preds_xgb' are int numbers."
        assert all(isinstance(item, np.int64) for item in y_test_preds_lr) is True, \
            "Not all elements of 'y_test_preds_lr' are int numbers."
        assert all(isinstance(item, np.int64) for item in y_test_preds_rf) is True, \
            "Not all elements of 'y_test_preds_rf' are int numbers."
        assert all(isinstance(item, np.int32) for item in y_test_preds_xgb) is True, \
            "Not all elements of 'y_test_preds_xgb' are int numbers."

        dict_pred_results = {
            'LogisticRegression': [
                y_train_preds_lr,
                y_test_preds_lr,
                logreg_pkl_file,
                test_report_lr,
                train_report_lr],
            'RandomForestClassifier': [
                y_train_preds_rf,
                y_test_preds_rf,
                rfc_pkl_file,
                test_report_rfc,
                train_report_rfc],
            'XGBClassifier': [
                y_train_preds_xgb,
                y_test_preds_xgb,
                xgbc_pkl_file,
                test_report_xgbc,
                train_report_xgbc],
        }

        assert dict_pred_results is not None, "'dict_pred_results' is none,\
            no classifier predictions exist."
        assert len(dict_pred_results) == 3, "Prediction dict does not include 3 classifiers."
        assert dict_pred_results.get('LogisticRegression') == [y_train_preds_lr,
                                                               y_test_preds_lr, logreg_pkl_file,
                                                               test_report_lr, train_report_lr], \
            "No appropriate prediction results of LogisticRegression available."
        assert dict_pred_results.get('RandomForestClassifier') == [y_train_preds_rf,
                                                                   y_test_preds_rf, rfc_pkl_file,
                                                               test_report_rfc, train_report_rfc],\
            "No appropriate prediction results of RandomForestClassifier available."
        assert dict_pred_results.get('XGBClassifier') == [y_train_preds_xgb,
                                                          y_test_preds_xgb, xgbc_pkl_file,
                                                          test_report_xgbc, train_report_xgbc], \
            "No appropriate prediction results of XGBClassifier available."

        logging.info(
            "SUCCESS: Successful train models & store best classifiers and scores.")
    except AssertionError as err:
        logging.error(
            "ERROR: Train models & store best classifiers and their scores fails.")
        raise err

    return dict_pred_results


#
# Model Evaluation
#
def feature_importance_plot(model, X_data,
                            title='Feature Importance of Classifier',
                            filename="",
                            output_pth=""):
    '''
    Creates and stores the feature importances plot in pth.

    input:
        model: model object containing feature_importances_
        X_data: pandas dataframe of X values
        title: (str) titel of the diagram
        filename: (str) storage filename, if empty string set to 'Churn_feat_import_plot.png'
        output_pth: (str) dir path to store the figure,
                if not set default is RESULTS_DIR (i.e. './images/results/') of config file
    output:
         None
    '''
    try:
        assert model is not None, "'model' param does not exist for feature importance plot."
        assert X_data is not None, "'X_data' param does not exist for feature importance plot."
        assert isinstance(X_data, pd.core.frame.DataFrame) is True, \
            "'X_data' param is no dataframe. No feature importance plot possible."

        if filename == "":
            logging.info(
                'Filename for feature importance plot does not exist.\
                Create as default "Churn_feat_import_plot.png".')
            filename = "Churn_feat_import_plot.png"

        if output_pth == "":
            logging.info(
                'Output path for feature importance plot does not exist.\
                Create as default ./images/results/ churn directory.')
            output_pth = cfg.RESULTS_DIR

        logging.info("Start to create feature importance plot ...")
        # Calculate feature importances
        importances = model.feature_importances_
        # Sort feat. importances in descending order
        indices = np.argsort(importances)[::-1]
        # Rearrange feature names so they match the sorted feature importances
        names = [X_data.columns[i] for i in indices]

        # Create plot
        plt.figure(figsize=(20, 5))
        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylabel('Importance', fontsize=12, fontweight='bold')
        plt.xlabel('Features', fontsize=12, fontweight='bold')
        plt.bar(range(X_data.shape[1]), importances[indices])
        plt.xticks(range(X_data.shape[1]), names, rotation=90)

        plt.savefig(
            os.path.join(
                output_pth,
                filename),
            dpi=300,
            bbox_inches='tight',
            pad_inches=0)
        plt.clf()
        logging.info(
            "SUCCESS: Successful creation & storage of feature importance '%s'.", filename)
    except AssertionError as err:
        logging.error(
            "ERROR: Creation and storage of feature importance '%s' file fails.",
            filename)
        raise err


def store_classification_report(train_report, test_report,
                                filename="ClassificationReport.png",
                                model_type="Model"):
    '''
    Stores given classification reports as .png files in RESULTS_DIR (default: './images/results').

    input:
        train_report: txt file of the models classification train report
        test_report: txt file of the modles classification test report
        filename: (str) storage label of the .png file; default is 'ClassificationReport.png'
        model_type: (str) model label used to improve the classification information;
                    default is 'Model'
    output:
        None
    '''
    try:
        assert train_report is not None, 'No model classification train report exists.'
        assert test_report is not None, 'No mode classification test report exists.'

        logging.info('Start to store report for %s...', model_type)
        plt.rc('figure', figsize=(5, 5))
        plt.text(0.01,
                 1.25,
                 str('Evaluation of ' + model_type + ' Classification'),
                 {'fontsize': 12,
                  'fontweight': 'bold'},
                 fontproperties='monospace')
        plt.text(0.01, 1.15, str('' + model_type + ' Train'),
                 {'fontsize': 10, 'fontweight': 'bold'},
                 fontproperties='monospace')
        plt.text(
            0.01, 0.05, str(train_report), {
                'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01,
                 0.5,
                 str('' + model_type + ' Test'),
                 {'fontsize': 10,
                  'fontweight': 'bold'},
                 fontproperties='monospace')
        plt.text(
            0.01, 0.7, str(test_report), {
                'fontsize': 10}, fontproperties='monospace')
        plt.axis('off')

        plt.savefig(os.path.join(cfg.RESULTS_DIR, filename), dpi=300,
                    bbox_inches='tight', pad_inches=0)
        # clear figure instance or plot info is part of next one and mixed it
        # up in .png file
        plt.clf()
        logging.info(
            "SUCCESS: Classification reports storage as .png file happens successfully.")
    except AssertionError as err:
        logging.error(
            "ERROR: Classification reports storage as .png file fails.")
        raise err


def plot_classification_report_image(dict_class_reports):
    '''
    Creates classification report images for training and testing results and stores them
    in './images/result' folder.

    For the given models classification reports the following parameters are used:
        y_train: training response values
        y_test:  test response values
        y_train_preds_lr: training predictions from logistic regression
        y_train_preds_rf: training predictions from random forest
        y_train_preds_xgb: training predictions from XGBClassifier
        y_test_preds_lr: test predictions from logistic regression
        y_test_preds_rf: test predictions from random forest
        y_test_preds_xgb: test predictions from XGBClassifier

    input:
         dict_class_reports: dictionary of test and train reports of the given models
    output:
         None
    '''
    try:
        assert dict_class_reports is not None, "No reports dictionary exists."
        assert dict_class_reports.keys() is not None, \
            "Report dictionary includes no keys."
        assert dict_class_reports.get('lg', {}).get('lg_test') is not None,\
            "No lg test report exists."
        assert dict_class_reports.get('lg', {}).get('lg_train') is not None,\
            "No lg train report exists."
        assert dict_class_reports.get('rfc', {}).get('rfc_test') is not None,\
            "No rfc test report exists."
        assert dict_class_reports.get('rfc', {}).get('rfc_train') is not None,\
            "No rfc train report exists."
        assert dict_class_reports.get('xgbc', {}).get(
            'xgbc_test') is not None, "No xgbc test report exists."
        assert dict_class_reports.get('xgbc', {}).get(
            'xgbc_train') is not None, "No xgbc train report exists."

        logging.info("Start to create classification report images...")
        store_classification_report(
            train_report=dict_class_reports.get(
                'xgbc',
                {}).get('xgbc_train'),
            test_report=dict_class_reports.get(
                'xgbc',
                {}).get('xgbc_test'),
            filename="Best_XGBC_TrainTestClassReport.png",
            model_type="XGBClassifier")

        store_classification_report(
            train_report=dict_class_reports.get(
                'rfc',
                {}).get('rfc_train'),
            test_report=dict_class_reports.get(
                'rfc',
                {}).get('rfc_test'),
            filename="Best-RFC_TrainTestClassReport.png",
            model_type="RandomForestClassifier")

        store_classification_report(
            train_report=dict_class_reports.get(
                'lg',
                {}).get('lg_train'),
            test_report=dict_class_reports.get(
                'lg',
                {}).get('lg_test'),
            filename="Best-LogReg_TrainTestClassReport.png",
            model_type="Logistic Regression")
        logging.info(
            'SUCCESS: All best estimators classification report png files are stored.')
    except AssertionError as err:
        logging.error(
            "ERROR: Not all best estimators classification report png files are stored.")
        raise err


def plot_roc_auc(y_test, X_test,
                 estimators,
                 title='Classifiers ROC AUC Curves',
                 filename='BestEstimators_rocauc_clf.png',):
    '''
    Plots the roc auc diagram of the given classifier prediction results and stores the diagram
    in the './images/results/' directory.

    input:
        title: (str) title for the roc auc diagram, by default 'Classifiers ROC AUC'
        filename: (str) name of .png diagram, by defaullt 'BestEstimators_rocauc_clf.png'
        estimators: ([]) the list of the 3 best found estimators of
                LogisticRegression, RandomForestClassifier and XGBoostclassifier
        y_test: the target values the fitted model shall be evaluated for
        X_test: the X_test input values for predictions made by the fitted model
    output:
        None
    '''
    try:
        assert estimators is not None, \
            "'estimators param does not exist. No roc auc diagram possible."
        assert y_test is not None, "'y_test' param does not exist. No roc auc diagram possible."
        assert X_test is not None, "'X_test' param does not exist. No roc auc diagram possible."
        assert isinstance(
            estimators, list) is True, "'estimators' is not a list."
        assert y_test is not None, "'y_test' param is none, no roc auc diagram possible."
        assert isinstance(y_test, pd.core.series.Series) is True, \
            "'y_test' param is no series type."
        assert isinstance(X_test, pd.core.frame.DataFrame) is True, \
            "'X_test' param no dataframe type."
        assert all(isinstance(item.steps[1][1],
                              (LogisticRegression, RandomForestClassifier, XGBClassifier))
                   for item in estimators) is True, \
            "Given classifiers list does not include valid instances for roc auc plot."

        logging.info("Start to plot roc auc diagram, one for all models ...")
        plt.figure(figsize=(15, 8))
        ax = plt.gca()  # all clf instances are included in plot via ax param

        for clf in estimators:
            if isinstance(clf.steps[1][1], LogisticRegression):
                logging.info(
                    "Creation of ROC AUC curve for LogisticRegression...")
                RocCurveDisplay.from_estimator(
                    clf, X_test, y_test, name='LogisticRegression', ax=ax, alpha=0.8)
            if isinstance(clf.steps[1][1], RandomForestClassifier):
                logging.info(
                    "Creation of ROC AUC curve for RandomForestClassifier...")
                RocCurveDisplay.from_estimator(
                    clf, X_test, y_test, name='RandomForestClassifier', ax=ax, alpha=0.8)
            if isinstance(clf.steps[1][1], XGBClassifier):
                logging.info("Creation of ROC AUC curve for XGBClassifier...")
                RocCurveDisplay.from_estimator(
                    clf, X_test, y_test, name='XGBClassifier', ax=ax, alpha=0.8)

        plt.title(title, fontsize=13, fontweight='bold')

        # Store the plot image in the images/results dir
        plt.savefig(os.path.join(cfg.RESULTS_DIR, filename), dpi=300,
                    bbox_inches='tight', pad_inches=0)
        logging.info(
            "SUCCESS: Creation & storage of roc auc .png diagram happens successfully.")
        plt.clf()
    except AssertionError as err:
        logging.error(
            "ERROR: Creation & storage of roc auc .png diagram fails.")
        raise err


def plot_mean_shapvalues_xtest(best_estimator, X_test, filename=""):
    '''
    Plots and stores the mean shap values for the X_test set as bar diagram.
    Delivers information about average impact on model output.

    input:
        best_estimator: best estimator identified during models evaluation
        filename: (str) name of the .png shap values bar diagram,
                default is 'ShapValues_meanImpactOnTest_bar.png'
    output:
        None
    '''
    try:
        assert best_estimator is not None, "No estimator param given, shap values not created."
        assert isinstance(best_estimator,
                          (LogisticRegression, RandomForestClassifier, XGBClassifier)) is True, \
            "'best_estimator' param for shap values is not a valid model instance."
        assert X_test is not None, "X_test ist not given, is None. No shap value calculated."
        if filename == "":
            filename = "ShapValues_meanImpactOnTest_bar.png"

        logging.info("Start to create shap mean values bar chart ...")
        explainer = shap.TreeExplainer(best_estimator)
        shap_values = explainer.shap_values(X_test)
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)

        plt.savefig(os.path.join(cfg.RESULTS_DIR, filename), dpi=300,
                    bbox_inches='tight', pad_inches=0)
        logging.info(
            "SUCCESS: Successful creation & storage of shape '%s' bar diagram happens.",
            filename)
        plt.clf()
    except AssertionError as err:
        logging.error(
            "ERROR: Creation and storage of shap values '%s' bar diagram fails.",
            filename)
        raise err


##############################
# Main Call
##############################

def main():
    '''
    Workflow of this churn classifier library started e.g. via command line interface.
    '''
    # Administrative
    logging.info('---   %s: Start churn prediction workflow   ---', date_str)
    print('---   ' + date_str + ': Start churn prediction workflow!   ---')
    print('Logging of main library versions ...')
    get_library_version_info()

    # data handling
    data_path = os.path.join(cfg.DATA_DIR, cfg.BANK_DATA)
    print(f'Loading data...\n    DATABASE: {data_path}')
    df = import_data(data_path)

    # first preprocessing
    print('First df preprocessing started...')
    df = prepare_df(df)

    # EDA
    logging.info('---  EDA started...  ---')
    print('EDA started...')
    perform_eda(df=df, out_pth=cfg.EDA_DIR)

    # Feature Engineering
    logging.info('---  Feature Engineering started...  ---')
    print('Feature Engineering started...')

    # for original, preprocessed dataset: cat_columns structure shall be:
    # cat_columns = [
    #    'Sex',
    #    'Education_Level',
    #    'Marital_Status',
    #    'Income_Category',
    #    'Card_Category'
    # ]

    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.select_dtypes.html
    # don't use mentioned 'category', list would be empty
    cat_columns = df.select_dtypes(include='object').columns
    logging.info("The following categorical columns are identified:")
    logging.info(cat_columns)
    df = encoder_helper(df=df, category_lst=cat_columns, response="Churn")
    X_train, X_test, y_train, y_test, train_label_0, train_label_1 = \
        perform_feature_engineering(df)

    # Model Training & Prediction
    logging.info('---  Model Training and Prediction started...  ---')
    print('Model Training and Prediction started...')
    dict_pred_results = train_models(
        X_train,
        X_test,
        y_train,
        y_test,
        train_label_0,
        train_label_1)
    logging.info('Prediction results of models are:')
    logging.info(dict_pred_results)

    # dictionary checks happen in creation function, structure shall be:
    # dict_pred_results = {
    #    'LogisticRegression' : [y_train_preds_logreg, y_test_preds_logreg, logreg_pkl_file,
    #                            test_report_lr, train_report_lr],
    #    'RandomForestClassifier' : [y_train_preds_rf, y_test_preds_rf, rfc_pkl_file,
    #                                test_report_rfc, train_report_rfc],
    #    'XGBClassifier' : [y_train_preds_xgb, y_test_preds_xgb, xgbc_pkl_file,
    #                       test_report_xgbc, train_report_xgbc],
    # }

    dict_class_reports = {
        'lg': {
            'lg_test': dict_pred_results.get('LogisticRegression')[3],
            'lg_train': dict_pred_results.get('LogisticRegression')[4],
        },
        'rfc': {
            'rfc_test': dict_pred_results.get('RandomForestClassifier')[3],
            'rfc_train': dict_pred_results.get('RandomForestClassifier')[4],
        },
        'xgbc': {
            'xgbc_test': dict_pred_results.get('XGBClassifier')[3],
            'xgbc_train': dict_pred_results.get('XGBClassifier')[4],
        },
    }

    # Model Evaluation
    logging.info('---  Model Evaluation started...  ---')
    print('Model Evaluation started...')

    # classification reports
    plot_classification_report_image(dict_class_reports)

    # ROC AUC
    # load the best models, the whole pipe is returned,
    # something like read_model('./models/best_churn_clf_lr.pkl') with
    # dedicated file string
    lg_estimator = read_model(dict_pred_results.get('LogisticRegression')[2])
    rfc_estimator = read_model(
        dict_pred_results.get('RandomForestClassifier')[2])
    xgbc_estimator = read_model(dict_pred_results.get('XGBClassifier')[2])

    estimators = [lg_estimator, rfc_estimator, xgbc_estimator]
    logging.info(
        "Estimators pipe list shall include lr, rfc, xgbc: %s",
        estimators)
    plot_roc_auc(y_test=y_test, X_test=X_test,
                 estimators=estimators,
                 title='Classifiers ROC AUC',
                 filename='BestEstimators_rocauc_clf.png',)

    # NOTES: feature importance, shap values of ensemble tree models
    # it is known that the single tree model LogisticRegression is the worst classifier
    # and it has no feature importance attribute or is supported by shape TreeExplainer
    # message 'ntree_limit is deprecated, use `iteration_range` or model slicing instead.'
    # is from XGBClassifier training (internal code issue)
    # <Figure size 576x540 with 0 Axes> comes from plt.clf(), after no.>20 plots ...

    for model in estimators:
        if isinstance(model.steps[1][1], LogisticRegression):
            pass

        if isinstance(model.steps[1][1], RandomForestClassifier):
            title = 'Feature Importance of RandomForestClassifier'
            filename_feat = 'Churn_rfc_feat_import.png'
            filename_shap = 'Churn_rfc_shap_meanImpactOnTest.png'

            feature_importance_plot(model=model.steps[1][1],
                                    X_data=X_train,  # column labels are needed
                                    title=title,
                                    filename=filename_feat,
                                    output_pth=cfg.RESULTS_DIR)
            plot_mean_shapvalues_xtest(best_estimator=model.steps[1][1],
                                       X_test=X_test, filename=filename_shap)

        if isinstance(model.steps[1][1], XGBClassifier):
            title = 'Feature Importance of XGBClassifier'
            filename_feat = 'Churn_xgbc_feat_import.png'
            filename_shap = 'Churn_xgbc_shap_meanImpactOnTest.png'

            feature_importance_plot(model=model.steps[1][1],
                                    X_data=X_train,  # column labels are needed
                                    title=title,
                                    filename=filename_feat,
                                    output_pth=cfg.RESULTS_DIR)
            plot_mean_shapvalues_xtest(best_estimator=model.steps[1][1],
                                       X_test=X_test, filename=filename_shap)


if __name__ == "__main__":
    main()
