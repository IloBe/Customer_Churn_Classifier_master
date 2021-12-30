[//]: # (Image References)

[image1]: ./assets/KaggleChurnDatasetFirstRows.png "Churn dataset rows"
[image2]: ./assets/AmountOfChurnersBySex.png "Imbalanced dataset"
[image3]: ./assets/DistributionChurnersByFewProps.png "Churner properties"
[image4]: ./assets/ScatterplotByChurn "Scatterplot feature relations"
[image5]: ./assets/BestEstimators_rocauc_clf.png "Best estimators ROC AUC curves"
[image6]: ./assets/Best_XGBC_TrainTestClassReport.png "XGBClassifier classification report"


# Predict Customer Churn

In this Udacity ML DevOps Engineer Nanodegree project, we implement learnings about <b>software engineering principles</b> to identify credit card customers that are most likely to churn. The completed classification project includes a Python package for a machine learning project that follows coding (PEP8) and engineering best practices for implementing software that is modular, documented, and tested. The package will also have the flexibility of being run interactively or from the command-line interface (CLI).

The dataset for this project was pulled from [Kaggle](https://www.kaggle.com/sakshigoyal7/credit-card-customers). There the following description is given:<br>
"A manager at the bank is disturbed with more and more customers leaving their credit card services. They would really appreciate if one could predict for them who is gonna get churned so they can proactively go to the customer to provide them better services and turn customers' decisions in the opposite direction.

I got this dataset from a website with the URL as https://leaps.analyttica.com/home. I have been using this for a while to get datasets and accordingly work on them to produce fruitful results. The site explains how to solve a particular business problem.

Now, this dataset consists of 10,000 customers mentioning their age, salary, marital_status, credit card limit, credit card category, etc. There are nearly 18 features.

We have only 16.07% of customers who have churned. Thus, it's a bit difficult to train our model to predict churning customers."

![Churn dataset rows][image1]


## Project Description

### General Information
Last information tells us, that we have to deal with an <i>imbalanced dataset</i>, because the minority class is of interest.

After some general preprocessings of the dataset, an exploratory data analysis (<i>EDA</i>) profile report is created. It is an interactive .html file stored in the `.\images\eda\` folder delivering an overview of the data conditions.

Some detailed properties of the dataset and the customers, being a churner or not, can be identified by some of the other analysis diagrams already, like the ones below. 

![Imbalanced dataset][image2]     ![Churner properties][image3]

As a consequence, we have to take care of this imbalanced insight selecting an appropriate model resp. evaluation metrics and modify the dataset before usage of the selected model. Feature relationships, visualised by the scatterplot diagram of churners shows already some skewness and correlation. 

![Scatterplot feature relations][image4]
 
So, e.g. scaling is necessary, before a classification model can be triggered. This coding is realised by an ML pipeline approach. In general, tree models can handle imbalanced datasets better compared to other model types. After training and prediction of the 3 selected classification tree models

- LogisticRegression
- RandomForestClassifier
- XGBClassifier

their best estimators are stored in the `model` directory as a pickle file.

Their evaluation results - classification reports, feature importance and shap values - are visualised as well and stored in the `.\images\results\` folder. 

![Best estimators ROC AUC curves][image5]

![XGBClassifier classification report][image6]


**Notes:**<br>
Project focus is not to find best hyperparameter settings for the selected ML classifiers. So, some cross validation is done and evaluation showed, XGBClassifier is the best estimator, but its model parameters can still be improved.

Using some development tools, the following installation is necessary by using Python 3 version:
From PYPI install the linter and auto-formatter (see files chapter):
```
pip3 install pylint
pip3 install autopep8
```
and the projects requirements.txt file creator tool (see environment settings chapter):
```
pip3 install pipreqs
```

### Files
Our task is to complete the following given files by adding the churn features coding implementation in the .py files and general project information in this readme file:

1. `churn_library.py`
2. `churn_script_logging_and_tests.py` 
3. `README.md` 

For ease of use, all constants are separated from coding and stored in its own additionally added file stored in the `config` folder:

4. `churn_config.py`

Furthermore, provided are this two files as starting point including some (outdated) workflow information and some software results that need heavy refactoring and add-ons. This refactoring, modifications and additional testings are included in the formerly mentioned .py files to fulfill the required software quality.
- `Guide.ipynb`
- `churn_notebook.ipynb`

Regarding Python coding style, to meet pep8 standards as much as possible (variables or file length don't fit in all cases), `pylint` is used as command line tool by help of `autopep8`, e.g. to remove all trailing spaces automatically. For all files a score >9/10 is reached. The commands are:

```
pylint churn_library.py
pylint churn_script_logging_and_tests.py
pylint churn_config.py
```
and
```
autopep8 --in-place --aggressive --aggressive churn_script_logging_and_tests.py
autopep8 --in-place --aggressive --aggressive churn_library.py
autopep8 --in-place --aggressive --aggressive churn_config.py
```


## Running Files
How do set up this project and its environment? How do you run your files? What should happen when you run your files? All this questions are explained now.

### Environment settings
If you want to use this project files, you have to import the Python libraries used for this implementation and its dependencies. They are mentioned in the `requirements.txt` file that is created by usage of the `pipreqs`command as described in the following [blog](https://blog.jcharistech.com/2020/11/02/pipreqs-tutorial-how-to-create-requirements-txt-file-in-python-with-ease/) post.

#### Set-up
0. Create your project directory for your new repository
1. Clone this GitHub project or use its zip file to store the structure and files in your new repository
2. **If you are running the project on your local machine (and not using AWS)** create and activate a new virtual environment, called `customer-churn-project`. First, move to your directory `path/to/churn-classifier-project`.
  - __Windows__
  ```
  conda create --name customer-churn-project python=3.9
  activate customer-churn-project
  pip install -r requirements/requirements.txt
  ```
  
3. **If you are running the project on your local machine (and not using AWS)**, create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `customer-churn-project` environment. 
```
python -m ipykernel install --user --name customer-churn-project --display-name "customer-churn-project"
```

4. Open jupyter notebook to get the project structure and files
```
jupyter notebook
```

5. **If you are running the project on your local machine (and not using AWS)**, before running code, change the kernel to match the customer-churn-project environment by using the drop-down menu (**Kernel > Change kernel > customer-churn-project**).


#### License
This project coding is released under the [MIT](https://github.com/IloBe/Customer_Churn_Classifier_master/blob/main/LICENSE) license.


### Scripts
The .py files include the main() function call for usage of the command line interface. In other words, an `if __name__ == "__main__"` block is included.

Running the files `python` or `ipython` command on the terminal, the general project workflow, its status information and any kind of errors or in the test file case the testing results or errors are logged in a file stored in the `logs` folder. Some general dataset and workflow information is printed on the terminal as well, knowing the status of the run.
If some unexpected exceptions will appear, it is mentioned in the log file and an error raises on the terminal, terminating the script. Such issues have to be handled by software complaints for this project. An associated issue record is appreciated.

#### Run the project file
```
ipython churn_library.py
```
or
```
python churn_library.py
```

#### Run the test file
```
ipython churn_script_logging_and_tests.py
```
or
```
python churn_script_logging_and_tests.py
```

