# Sparkify - Large scale machine learning to predict customer churn using pysaprk 

## Table of Contents

1. [Overview](#overview)
2. [Installations](#installations)
3. [Data](#data)
4. [Modelling](#Modelling)
5. [Results](#results)
6. [Deployment using Flask](#flask)

## <a name="overview"></a> Overview

This project involves predicting **Customer Churn for a hypothetical music streaming app Sparkify**, using Spark's MLlib to engineer features and build a classification model. The dataset used here is around 12 GB in size. <br>
This project is worked on [IBM Cloud's Watson Studio](https://www.ibm.com/se-en/cloud/watson-studio), uploading the data cluster, with a Python 3.8/Spark 3.0 enabled Jupyter Notebook. <br>

Using `pyspark`, the project broadly involves the following:

- Preprocessing cleaning the data.
- Exploratory Data Analysis - where the data is explored in its depth, and data visualizations to help with selecting features, and to basically understand the data more.
- Feature Engineering - appropriate features are selected based on the EDA, as well as creating new features from existing data, to create a final dataset ready for training using Spark's ML.
- Modelling - four different classification algorithms are tested, based on its characteristics, and is evaluated using metrics. This section also involves further tuning of the model that has highest potential (based on the metric defined)
- Concluding Remarks - This section summarizes the whole project, along with a couple of thoughts and possible improvements as future work.
## <a name="installations"></a> Installations

- Python 3.6+
- Jupyter 
- Other files to run the web app can be found in `requirements.txt`


## <a name="data"></a> Data

The data is provided by Udacity. Sample data is available in the `data` folder in json format. <br>
A broad overview of the raw data:

- `artist` - the artist of the soundtrack
- `auth` - variable indicating whether the user has cancelled the subscription or not
- `firstName` - first name of the user
- `gender` - gender of the user
- `ItemInSession` - Item ID for each session (row) recorded
- `lastName` - last name of the user
- `length` - length of each session by the user
- `level` - the level of subscription of the user (free trial or paid)
- `location` - location data of the user (city and state)
- `page` - the page in Sparkify the user visited in each session
- `song` - the song listened to in each session, by the user
- `ts` - the timestamp of each session
- `userAgent` - the user agent used by the user to visit sparkify
- `userId` - unique number identifying each user

The Target variable for modelling, which indicates whether the customer has churned or not, is indicated by the event `Cancellation Confirmation`, in the `page` column. This is then encoded into a `Churn` variable.

For more information about the data, visit the Sparkify Notebook in notebooks folder.

## <a name="modelling"></a> Modelling

1. The classification algorithms tested here are - `RandomForestClassifier`, Gradient Boosted Trees Classifier (`GBTClassifier`), `LogisticRegression` and Linear Support Vector Machines (`LinearSVM`). The thorough documentation for the algorithms and its application is available in the [Spark ML documentation](https://spark.apache.org/docs/latest/ml-classification-regression.html).
2. Based on the performance metric, the `F1-Score`, 
the models are evaluated and hyperparameter tuning is done, 
which is the second part in  the notebook. 
For more detailed discussion of the parameters and algorithms 
selected to tune, visit the section Modelling, linked in the 
Sparkify notebook in the repo.

## <a name = "results"></a> Results 

The Random Forest classifier performed the best, with an `F1-Score of 73.86%` approximately. The other algorithms performed  well too. However:

- This result must be taken with a grain of salt, as the target variable `Churn`, is imbalanced. Even if the F1-Score does account into the metric the False Positives and False Negatives.
- Further SMOTE-like oversampling or undersampling to equalize the size of the two classes could be more unbiased, but they come with disadvantages too. They are discussed in the last section of the notebook above.

## <a name = "flask"></a> Deployment using flask
The web application is implemented with:

- Flask running the back-end,
- Bootstrap controls of front-end. The web application consists of the following parts:
-    Python script process_data.py which builds the machine learning model. This script accepts the path to the dataset and the path where the resulting model should be saved as parameters.
- The machine learning model, which is created by script create_model.py. The application loads the model and uses it to make predictions.
- Python script run.py, which runs the logic of the application and renders web pages. The script loads the model on start and applies it to make predictions out of the data provided by the user on the web page.
- Web page templates master.html and go.html of application web pages. Pages use bootstrap controls. The web application allows the user to enter the information about the customer and then tells whether the customer is about to churn based on this information.

#### Running the web App
There are two steps to get the application up and runnning with the web app.


- Data Cleaning and ML Model Building
Go to the project directory and then run the following command:

`python data/process_data.py data/mini_sparkify_event_data.json model/classifier`

- This will perform cleaning operations on the data and build a machine learning model that is saved to 'model' folder.


- Starting the web app
After performing the above processes, you can now run the web app to see the visualizations.

Go the app directory and run the following command:


`python run.py`
- This will start the web app and will direct you to a URL where you can enter messages and get classification results for it. 
- P.S: Running on Windows requires you to use http://localhost:8001