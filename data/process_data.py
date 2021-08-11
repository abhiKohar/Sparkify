import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import re
from sqlalchemy import create_engine
from datetime import datetime

import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler, Normalizer
from pyspark.ml.feature import OneHotEncoderEstimator
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType, StringType
from pyspark.sql.functions import to_date, datediff
from pyspark.sql.functions import lit, avg, when, count, col, min, max, round
from pyspark.sql import Window
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.mllib.evaluation import MulticlassMetrics

def load_data(data_filepath):
    '''
    Function, which loads dataset
    
    INPUT:
        path - path to json file, which contains Sparkify data
        
    OUTPUT:
        df - pyspark dataframe, which contains dataset
        spark - spark context
    '''

    spark = SparkSession.builder \
        .master("local") \
        .appName("Sparkify") \
        .config("spark.sql.broadcastTimeout", "36000") \
        .getOrCreate()
    
    mini_sparkify_event_data = data_filepath
    
    df = spark.read.json(mini_sparkify_event_data)
    df.persist()

    return spark, df


def clean_data(df):
    """
    This function is used to clean the row dataset
    input: dataframe to be cleaned
    output: final clean dataframe 
    """   
    

    # drop na values from the userId, sessionId
    # df_clean = df.dropna(how = "any", subset = ["userId", "sessionId"]).dropDuplicates()
    # drop empty values from the userId
    df_clean = df.filter(df["userId"] != "")
    
    return df_clean


def process_dataset(df):
    '''
    Function for preparation of dataset for machine learning
    INPUT:
    df - initial dataset loaded from json file
    
    OUTPUT:
    df_ft - new dataset prepared for machine learning
    contains the following columns:
    1. userId - initial id of the user
    2. gender - user's gender
    3. avg_events - average number of events per day for the user
    4. avg_songs - average number of songs the user listens to per day
    5. thumbs_up - number of thumbs up events
    6. thumbs_down - number of thumbs down events
    7. active_days - days since user's firts event
    8. last_location - location of the last event
    9. last_level - user's last level (paid or free)
    10. addfriends - number of add friends events
    '''
    
    # clean dataset using clean_data function
    df = clean_data(df)
    
    # define cancellation udf
    cancellation_event = udf(lambda x: 1 if x == "Cancellation Confirmation" else 0, IntegerType())
    
    # set churn = 1 for rows where page == 'Cancellation Confirmation'
    df = df.withColumn("churn", cancellation_event("page"))
    
    
    # get userId with churn == 1
    cancelled_users = df.select(['userId', 'churn']).where(df.churn == 1).groupby('userId').count().toPandas()['userId'].values
    
    # create udf, which sets churn of a row to 1 if userId is in cancelled_users list
    def replace_data(userId, features):
        if(userId in cancelled_users): return 1
        else : return 0
    
    # set churn == 1 for all rows for users who cancelled their subscription
    fill_array_udf = udf(replace_data, IntegerType())
    df = df.withColumn("churn", fill_array_udf(col("userId"), col("churn")))
        
    # set column last ts with the first and the last event timestamp
    w = Window.partitionBy('userId')
    df = df.withColumn('last_ts', max('ts').over(w))
    df = df.withColumn('first_ts', min('ts').over(w))
    
    # convert timestamp to date (string)
    def get_date(ts):
        return str(datetime.utcfromtimestamp(ts / 1000).strftime('%Y-%m-%d'))
    
    get_date_from_ts_udf = udf(get_date, StringType())
    df = df.withColumn('last_date', get_date_from_ts_udf(col('last_ts')))
    df = df.withColumn('first_date', get_date_from_ts_udf(col('first_ts')))
    
    # add column date and convert timetamp to date
    df = df.withColumn('date', get_date_from_ts_udf(col('ts')))
    
    # set column last_level to level when timestamp is last timestamp
    df = df.withColumn('last_level', when(df.last_ts == df.ts, df.level))
    
    #aditional feature: Gender
    # flag_gender = udf(lambda x: 1 if x == 'M' else 0, IntegerType())
    # gender = df.select("userId", "gender").dropDuplicates()
    # gender = df.withColumn("gender", flag_gender("gender"))
    
    # create column avg_songs to calculate average number of events per day
    w = Window.partitionBy('userId', 'date')
    events = df.select('userId', 'date', count('userId').over(w).alias('events')).distinct()
    w = Window.partitionBy('userId')
    events = events.withColumn('avg_events', avg('events').over(w))
    events = events.select(col("userId").alias("events_userId"), 'avg_events')
    events = events.withColumn("avg_events", round(events["avg_events"], 2))
    
    # create column avg_songs to calculate average number of songs per day
    w = Window.partitionBy('userId', 'date')
    songs = df.where(df.page == 'NextSong').select('userId', 'date', count('userId').over(w).alias('songs')).distinct()
    w = Window.partitionBy('userId')
    songs = songs.withColumn('avg_songs', avg('songs').over(w))
    songs = songs.select(col("userId").alias("songs_userId"), 'avg_songs')
    songs = songs.withColumn("avg_songs", round(songs["avg_songs"], 2))
    
    # calculate number of thumbs up for a user
    w = Window.partitionBy('userId')
    thumbsup = df.where(df.page == 'Thumbs Up').select('userId', count('userId').over(w).alias('thumbs_up')).distinct()
    thumbsup = thumbsup.select(col("userId").alias("thumbsup_userId"), 'thumbs_up')
    
    # calculate number of thumbs down for a user
    w = Window.partitionBy('userId')
    thumbsdown = df.where(df.page == 'Thumbs Down').select('userId', count('userId').over(w).alias('thumbs_down')).distinct()
    thumbsdown = thumbsdown.select(col("userId").alias("thumbsdown_userId"), 'thumbs_down')
    
    # calculate days since the date of the first event
    df = df.withColumn("days_active", 
              datediff(to_date(lit(datetime.now().strftime("%Y-%m-%d %H:%M"))),
                       to_date("first_date","yyyy-MM-dd")))
        
    # add column with state of the event based on location column
    def get_state(location):
        location = location.split(',')[-1].strip()
        if (len(location) > 2):
            location = location.split('-')[-1].strip()
    
        return location
    
    state_udf = udf(get_state, StringType())
    df = df.withColumn('state', state_udf(col('location')))
    
    #add column with last location of the user
    df = df.withColumn('last_state',when(df.last_ts == df.ts, df.state))
    
    # calculate number of add friends for a user
    w = Window.partitionBy('userId')
    addfriend = df.where(df.page == 'Add Friend').select('userId', count('userId').over(w).alias('addfriend')).distinct()
    addfriend = addfriend.select(col("userId").alias("addfriend_userId"), 'addfriend')
    
    # merge all results together
    df_ft = df.select('userId', 'gender', 'churn', 'last_level', 'days_active', 'last_state')\
    .dropna().drop_duplicates()
    
    df_ft = df_ft.join(songs, df_ft.userId == songs.songs_userId).distinct()
    df_ft = df_ft.join(events, df_ft.userId == events.events_userId).distinct()
    df_ft = df_ft.join(thumbsup, df_ft.userId == thumbsup.thumbsup_userId, how='left').distinct()
    df_ft = df_ft.fillna(0, subset=['thumbs_up'])
    df_ft = df_ft.join(thumbsdown, df_ft.userId == thumbsdown.thumbsdown_userId, how='left').distinct()
    df_ft = df_ft.fillna(0, subset=['thumbs_down'])
    df_ft = df_ft.join(addfriend, df_ft.userId == addfriend.addfriend_userId, how='left').distinct()
    df_ft = df_ft.fillna(0, subset=['addfriend'])
    df_ft = df_ft.drop('songs_userId','events_userId', 'thumbsup_userId', 'thumbsdown_userId', 'addfriend_userId')
    
    return df, df_ft

def build_model(df_ft):
    '''
    Function builds machine learning model to predict churn
    
    INPUT:
        df_ft - dataset which contains user features to predict customer churn
        
    OUTPUT:
        model - model which predicts customer churn
    '''
    
    # split into train, test and validation sets (60% - 20% - 20%)
    df_ft = df_ft.withColumnRenamed("churn", "label")

    train, test_valid = df_ft.randomSplit([0.6, 0.4], seed = 42)
    test, validation = test_valid.randomSplit([0.5, 0.5], seed = 42)

    # index and encode categorical features gender, level and state

    stringIndexerGender = StringIndexer(inputCol="gender", outputCol="genderIndex", handleInvalid = 'skip')
    stringIndexerLevel = StringIndexer(inputCol="last_level", outputCol="levelIndex", handleInvalid = 'skip')
    stringIndexerState = StringIndexer(inputCol="last_state", outputCol="stateIndex", handleInvalid = 'skip')

    encoder = OneHotEncoderEstimator(inputCols=["genderIndex", "levelIndex", "stateIndex"],
                                        outputCols=["genderVec", "levelVec", "stateVec"],
                                    handleInvalid = 'keep')

    # create vector for features
    features = ['genderVec', 'levelVec', 'stateVec', 'days_active', 'avg_songs', 'avg_events', 'thumbs_up', 'thumbs_down', 'addfriend']
    assembler = VectorAssembler(inputCols=features, outputCol="rawFeatures")
    
    # normalize features
    normalizer = Normalizer(inputCol="rawFeatures", outputCol="features", p=1.0)

     # initialize random forest classifier with tuned hyperparameters
    rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=100, impurity = 'gini', maxDepth = 5, featureSubsetStrategy = 'sqrt')

    # assemble pipeline
    pipeline = Pipeline(stages = [stringIndexerGender, stringIndexerLevel, stringIndexerState, encoder, assembler, normalizer, rf])
    
    # fit model
    model = pipeline.fit(train)
    
    # predict churn
    pred_train = model.transform(train)
    pred_test = model.transform(test)
    pred_valid = model.transform(validation)
    
    # evaluate results
    predictionAndLabels = pred_train.rdd.map(lambda lp: (float(lp.prediction), float(lp.label)))

    # Instantiate metrics object
    metrics = MulticlassMetrics(predictionAndLabels)

    # print F1-score
    print("F1 score on train dataset is %s" % metrics.fMeasure())
    
    predictionAndLabels = pred_test.rdd.map(lambda lp: (float(lp.prediction), float(lp.label)))

    # Instantiate metrics object
    metrics = MulticlassMetrics(predictionAndLabels)

    # F1 score
    print("F1 score on test dataset is %s" % metrics.fMeasure())
    
    predictionAndLabels = pred_valid.rdd.map(lambda lp: (float(lp.prediction), float(lp.label)))

    # Instantiate metrics object
    metrics = MulticlassMetrics(predictionAndLabels)

    # F1 score
    print("F1 score on validation dataset is %s" % metrics.fMeasure())
    
    return model


def save_model(sc, model, model_filepath):
    '''
    Function saves created model
    
    INPUTS:
        1. sc - spark context
        2. model - model to be saved
        3. model_filepath - filepath where to save the model
    '''
    #pickle.dump(model, open(model_filepath, 'wb'))
    model.save(model_filepath)


def main():
    if len(sys.argv) == 3:

        data_filepath, model_filepath = sys.argv[1:]

        print('Loading data...\n    DATA: {} '\
              .format(data_filepath))
        sc, df = load_data(data_filepath)

        # clean data and prepare dataset for machine learning
        print('Processing data \n')
        df, df_ml = process_dataset(df)
        
        df_ml.persist()
        # build machine learning model
        print('Building model \n')
        model = build_model(df_ml)
        
        # save model
        print('Saving model \n')
        save_model(sc, model, model_filepath)
        
        print('Model successfully created \n')
       
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()