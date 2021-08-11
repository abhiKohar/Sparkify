import json
import plotly
import pandas as pd
import numpy as np
from collections import Counter

import re

import operator
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from plotly.graph_objs import Pie
from datetime import datetime

import findspark
findspark.init() # find spark

from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.sql.functions import avg, col, concat, desc, explode, lit, min, max, when, split, udf, isnull, count,  mean, stddev, round
from pyspark.sql.types import IntegerType, DoubleType
from pyspark.ml import PipelineModel

app = Flask(__name__)




# get spark session
spark = SparkSession.builder \
    .master("local") \
    .appName("Sparkify") \
    .getOrCreate()

# load model
model = PipelineModel.load('../model/classifier')


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
     # get spark context
    sc = SparkContext.getOrCreate()
    
    # create spark dataframe to predict customer churn using the model
    #[gender, level, days_active, location, avgSongs, avgEvents, thumbsup, thumbsdown, add_friend]
    gender = ''
    level = 0 
    days_active = 0 
    location = 0  
    avgSongs = 0  
    avgEvents = 0  
    thumbsup = 0  
    thumbsdown = 0  
    add_friend = 0 
    df = sc.parallelize([[gender, level, days_active, location, avgSongs, avgEvents, thumbsup, thumbsdown, add_friend]]).\
    toDF(["gender", "last_level", "days_active", "last_state", "avg_songs", "avg_events" , "thumbs_up", "thumbs_down", "addfriend"])


    # df = sc.toDF(["gender", "last_level", "days_active", "last_state", "avg_songs", "avg_events" , "thumbs_up", "thumbs_down", "addfriend"])

    #Basic ananlysis for visualisations
    df.show(5)
    return render_template('master.html')


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    # get parameters from the form
    gender = request.args.get('gender', '') 
    avgEvents = request.args.get('avgEvents', 0) 
    avgSongs = request.args.get('avgSongs', 0)
    thumbsup = request.args.get('thumbsup', 0)
    thumbsdown = request.args.get('thumbsdown', 0)
    add_friend = request.args.get('add_friend', 0)
    reg_date = request.args.get('reg_date', '') # 2018-08-19
    level = request.args.get('level', '')
    location = request.args.get('location', '')
    
    # calculate number of days since the 1st event for the user
    days_active = (datetime.now() - datetime.strptime(reg_date, '%Y-%m-%d')).days
    
    # encode gender values
    if gender == 'male':
        gender = 'M'
    else:
        gender = 'F'
   
    # get spark context
    sc = SparkContext.getOrCreate()
    
    # create spark dataframe to predict customer churn using the model
    df = sc.parallelize([[gender, level, days_active, location, avgSongs, avgEvents, thumbsup, thumbsdown, add_friend]]).\
    toDF(["gender", "last_level", "days_active", "last_state", "avg_songs", "avg_events" , "thumbs_up", "thumbs_down", "addfriend"])
    
    # set correct data types
    df = df.withColumn("days_active", df["days_active"].cast(IntegerType()))
    df = df.withColumn("avg_songs", df["avg_songs"].cast(DoubleType()))
    df = df.withColumn("avg_events", df["avg_events"].cast(DoubleType()))
    df = df.withColumn("thumbs_up", df["thumbs_up"].cast(IntegerType()))
    df = df.withColumn("thumbs_down", df["thumbs_down"].cast(IntegerType()))
    df = df.withColumn("addfriend", df["addfriend"].cast(IntegerType()))
    #df = df.withColumn("last_state", df["last_state"].cast(IntegerType()))

    df.show()
    # predict using the model
    pred = model.transform(df)
    
    pred.show()
    if pred.count() == 0:
        # if model failed to predict churn then return -1
        prediction = -1
    else:
        # get prediction (1 = churn, 0 = stay)
        prediction = pred.select(pred.prediction).collect()[0][0]
    
    # print out prediction to the app console
    print("Prediction for the customer is {prediction}.".format(prediction = prediction))
    
    # render the go.html passing prediction resuls
    return render_template(
        'go.html',
        result = prediction
    )

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=8001, debug=True)


if __name__ == '__main__':
    main()