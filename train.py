#!/usr/bin/env python
# -*- coding: utf-8 -*-


# We need sys to get the command line arguments
import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession

# TODO: you may need to add imports here

from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer

def main(spark, model_file):

    ###
    train = spark.read.parquet('./train.parquet')
    #validation = spark.read.parquet('hdfs:/user/bm106/pub/project/cf_validation.parquet')
    #test = spark.read.parquet('hdfs:/user/bm106/pub/project/cf_test.parquet')

    als = ALS(rank =5, regParam = 10, alpha = 25, userCol = 'convert_user_id', \
              itemCol = 'convert_track_id', ratingCol = 'count', implicitPrefs = True)
    train_model = als.fit(train)
    train_model.save(model_file)


# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('recommender_train').getOrCreate()

    # And the location to store the trained model
    model_file = sys.argv[1]

    # Call our main routine
    main(spark, model_file)
