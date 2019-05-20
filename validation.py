#!/usr/bin/env python
# -*- coding: utf-8 -*-


# We need sys to get the command line arguments
import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession

# TODO: you may need to add imports here

from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer

def main(spark, model_file):

    ###
    train = spark.read.parquet('./train.parquet')
    validation = spark.read.parquet('./validation.parquet')
    #test = spark.read.parquet('./test.parquet')


    train_model = ALSModel.load(model_file)
    users = validation.select('convert_user_id').distinct()
    user_recs = train_model.recommendForUserSubset(users, 500)
    prediction_df = user_recs.select('convert_user_id','recommendations.convert_track_id')
    true_df = validation.groupBy('convert_user_id').agg(expr('collect_list(convert_track_id) as true_items'))

    prediction_rdd = prediction_df.join(true_df, 'convert_user_id') \
    .rdd \
    .map(lambda row: (row[1], row[2]))
    rankingMetrics = RankingMetrics(prediction_rdd)
    print(rankingMetrics.meanAveragePrecision)
    print(rankingMetrics.precisionAt(500))


# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('recommender_validation').config("spark.sql.broadcastTimeout", "36000").getOrCreate()

    # And the location to store the trained model
    model_file = sys.argv[1]

    # Call our main routine
    main(spark, model_file)
