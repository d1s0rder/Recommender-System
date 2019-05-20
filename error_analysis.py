#!/usr/bin/env python
# -*- coding: utf-8 -*-


# We need sys to get the command line arguments
import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession

# TODO: you may need to add imports here

import pyspark.sql.functions as F

def main(spark):

    ###
    # Read files: true, pred, train_track, meta, tag
    # true('user_id', 'true_track_id')
    # pred('user_id', 'pred_track_id')
    # train_track('track_id','convert_track_id')

    true = spark.read.parquet('./true_count.parquet')
    #true = true.sample(False,0.01) Only used for downsample
    pred = spark.read.parquet('./recommendation_count.parquet')
    train_track = spark.read.parquet('./train_track.parquet')
    meta = spark.read.parquet('hdfs:/user/bm106/pub/project/metadata.parquet')
    tag = spark.read.parquet('hdfs:/user/bm106/pub/project/tags.parquet')

    true.createOrReplaceTempView('true')
    pred.createOrReplaceTempView('pred')
    train_track.createOrReplaceTempView('train_track')
    meta.createOrReplaceTempView('meta')
    tag.createOrReplaceTempView('tag')

    pred = spark.sql('SELECT * FROM pred WHERE convert_user_id IN (SELECT convert_user_id FROM true)')
    true = true.withColumn('items', F.explode(F.col('true_items')))
    pred = pred.withColumn('items', F.explode(F.col('convert_track_id')))
    true.createOrReplaceTempView('true')
    pred.createOrReplaceTempView('pred')

    under = spark.sql('SELECT distinct items FROM true WHERE convert_user_id IN (SELECT convert_user_id FROM pred) AND items not in (SELECT items from pred)')
    over = spark.sql('SELECT distinct items FROM pred WHERE convert_user_id IN (SELECT convert_user_id FROM true) AND items not in (SELECT items from true)')
    under.createOrReplaceTempView('under')
    over.createOrReplaceTempView('over')

    over = spark.sql('SELECT track_id FROM train_track  WHERE convert_track_id IN (SELECT items FROM over) ')
    under = spark.sql('SELECT track_id FROM train_track  WHERE convert_track_id IN (SELECT items FROM under)')
    over.createOrReplaceTempView('over')
    under.createOrReplaceTempView('under')


  
    over_year = spark.sql('SELECT year, COUNT(track_id) as freq FROM meta WHERE track_id IN (SELECT track_id FROM over) GROUP BY year ORDER BY freq DESC ')
    under_year = spark.sql('SELECT year, COUNT(track_id) as freq FROM meta WHERE track_id IN (SELECT track_id FROM under) GROUP BY year ORDER BY freq DESC ')	
    over_year.show()
    under_year.show()

    over_score = spark.sql('SELECT score, COUNT(track_id) as freq FROM tag WHERE track_id IN (SELECT track_id FROM over) GROUP BY score ORDER BY freq DESC ')
    under_score = spark.sql('SELECT score, COUNT(track_id) as freq FROM tag WHERE track_id IN (SELECT track_id FROM under) GROUP BY score ORDER BY freq DESC ')
    over_score.show()
    under_score.show()                           

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('recommender_train').getOrCreate()

    # And the location to store the trained model
    # model_file = sys.argv[1]

    # Call our main routine
    main(spark)                               
