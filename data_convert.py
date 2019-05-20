#!/usr/bin/env python
# -*- coding: utf-8 -*-


# We need sys to get the command line arguments
import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession

# TODO: you may need to add imports here
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer

def main(spark):

    ###


    train = spark.read.parquet('hdfs:/user/bm106/pub/project/cf_train.parquet')
    validation = spark.read.parquet('hdfs:/user/bm106/pub/project/cf_validation.parquet')
    test = spark.read.parquet('hdfs:/user/bm106/pub/project/cf_test.parquet')


    indexers = [StringIndexer(inputCol = column, outputCol = 'convert_'+column, handleInvalid = 'keep')\
               for column in ['user_id', 'track_id']]
    pipeline = Pipeline(stages = indexers)
    train_fit = pipeline.fit(train)
    train_transform = train_fit.transform(train)
    train_original = train_transform.select('track_id', 'convert_track_id')
    train_original.write.parquet('train_track.parquet')
    #train_final = train_transform.select('convert_user_id', 'convert_track_id', 'count')
    #train_final.write.parquet('train.parquet')

    #validation_transform = train_fit.transform(validation)
    #validation_final = validation_transform.select('convert_user_id', 'convert_track_id', 'count')
    #validation_final.write.parquet('validation.parquet')

    #test_transform = train_fit.transform(test)
    #test_final = test_transform.select('convert_user_id', 'convert_track_id', 'count')
    #test_final.write.parquet('test.parquet')



# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('data_convert').getOrCreate()

    # Call our main routine
    main(spark)
