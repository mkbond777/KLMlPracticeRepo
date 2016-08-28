package com.machine.learning
import org.apache.spark.SparkConf
import org.apache.spark.ml.feature._
import org.apache.spark.sql.functions._
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.SparkSession
import org.apache.spark.streaming.twitter._
import org.apache.spark.sql.execution.datasources.csv


/**
  * Created by mk13935 on 7/28/2016.
  */
object TwitterSentimentAnalysisTraining {

  def main(args: Array[String]) {
    if (args.length == 0) {
      System.err.println("Usage: " + this.getClass.getSimpleName + " <training file> ")
      System.exit(1)
    }


    val spark = SparkSession
      .builder
      .appName("Twitter Sentiment Analyzer")
      .config("spark.scheduler.mode", "FAIR").config("spark.sql.warehouse.dir", "file:///c:/tmp/spark-warehouse")
      .getOrCreate()

    //val stopWords = spark.broadcast(loadStopWords("/stopwords.txt")).value

    val allData = spark.
      read.
      option("header", "true").
      option("inferSchema", "true").csv(args(0)) // Automatically infer data types
    allData.show(20)
    val splits = allData.randomSplit(Array(0.8, 0.2), seed = 11L)
    val training = splits(0)
    val test = splits(1)

    val tokenizer = new Tokenizer().setInputCol("SentimentText").setOutputCol("words")
    //    print(tokenizer.inputCol)

    val wordsData = tokenizer.transform(training)

    val hashingTF = new HashingTF()
      .setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(20)

    val featurizedData = hashingTF.transform(wordsData)

    featurizedData.show()




  }
}
