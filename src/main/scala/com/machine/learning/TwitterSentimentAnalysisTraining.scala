package com.machine.learning
import org.apache.spark.SparkConf
import org.apache.spark.ml.feature.{CountVectorizer, RegexTokenizer, StopWordsRemover}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.Pipeline
import org.apache.spark.streaming.twitter._


/**
  * Created by mk13935 on 7/28/2016.
  */
object TwitterSentimentAnalysisTraining {

  def main(args: Array[String]) {
    if (args.length == 0) {
      System.err.println("Usage: " + this.getClass.getSimpleName + " <training file> ")
      System.exit(1)
    }
  }

  val sparkConf = new SparkConf().
    setAppName("Twitter Sentiment Analyzer")
    .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")


}
