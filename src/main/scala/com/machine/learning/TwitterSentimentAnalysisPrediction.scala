package com.machine.learning


import org.apache.spark.SparkConf
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.SparkSession
import org.apache.spark.streaming.twitter.TwitterUtils
import org.apache.spark.streaming.{Seconds, StreamingContext}
import twitter4j.auth.OAuthAuthorization
import twitter4j.conf.ConfigurationBuilder

/**
  * Created by mk13935 on 9/1/2016.
  */
object TwitterSentimentAnalysisPrediction {
  def main(args: Array[String]) {

    println("Initializing Streaming Spark Context...")
    val conf = new SparkConf().setAppName(this.getClass.getSimpleName)
    val ssc = new StreamingContext(conf, Seconds(5))

    println("Initializing Twitter stream...")
    val tweets = TwitterUtils.createStream(ssc, getAuth)
    val statuses = tweets.filter(_.getLang == "en").map(_.getText)

    twitter4j.
    val model = new NaiveBayes().fit(ssc.sparkContext, Array(ParamMap))

  }

  def getAuth = {
    Some(new OAuthAuthorization(new ConfigurationBuilder().build()))
  }
}
