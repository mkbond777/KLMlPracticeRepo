package com.machine.learning

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.recommendation.ALS

case class Movie(movieId: Int, title: String, genres: Seq[String])

case class User(userId: Int, gender: String, age: Int,
               occupation: Int, zip: String)


object RecomEngine{

  private def parseMovie(str: String): Movie = {
    val fields = str.split("::")
    assert(fields.size == 3)
    Movie(fields(0).toInt, fields(1), Seq(fields(2)))
  }

  private def parseUser(str: String): User = {
    val fields = str.split("::")
    assert(fields.size == 5)
    User(fields(0).toInt, fields(1).toString, fields(2).toInt,
      fields(3).toInt, fields(4).toString)
  }

  def main(args: Array[String]) {

    val conf = new SparkConf().setAppName("Simple Application")
    val sc = new SparkContext(conf)
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    //val ratingDf = sqlContext.

  }
}