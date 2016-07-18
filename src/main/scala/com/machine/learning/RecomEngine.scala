package com.machine.learning

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.ml.recommendation.ALS.Rating
import org.apache.spark.sql.DataFrame


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

    val ratingsFilePath = args(0)
    val usersFilePath = args(1)
    val moviesFilePath = args(2)

    val conf = new SparkConf().setAppName("Sample Recommendation Engine")
    val sc = new SparkContext(conf)
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)

    import sqlContext.implicits._

    val ratingsDF = sc.textFile(ratingsFilePath).map(_.split("::")).
      map(r => Rating(r(0).toInt,r(1).trim.toInt,r(2).trim.toFloat)).toDF()

    println("Total number of ratings: " + ratingsDF.count())

    println("Total number of movies rated: " + ratingsDF.select("item").distinct().count())

    println("Total number of users who rated movies: " + ratingsDF.select("user").distinct().count())

    val usersDF = sc.textFile(usersFilePath).map(parseUser).toDF()

    val moviesDF = sc.textFile(moviesFilePath).map(parseMovie).toDF()


  }
}