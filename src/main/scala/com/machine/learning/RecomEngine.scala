package com.machine.learning

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.ml.recommendation.ALS.Rating


//import org.apache.spark.sql.DataFrame


case class Movie(movieId: Int, title: String, genres: Seq[String])

case class User(userId: Int, gender: String, age: Int,
                occupation: Int, zip: String)


object RecomEngine {

  private def parseMovie(str: String): Movie = {
    val fields = str.split("::")
    assert(fields.size == 3)
    Movie(fields(0).toInt, fields(1), Seq(fields(2)))
  }

  private def parseUser(str: String): User = {
    val fields = str.split("::")
    assert(fields.size == 5)
    User(fields(0).toInt, fields(1), fields(2).toInt,
      fields(3).toInt, fields(4))
  }

  def main(args: Array[String]) {

    val ratingsFilePath = args(0)
    val usersFilePath = args(1)
    val moviesFilePath = args(2)

    val conf = new SparkConf().setAppName("Sample Recommendation Engine")
    val sc = new SparkContext(conf)
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)

    import sqlContext.implicits._

    val ratingsRDD = sc.textFile(ratingsFilePath).map(_.split("::")).
      map(r => Rating(r(0).toInt, r(1).trim.toInt, r(2).trim.toFloat))

    val ratingsDF = ratingsRDD.toDF()

    //println("Total number of ratings: " + ratingsDF.count())

    //println("Total number of movies rated: " + ratingsDF.select("item").distinct().count())

    //println("Total number of users who rated movies: " + ratingsDF.select("user").distinct().count())

    val usersDF = sc.textFile(usersFilePath).map(parseUser).toDF()

    val moviesDF = sc.textFile(moviesFilePath).map(parseMovie).toDF()

    usersDF.printSchema()

    moviesDF.printSchema()

    ratingsDF.printSchema()

    ratingsDF.registerTempTable("ratings")
    moviesDF.registerTempTable("movies")
    usersDF.registerTempTable("users")

    val results = sqlContext.sql(
      """select movies.title, movierates.maxr, movierates.minr, movierates.cntu
    from(SELECT ratings.item, max(ratings.rating) as maxr,
    min(ratings.rating) as minr,count(distinct user) as cntu
    FROM ratings group by ratings.item ) movierates
    join movies on movierates.item=movies.movieId
    order by movierates.cntu desc""")

    //results.show()

    //results.coalesce(10).write.format("json").mode("overwrite").save("file:\\C:\\Manish\\amazonMl\\abc")

    // Show the top 10 most-active users and how many times they rated
    // a movie
    val mostActiveUsersSchemaRDD = sqlContext.sql(
      """SELECT ratings.user, count(*) as ct from ratings
  group by ratings.user order by ct desc limit 10""")

    //println(mostActiveUsersSchemaRDD.collect().mkString("\n"))

    // Find the movies that user 4169 rated higher than 4
    val resultsFor4169 = sqlContext.sql("""SELECT ratings.user, ratings.item as MovieID,
  ratings.rating, movies.title FROM ratings JOIN movies
  ON movies.movieId=ratings.item
  where ratings.user=4169 and ratings.rating > 4""")

    //resultsFor4169.show

    val splits = ratingsDF.randomSplit(Array(0.8, 0.2), 0L)

    val trainingRatingsRDD = splits(0).cache()
    val testRatingsRDD = splits(1).cache()

    val numTraining = trainingRatingsRDD.count()
    val numTest = testRatingsRDD.count()
    println(s"Training: $numTraining, test: $numTest.")

    // build a ALS user product matrix model with rank=20, iterations=10
    val model = new ALS().setRank(20).setMaxIter(10).fit(trainingRatingsRDD)

    model.





  }
}