package com.machine.learning


import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.recommendation.{ALS, ALSModel}
import org.apache.spark.ml.recommendation.ALS.Rating


//import org.apache.spark.sql.DataFrame

//input format MovieID::Title::Genres
case class Movie(movieId: Int, title: String, genres: Seq[String])

//input format is UserID::Gender::Age::Occupation::Zip-code
case class User(userId: Int, gender: String, age: Int,
                occupation: Int, zip: String)


object RecomEngine {

  // function to parse input into Movie class
  private def parseMovie(str: String): Movie = {
    val fields = str.split("::")
    assert(fields.size == 3)
    Movie(fields(0).toInt, fields(1), Seq(fields(2)))
  }

  // function to parse input into User class
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

    // load the data into a RDD and then convert it to DF
    val ratingsRDD = sc.textFile(ratingsFilePath).map(_.split("::")).
      map(r => Rating(r(0).toInt, r(1).trim.toInt, r(2).trim.toFloat))

    val ratingsDF = ratingsRDD.toDF()

    val usersDF = sc.textFile(usersFilePath).map(parseUser).toDF()

    val moviesDF = sc.textFile(moviesFilePath).map(parseMovie).toDF()

    //println("Total number of ratings: " + ratingsDF.count())

    //println("Total number of movies rated: " + ratingsDF.select("item").distinct().count())

    //println("Total number of users who rated movies: " + ratingsDF.select("user").distinct().count())

    usersDF.printSchema()

    moviesDF.printSchema()

    ratingsDF.printSchema()

    //Registering dataframes as temp table.
    ratingsDF.registerTempTable("ratings")
    moviesDF.registerTempTable("movies")
    usersDF.registerTempTable("users")

    // Get the max, min ratings along with the count of users who have
    // rated a movie.
    val results = sqlContext.sql(
      """select movies.title, movierates.maxr, movierates.minr, movierates.cntu
    from(SELECT ratings.item, max(ratings.rating) as maxr,
    min(ratings.rating) as minr,count(distinct user) as cntu
    FROM ratings group by ratings.item ) movierates
    join movies on movierates.item=movies.movieId
    order by movierates.cntu desc""")

    //results.show()

    // Show the top 10 most-active users and how many times they rated
    // a movie
    val mostActiveUsersSchemaDF = sqlContext.sql(
      """SELECT ratings.user, count(*) as ct from ratings
  group by ratings.user order by ct desc limit 10""")

    //println(mostActiveUsersSchemaRDD.collect().mkString("\n"))

    // Find the movies that user 4169 rated higher than 4
    val resultsFor4169 = sqlContext.sql("""SELECT ratings.user, ratings.item as MovieID,
  ratings.rating, movies.title FROM ratings JOIN movies
  ON movies.movieId=ratings.item
  where ratings.user=4169 and ratings.rating > 4""")

    //resultsFor4169.show

    // Randomly split ratings DF into training
    // data DF (80%) and test DF RDD (20%)
    val splits = ratingsDF.randomSplit(Array(0.8, 0.2), 0L)

    val trainingRatingsDF = splits(0)
    val testRatingsDF = splits(1)

    //val numTraining = trainingRatingsDF.count()
    //val numTest = testRatingsDF.count()
    println("Training Data count: " + trainingRatingsDF.count() + "and test data Count: " + testRatingsDF.count())

    // build a ALS user product matrix model with rank=5, iterations=2
    val model : ALSModel = new ALS().setRank(5).setMaxIter(2).fit(trainingRatingsDF)


    // Get the top 4 movie predictions for user 4169
    val topRecsForUsers = model
      .transform(trainingRatingsDF)

    val movieRecomfor4169 = topRecsForUsers.join(moviesDF, topRecsForUsers("item") === moviesDF("movieId"))
      .select("user","title","prediction").where("user = 4169")
      .sort($"prediction".desc)

    println("Movie Recommendation for 4169 :" + movieRecomfor4169.show)

    //prepare test rating for comparison
    val testUserItemDf = testRatingsDF.select("user","item")

    // get predicted ratings to compare to test ratings
    val predictionsForTestDF  = model.setPredictionCol("prating").transform(testUserItemDf)

    //predictionsForTestDF.show()

    //Join the test with predictions
    val testAndPredictionsJoinedDF = testRatingsDF.join(predictionsForTestDF,Seq("user","item"))

    //testAndPredictionsJoinedDF.show(5)


    //finds false positives by finding predicted ratings which were >= 4 when the actual test rating was <= 1
    val falsePositives = testAndPredictionsJoinedDF.where("rating<=1 and prating >= 4")

    val countFalsePositives=falsePositives.count()

    println(countFalsePositives)

    // Evaluate the model using Mean Absolute Error (MAE) between test
    // and predictions
    val meanAbsoluteError = testAndPredictionsJoinedDF.withColumn("err", $"rating" - $"prating")
      .selectExpr("abs(err) as absErr")
      .groupBy("absErr")
      .avg("absErr")

    println("meanAbsoluteError = " + meanAbsoluteError.show)

  }
}