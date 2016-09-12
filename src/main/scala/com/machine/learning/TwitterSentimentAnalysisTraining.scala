package com.machine.learning
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.sql.SparkSession


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
      .config("spark.sql.warehouse.dir", "file:///c:/tmp/spark-warehouse")
      .getOrCreate()

    //val stopWords = spark.broadcast(loadStopWords("/stopwords.txt")).value

    val allData = spark.
      read.
      option("header", "true").
      option("inferSchema", "true") // Automatically infer data types
      .csv(args(0))
//    allData.show(20)
    val splits = allData.randomSplit(Array(0.8, 0.2), seed = 11L)
    val trainingData = splits(0)
    val testData = splits(1)

    val tokenizer = new Tokenizer().setInputCol("SentimentText").setOutputCol("words")
    //    print(tokenizer.inputCol)

    val wordsData = tokenizer.transform(trainingData)

    val hashingTF = new HashingTF()
      .setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(10000)

    val featurizeTrainingdData = hashingTF.transform(wordsData)

    //featurizeTrainingdData.printSchema()



    println("\n\n********* Training **********\n\n")
    val model = new NaiveBayes().setFeaturesCol("rawFeatures").setLabelCol("Sentiment").fit(featurizeTrainingdData)


    println("\n\n********* Testing **********\n\n")
    val wordsTestData = tokenizer.transform(testData)
    val featurizeTestdData = hashingTF.transform(wordsTestData)



    val prediction = model.transform(featurizeTestdData)

    //prediction.show()

    // Select (prediction, true label) and compute test error
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("Sentiment")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    val accuracy = evaluator.evaluate(prediction)

    println("Accuracy: " + accuracy)
  }

}
