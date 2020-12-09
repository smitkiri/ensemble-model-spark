package ensemble

import nb.NaiveBayesClassifier
import org.apache.log4j.LogManager
import org.apache.spark.{HashPartitioner, SparkConf, SparkContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import scala.collection.mutable
import scala.collection.mutable.ListBuffer


object NaiveBayesEnsemble {

  def createBootStrap(train: Array[Array[Double]], seed: Int): Array[Array[Double]] = {
    val r = new scala.util.Random
    r.setSeed(seed)

    val length = train.length
    var bootstrap: ListBuffer[Array[Double]] = new ListBuffer[Array[Double]]()

    while(bootstrap.length < length) {
      val index = r.nextInt(length)
      bootstrap += train(index)
    }

    bootstrap.toArray
  }

  def main(args: Array[String]) {
    val logger: org.apache.log4j.Logger = LogManager.getRootLogger
    if (args.length != 3) {
      logger.error("Usage:\nensemble.NaiveBayesEnsemble <input dir> <output dir> <num-models>")
      System.exit(1)
    }
    val conf = new SparkConf().setAppName("Naive Bayes Ensemble Classification")
    val sc = new SparkContext(conf)

    val ss = SparkSession.builder().getOrCreate()
    import ss.implicits._

    // Number of trees in RandomForest
    val numModels = args(2).toInt

    val dataFrame = ss.read.format("csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load("input/sample_train.csv")

    // Split and broadcast training and testing data
    val Array(trainingData, testData) = dataFrame.randomSplit(Array(0.7, 0.3))
    val testDataWithId = testData.withColumn("Id", monotonically_increasing_id())

    val trainArray = trainingData.collect().map(_.toSeq.asInstanceOf[mutable.WrappedArray[Double]].toArray)
    val broadcastTrain = sc.broadcast(trainArray)

    val testFeatures = testDataWithId
      .drop("# label", "Id")
      .collect()
      .map(_.toSeq.asInstanceOf[mutable.WrappedArray[Double]].toArray)

    val testIdArray = testDataWithId.select("Id").collect().map(_.getLong(0))

    val broadcastTestFeatures = sc.broadcast(testFeatures)
    val broadcastTestIds = sc.broadcast(testIdArray)

    // RDD of multiple Naive Bayes models
    var modelArray = Array[(Int, NaiveBayesClassifier)]()
    for (i <- 0 until numModels) {
      modelArray = modelArray :+ (i, new NaiveBayesClassifier())
    }

    val modelRDD: RDD[(Int, NaiveBayesClassifier)] = sc.parallelize(modelArray)
      .partitionBy(new HashPartitioner(math.ceil(numModels.toDouble / 10.0).toInt))

    // Train all Naive Bayes Models
    val models = modelRDD.map {case (idx, model)  =>
      // Create a bootstrap sample for Naive Bayes
      val trainArray = broadcastTrain.value
      val bSample = createBootStrap(trainArray, idx)

      (idx, model.fit(bSample))
    }

    // Get predictions from all models
    val predictions: RDD[(Long, Int)] = models.flatMap {
      case (_, model) =>
        val preds = model.predict(broadcastTestFeatures.value)
        broadcastTestIds.value.zip(preds)
    }

    // Set the majority vote as final prediction
    val allPredictions = predictions
      .reduceByKey((x, y) => x + y)
      .mapValues(x => math.round(x.toDouble / numModels.toDouble))

    // Combine with original data for comparison
    val results = allPredictions.toDF("Id", "prediction").as("preds")
      .join(testDataWithId.as("test"), col("preds.Id") === col("test.Id"))
      .select("test.Id", "preds.prediction", "test.# label")
      .repartitionByRange(10, col("Id"))
      .sort("Id")

    // Check model performance
    val numCorrect = sc.longAccumulator("numCorrect")
    val numTotal = sc.longAccumulator("total")

    results.foreachPartition(p => {
      for (row <- p) {
        numTotal.add(1L)
        if (row(1) == row(2))
          numCorrect.add(1L)
      }
    })

    val accuracy = numCorrect.value.toDouble / numTotal.value.toDouble
    logger.info("Accuracy: " + accuracy.toString)

    results.write.option("header", "true").csv(args(1))
  }
}