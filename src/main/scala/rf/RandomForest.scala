package rf

import nb.NaiveBayesClassifier
import org.apache.log4j.LogManager
import org.apache.spark.{HashPartitioner, SparkConf, SparkContext, sql}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.feature.{VectorAssembler, VectorSlicer}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._


object RandomForest {

  def main(args: Array[String]) {
    val logger: org.apache.log4j.Logger = LogManager.getRootLogger
    if (args.length != 3) {
      logger.error("Usage:\nrf.RandomForestMain <input dir> <output dir>")
      System.exit(1)
    }
    val conf = new SparkConf().setAppName("Naive Bayes Ensemble Classification")
    val sc = new SparkContext(conf)

    val ss = SparkSession.builder().getOrCreate()
    import ss.implicits._

    // Number of trees in RandomForest
    val numModels = 3

    var dataFrame = ss.read.format("csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load("input/sample_train.csv")

    dataFrame = dataFrame
      .withColumn("# label", dataFrame.col("# label")
        .cast(sql.types.IntegerType))

    // Split and broadcast training and testing data
    val Array(trainingData, testData) = dataFrame.randomSplit(Array(0.7, 0.3))
    val testDataWithId = testData.withColumn("Id", monotonically_increasing_id())

    val broadcastTrain = sc.broadcast(trainingData)
    val broadcastTest = sc.broadcast(testDataWithId)

    // RDD of multiple decision tree pipelines
    var modelArray = Array[(Int, NaiveBayesClassifier)]()
    for (i <- 0 until numModels) {
      modelArray = modelArray :+ (i, new NaiveBayesClassifier())
    }

    val modelRDD: RDD[(Int, NaiveBayesClassifier)] = sc.parallelize(modelArray)
      .partitionBy(new HashPartitioner(math.ceil(numModels.toDouble / 10.0).toInt))

    // Train all decision trees
    val models = modelRDD.map {case (idx, model)  =>
      // Create a bootstrap sample for Naive Bayes
      var trainArray = broadcastTrain.value.collect().map(_.toSeq.toArray)

      // Get a boostrap sample for Naive Bayes model
      val r = new scala.util.Random(idx)
      var bootStrapArray = r.shuffle(trainArray.toList).take(50)

      (idx, model.fit(bootStrapArray))
    }

    // Get predictions from all decision trees
//    val dtPredictions = models.flatMap {
//      case (_, (tree, randomFeatures)) =>
//        val slicer = new VectorSlicer()
//          .setInputCol("features")
//          .setOutputCol("sampledFeatures")
//
//        slicer.setIndices(randomFeatures)
//        val testData = slicer.transform(broadcastTest.value)
//        val pred = tree.transform(testData)
//        pred.select("Id", "prediction").rdd
//          .map(row => (row(0).asInstanceOf[Long], row(1).asInstanceOf[Double]))
//          .collect()
//    }
//
//    // Set the majority vote as final prediction
//    val allPredictions = dtPredictions
//      .reduceByKey(_.toInt + _.toInt)
//      .mapValues(x => if (x < numTrees.toDouble / 2) 0 else 1)
//
//    // Combine with original data for comparison
//    val results = allPredictions.toDF("Id", "prediction").as("preds")
//      .join(broadcastTest.value.as("test"), col("preds.Id") === col("test.Id"))
//      .select("test.Id", "preds.prediction", "test.# label")
//      .repartitionByRange(10, col("Id"))
//      .sort("Id")
//
//    val numCorrect = sc.longAccumulator("numCorrect")
//    val numTotal = sc.longAccumulator("total")
//
//    results.foreachPartition(p => {
//      for (row <- p) {
//        numTotal.add(1L)
//        if (row(1) == row(2))
//          numCorrect.add(1L)
//      }
//    })
//
//    val accuracy = numCorrect.value.toDouble / numTotal.value.toDouble
//    logger.info("Accuracy: " + accuracy.toString)
//
//    results.write.option("header", "true").csv(args(1))
  }
}