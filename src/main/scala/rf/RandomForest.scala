package rf

import org.apache.log4j.LogManager
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier}
import org.apache.spark.ml.feature.{VectorAssembler, VectorSlicer}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.{HashPartitioner, SparkConf, SparkContext, sql}


object RandomForest {

  def main(args: Array[String]) {
    val logger: org.apache.log4j.Logger = LogManager.getRootLogger
    if (args.length != 2) {
      logger.error("Usage:\nwc.RandomForestMain <input dir> <output dir>")
      System.exit(1)
    }
    val conf = new SparkConf().setAppName("Random Forest Classification")
    val sc = new SparkContext(conf)

    val ss = SparkSession.builder().getOrCreate()
    import ss.implicits._

    // Number of trees in RandomForest
    val numTrees = args(2).toInt

    var dataFrame = ss.read.format("csv")
                      .option("header", "true")
                      .option("inferSchema", "true")
                      .load(args(0))

    dataFrame = dataFrame
      .withColumn("# label", dataFrame.col("# label")
        .cast(sql.types.IntegerType))

    // Verify column names
    val columns = dataFrame.columns

    // Get list of feature columns, first column is label
    val features = columns.slice(1, columns.length)

    // Convert multiple feature columns to single vector column
    val assembler = new VectorAssembler()
      .setInputCols(features)
      .setOutputCol("features")

    val stages = Array(assembler)
    val preprocessPipeline = new Pipeline()
      .setStages(stages)

    dataFrame = preprocessPipeline.fit(dataFrame).transform(dataFrame)
    dataFrame = dataFrame.drop(features:_*)

    // Split and broadcast training and testing data
    val Array(trainingData, testData) = dataFrame.randomSplit(Array(0.7, 0.3))
    val testDataWithId = testData.withColumn("Id", monotonically_increasing_id())
    var models = sc.emptyRDD[(Int, (PipelineModel, Array[Int]))].partitionBy(new HashPartitioner(10))

    for (itr <- 0 until numTrees) {
      // Decision tree pipeline
      val dt = new DecisionTreeClassifier()
        .setLabelCol("# label")
        .setFeaturesCol("sampledFeatures")

      val trainingPipeline = new Pipeline()
        .setStages(Array(dt))

      var bootstrapSample = trainingData
        .sample(withReplacement = true, fraction = 1)

      // Get a random list of features for decision tree
      val r = new scala.util.Random
      r.setSeed(itr)
      var randomFeatures = Array[Int]()
      for (_ <- 0 until scala.math.sqrt(features.length).toInt) {
        randomFeatures = randomFeatures :+ r.nextInt(features.length)
      }

      val slicer = new VectorSlicer()
        .setInputCol("features")
        .setOutputCol("sampledFeatures")

      slicer.setIndices(randomFeatures)
      bootstrapSample = slicer.transform(bootstrapSample)
      val dtModel = trainingPipeline.fit(bootstrapSample)

      models = models.union(sc.parallelize(Seq((itr, (dtModel, randomFeatures)))))
    }

    var dtPredictions = sc.emptyRDD[(Long, Double)].partitionBy(new HashPartitioner(10))
    for (itr <- 0 until numTrees) {
      val slicer = new VectorSlicer()
        .setInputCol("features")
        .setOutputCol("sampledFeatures")

      val model = models.lookup(itr)
      val tree = model.head._1
      val randomFeatures = model.head._2

      slicer.setIndices(randomFeatures)
      val testData = slicer.transform(testDataWithId)
      val pred = tree.transform(testData)
      val tempPreds = pred.select("Id", "prediction").rdd
        .map(row => (row(0).asInstanceOf[Long], row(1).asInstanceOf[Double]))

      dtPredictions = dtPredictions.union(tempPreds)
    }

    // Set the majority vote as final prediction
    val allPredictions = dtPredictions
      .reduceByKey(_.toInt + _.toInt)
      .mapValues(x => if (x < numTrees.toDouble / 2) 0 else 1)

    // Combine with original data for comparison
    val results = allPredictions.toDF("Id", "prediction").as("preds")
      .join(testDataWithId.as("test"), col("preds.Id") === col("test.Id"))
      .select("test.Id", "preds.prediction", "test.# label")
      .repartitionByRange(10, col("Id"))
      .sort("Id")

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