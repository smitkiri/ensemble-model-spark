package rf

import org.apache.spark.{HashPartitioner, SparkConf, SparkContext, sql}
import org.apache.log4j.LogManager
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.feature.{VectorAssembler, VectorSlicer}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._


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
    val numTrees = 5

    var dataFrame = ss.read.format("csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load("input/sample_train.csv")

    dataFrame = dataFrame
      .withColumn("# label", dataFrame.col("# label")
        .cast(sql.types.IntegerType))

    // Verify column names
    logger.info("Data Columns:")
    val columns = dataFrame.columns
    for (col <- columns) {
      logger.info(col)
    }

    //Verify data types
    logger.info("Column data types:")
    val dataTypes = dataFrame.schema.fields.map(x=>x.dataType).map(x=>x.toString)
    for (data <- dataTypes) {
      logger.info(data)
    }

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

    val broadcastTrain = sc.broadcast(trainingData)
    val broadcastTest = sc.broadcast(testDataWithId)

    // Decision tree pipeline
    val dt = new DecisionTreeClassifier()
      .setLabelCol("# label")
      .setFeaturesCol("sampledFeatures")

    val trainingPipeline = new Pipeline()
      .setStages(Array(dt))

    // RDD of multiple decision tree pipelines
    var treeArray = Array[(Int, Pipeline)]()
    for (i <- 0 until numTrees) {
      treeArray = treeArray :+ (i, trainingPipeline)
    }
    val treeRDD: RDD[(Int, Pipeline)] = sc.parallelize(treeArray)
      .partitionBy(new HashPartitioner(math.ceil(numTrees.toDouble / 10.0).toInt))

    // Train all decision trees
    val models = treeRDD.map {case (idx, tree)  =>
      // Create a bootstrap sample for decision tree
      var bootstrapSample = broadcastTrain.value
        .sample(withReplacement = true, fraction = 1)

      // Get a random list of features for decision tree
      val r = new scala.util.Random
      r.setSeed(idx)
      var randomFeatures = Array[Int]()
      for (_ <- 0 until scala.math.sqrt(features.length).toInt) {
        randomFeatures = randomFeatures :+ r.nextInt(features.length)
      }

      val slicer = new VectorSlicer()
        .setInputCol("features")
        .setOutputCol("sampledFeatures")

      slicer.setIndices(randomFeatures)
      bootstrapSample = slicer.transform(bootstrapSample)

      (idx, (tree.fit(bootstrapSample), randomFeatures))
    }

    // Get predictions from all decision trees
    val dtPredictions = models.flatMap {
      case (_, (tree, randomFeatures)) =>
        val slicer = new VectorSlicer()
          .setInputCol("features")
          .setOutputCol("sampledFeatures")

        slicer.setIndices(randomFeatures)
        val testData = slicer.transform(broadcastTest.value)
        val pred = tree.transform(testData)
        pred.select("Id", "prediction").rdd
          .map(row => (row(0).asInstanceOf[Long], row(1).asInstanceOf[Double]))
          .collect()
    }

    // Set the majority vote as final prediction
    val allPredictions = dtPredictions
      .reduceByKey(_.toInt + _.toInt)
      .mapValues(x => if (x < numTrees.toDouble / 2) 0 else 1)

    // Combine with original data for comparison
    val results = allPredictions.toDF("Id", "prediction").as("preds")
      .join(broadcastTest.value.as("test"), col("preds.Id") === col("test.Id"))
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