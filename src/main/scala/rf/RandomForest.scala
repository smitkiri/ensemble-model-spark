package rf

import org.apache.spark.{HashPartitioner, RangePartitioner, SparkConf, SparkContext, sql}
import org.apache.log4j.LogManager
import org.apache.log4j.Level
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.feature.{VectorAssembler, VectorIndexer}
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.dsl.expressions.{DslExpression, StringToAttributeConversionHelper}
import org.apache.spark.sql.{DataFrame, SparkSession}

object RandomForestMain {

  def getDummyArray(num: Int, data: DataFrame): Array[(Int, DataFrame)] = {
    var dummy = Array[(Int, DataFrame)]()
    for (i <- 0 to num) {
      dummy = dummy :+ (i, data.sample(withReplacement = true, fraction = 1))
    }
    dummy
  }

  def getBootstrapSamples(sc: SparkContext, data: DataFrame, numSamples: Int,
                          numPartitions: Int): RDD[(Int, DataFrame)] = {

    sc.parallelize(getDummyArray(numSamples, data))
      .partitionBy(new HashPartitioner(numPartitions))

  }

  def decisionTree(data: DataFrame) = {

  }
  
  def main(args: Array[String]) {
    val logger: org.apache.log4j.Logger = LogManager.getRootLogger
    if (args.length != 2) {
      logger.error("Usage:\nwc.RandomForestMain <input dir> <output dir>")
      System.exit(1)
    }
    val conf = new SparkConf().setAppName("Random Forest Classification")
    val sc = new SparkContext(conf)

    val ss = SparkSession.builder().getOrCreate()

    var dataFrame = ss.read.format("csv")
                      .option("header", "true")
                      .option("inferSchema", "true")
                      .load("input/sample_train.csv")

    dataFrame = dataFrame.withColumn("# label", dataFrame.col("# label").cast(sql.types.IntegerType))
    val columns = dataFrame.columns
    for (col <- columns) {
      logger.info(col)
    }
    val features = columns.slice(1, columns.length)

    //Verify data types
    val dataTypes = dataFrame.schema.fields.map(x=>x.dataType).map(x=>x.toString)
    for (data <- dataTypes) {
      logger.info(data)
    }

    // Convert features to required format
    val assembler = new VectorAssembler()
      .setInputCols(features)
      .setOutputCol("features")

    val stages = Array(assembler)
    val pipeline = new Pipeline()
      .setStages(stages)

    dataFrame = pipeline.fit(dataFrame).transform(dataFrame)
    dataFrame = dataFrame.drop(features:_*)

    val Array(trainingData, testData) = dataFrame.randomSplit(Array(0.7, 0.3))

    val numTrees = 10
    val bootStrapSamples: RDD[(Int, DataFrame)] = getBootstrapSamples(sc, trainingData, numTrees, 5)

  }
}