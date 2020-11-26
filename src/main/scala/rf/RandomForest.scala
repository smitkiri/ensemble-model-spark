package rf

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.log4j.LogManager
import org.apache.log4j.Level

import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.mllib.util.MLUtils

object RandomForestMain {
  
  def main(args: Array[String]) {
    val logger: org.apache.log4j.Logger = LogManager.getRootLogger
    if (args.length != 2) {
      logger.error("Usage:\nwc.RandomForestMain <input dir> <output dir>")
      System.exit(1)
    }
    val conf = new SparkConf().setAppName("Random Forest Classification")
    val sc = new SparkContext(conf)

    val sqlContext= new org.apache.spark.sql.SQLContext(sc)
    import sqlContext.implicits._

    //val df = sqlContext.read.load("input/sample_train.csv", format="csv", header="true", inferSchema="true")

    val dataframe = sqlContext.read.format("csv")
                         .option("header", "true")
                         .option("inferSchema", "true")
                         .load("input/sample_train.csv")

  }
}