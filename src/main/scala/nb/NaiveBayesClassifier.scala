package nb

import org.apache.spark.{HashPartitioner, SparkConf, SparkContext, sql}
import org.apache.log4j.LogManager
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import breeze.linalg._
import breeze.numerics._

import scala.collection.mutable

class NaiveBayesClassifier() {

  def getPriors(target: Array[Int]): mutable.HashMap[Int, Float] = {
    // Total examples
    val no_of_examples = target.length
    val attrs = target.groupBy(identity).mapValues(_.length)
    var probability = mutable.HashMap[Int, Float]()

    for ((c, count) <- attrs) {
      probability(c) = count / no_of_examples.toFloat
    }

    probability
  }

  def fit(train : List[Array[Double]]): Unit = {
    // Convert data to a DenseMatrix
    val trainMatrix: DenseMatrix[Double] = DenseMatrix(train:_*)

    // Split the feature and target columns
    val yTrain: Array[Int] = trainMatrix(::, 0).mapValues(_.toInt).toArray
    val xTrain: DenseMatrix[Double] = trainMatrix.delete(0, Axis._1)

    val priors = getPriors(yTrain)
    val classes = yTrain.toSet

  }


}
