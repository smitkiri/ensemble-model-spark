package nb

import breeze.linalg._
import breeze.numerics._
import breeze.stats._
import scala.collection.mutable

class NaiveBayesClassifier() extends Serializable {
  var classes: Set[Int] = Set()
  var statistics: mutable.HashMap[Int, mutable.HashMap[String, Array[Double]]] = mutable.HashMap()
  var priors: mutable.HashMap[Int, Float] = mutable.HashMap()

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

  def fit(train : List[Array[Double]]): NaiveBayesClassifier = {
    // Convert data to a DenseMatrix
    val trainMatrix: DenseMatrix[Double] = DenseMatrix(train:_*)

    // Split the feature and target columns
    val yTrainVector: DenseVector[Int] = trainMatrix(::, 0).mapValues(_.toInt)
    val yTrain: Array[Int] = yTrainVector.toArray

    val xTrainMatrix: DenseMatrix[Double] = trainMatrix.delete(0, Axis._1)

    this.priors = getPriors(yTrain)
    this.classes = yTrain.toSet

    for (c <- this.classes) {
      val sub_idx = yTrainVector.findAll(_ == c)
      val subset = xTrainMatrix(sub_idx, ::).toDenseMatrix

      val avg = mean(subset, Axis._0).t.toArray
      val std = stddev(subset, Axis._0).t.toArray

      var classStats = mutable.HashMap[String, Array[Double]]()
      classStats += "mean" -> avg
      classStats += "std" -> std

      this.statistics += c -> classStats
    }

    this
  }

  def calculate_probability(features: Array[Double], featureMean: Array[Double],
                            featureStd: Array[Double]): Double = {
    val featuresMatrix = DenseMatrix(features)
    val meanMatrix = DenseMatrix(featureMean)
    val varianceMatrix = DenseMatrix(featureStd.map(v => 2 * scala.math.pow(v, 2)))

    val exponentVal = pow(featuresMatrix - meanMatrix, 2) / varianceMatrix
    val exponent = exponentVal.toArray.map(v => scala.math.exp(-1 * v))
    val const = featureStd.map(v => v * scala.math.sqrt(2 * scala.math.Pi))

    val probabilities = exponent.zip(const).map {case (x, y) => x / y}

    probabilities.product
  }

}
