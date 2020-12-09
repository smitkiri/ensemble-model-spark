package nb

import breeze.linalg._
import breeze.numerics._
import breeze.stats
import scala.collection.mutable

class NaiveBayesClassifier() extends Serializable {
  var classes: Set[Int] = Set()
  var statistics: mutable.HashMap[Int, mutable.HashMap[String, DenseVector[Double]]] = mutable.HashMap()
  var priors: mutable.HashMap[Int, Float] = mutable.HashMap()

  def getPriors(target: Array[Int]): mutable.HashMap[Int, Float] = {
    // Total examples
    val no_of_examples = target.length
    val attrs = target.groupBy(identity).mapValues(_.length)
    val probability = mutable.HashMap[Int, Float]()

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

      val avg = stats.mean(subset, Axis._0).t
      val std = stats.stddev(subset, Axis._0).t

      var classStats = mutable.HashMap[String, DenseVector[Double]]()
      classStats += "mean" -> avg
      classStats += "std" -> std

      this.statistics(c) = classStats
    }

    this
  }

  def calculateProbability(features: DenseVector[Double], featureMean: DenseVector[Double],
                            featureStd: DenseVector[Double]): Double = {

    val exponentVal = pow(features - featureMean, 2) / (pow(featureStd, 2) * 2.0)
    val numerator = exp(-1.0 * exponentVal)
    val denominator = sqrt(2.0 * math.Pi * featureStd)

    val probabilities = log(numerator / denominator)

    probabilities.toArray.sum
  }

  def predict(test: List[Array[Double]]): Array[Int] = {
    var predictions = Array[Int]()

    for (ex <- test) {
      val classProb = mutable.HashMap[Int, Double]()

      for (c <- this.classes) {
        val classStats = this.statistics(c)
        val classMean = classStats("mean")
        val classStd = classStats("std")

        val posterior = calculateProbability(DenseVector(ex), classMean, classStd)
        val prob = math.log(this.priors(c)) + posterior
        classProb(c) = prob
      }

      val pred = classProb.max._1
      predictions = predictions :+ pred
    }

    predictions
  }

}
