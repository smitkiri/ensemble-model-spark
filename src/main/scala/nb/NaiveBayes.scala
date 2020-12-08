package nb

import org.apache.spark.{HashPartitioner, SparkConf, SparkContext, sql}
import org.apache.log4j.LogManager
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

class NaiveBayesClassifier() {
  def fit(train : List[Array[Any]]): Unit =
  {
    "Hi!"

  }

}
