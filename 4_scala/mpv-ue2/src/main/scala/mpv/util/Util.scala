package mpv.util

import com.typesafe.config.{Config, ConfigFactory, ConfigValueFactory}

object RangeUtil {

  private val EPS = 0.5

  def splitIntoIntervals(lower: Int, upper: Int, n: Int) : Seq[(Int, Int)] = {
    val d = math.max((upper - lower) / n.toDouble, 1)
    val steps = lower.toDouble to (upper.toDouble-EPS) by d

    val s = steps.size
    val intervals =
      for (i <- 0 to s-2)
        yield (((steps(i)+EPS).toInt, (steps(i+1)+EPS).toInt-1))

    intervals :+ ((steps(s-1)+EPS).toInt, upper)
  }
}

object LoggingUtil {

  def loggingLevel(level: String): Config = {
    ConfigFactory.load()
      .withValue("akka.loglevel", ConfigValueFactory.fromAnyRef(level))
      .withValue("akka.stdout-loglevel", ConfigValueFactory.fromAnyRef(level))
  }
}

