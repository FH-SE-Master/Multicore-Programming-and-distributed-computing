package mpv.basics

import scala.util.{Failure, Success, Try}

/**
  * @author Thomas Herzog <herzog.thomas81@gmail.com>
  * @since 11/24/17
  */
object Monads extends App {

  // Wrapper method for converting strings to numbers
  def toInt(str: String): Try[Int] = Try {
    Integer.parseInt(str)
  }

  def divide(a: Int, b: Int): Try[Double] = Try {
    if (a != 0 && b != 0) {
      return Try {
        a.toDouble / b.toDouble
      }
    } else {
      throw new ArithmeticException("Division by zero")
    }
  }

  println("=============: Monads")

  // Without monad exception will be thrown and program execution ends
  //  for (str <- Seq("0", "1", "x")) {
  //    Integer.parseInt(str)
  //  }

  // Try { ... } => Try.apply(() => {})
  val resultValid: Try[Int] = Try {
    Integer.parseInt("0")
  }
  val resultInvalid: Try[Int] = Try {
    Integer.parseInt("x")
  }

  println(s"resultValid: $resultValid")
  println(s"resultInvalid: $resultInvalid")

  println("=============: Generate Try values")
  for (str <- Seq("0", "1", "x")) {
    val result = toInt(str)
    println(s"result: $result")
  }

  // Only access valid values, invalid values are ignored
  println("=============: foreach on Try monad")
  for (str <- Seq("0", "1", "x")) {
    toInt(str) foreach (i => println(s"SUCCESS: $i"))
  }

  // If we want to access error as well, we need to use pattern matching
  println("=============: pattern matching on Try monad")
  for (str <- Seq("0", "1", "x")) {
    toInt(str) match {
      case Success(v) => println(s"SUCCESS result: $v")
      case Failure(v) => println(s"FAILURE result: $v")
    }
  }

  // map wraps in monad therefore, we cannot use methods which return mondas
  println("=============: Call map on Try monad")
  for (str <- Seq("5", "10", "0", "x")) {
    val q = toInt(str) map (value => divide(10, value))
    println(s"Quotient: $q")
  }

  // If we have a method which returns a monad, we need to use flatMap
  println("=============: Call flatMap on Try monad")
  for (str <- Seq("5", "10", "0", "x")) {
    val q = toInt(str) flatMap (value => divide(10, value))
    println(s"Quotient: $q")
  }

  // Complex usage of map and flatMap
  println("=============: Map and flatMap on Monad")
  for (str <- Seq(("6", "2"), ("6", "x"), ("6", "0"))) {
    val (s1, s2) = str
    val q = toInt(s1) flatMap (a => toInt(s2) flatMap (b => divide(a, b)))
    println(s"Quotient: $q")
  }

  // Complex usage of for (not a loop in this case)
  println("=============: For on Monad")
  for (str <- Seq(("6", "2"), ("6", "x"), ("6", "0"))) {
    val (s1, s2) = str
    val q = for (a <- toInt(s1);
                 b <- toInt(s2);
                 q <- divide(a,b)) yield q
    println(s"Quotient: $q")
  }
}
