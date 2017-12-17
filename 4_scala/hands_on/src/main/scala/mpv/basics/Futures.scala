package mpv.basics

import scala.concurrent.{Await, Future}
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.duration.Duration
import scala.util.{Failure, Success}

/**
  * @author Thomas Herzog <herzog.thomas81@gmail.com>
  * @since 11/24/17
  */
object Futures extends App {

  def doWork(id: Int, steps: Int): Unit = {
    for (i <- 1 to steps) {
      println(s"$id: $i (thread id=${Thread.currentThread.getId})")
      if (i == 6) throw new IllegalArgumentException()
      Thread.sleep(200)
    }
  }

  def compute(id: Int, n: Int, result: Int): Int = {
    for (i <- 1 to n) {
      println(s"compute $id: $i (thread id=${Thread.currentThread.getId})")
      Thread.sleep(200)
    }
    result
  }

  def combine(value1: Int, value2: Int): Int = {
    for (i <- 1 to 5) {
      println(s"combine $i (thread id=${Thread.currentThread.getId})")
      Thread.sleep(200)
    }
    value1 + value2
  }

  def sequentialInvocation(): Unit = {
    doWork(1, 5)
    doWork(2, 5)
  }

  def simpleFutures(): Unit = {
    // We need to import the implicit execution context otherwise execution of futures will fail.
    // import scala.concurrent.ExecutionContext.Implicits.global

    // Futures will not block an return immediately
    val f1 = Future {
      doWork(1, 5)
    }

    val f2 = Future {
      doWork(2, 5)
    }

    // Just for testing, no examination relevant
    Await.ready(f1, Duration.Inf)
    Await.ready(f2, Duration.Inf)
  }

  def futuresWithCallback(): Unit = {
    val f1 = Future {
      doWork(1, 5)
    }

    val f2 = Future {
      doWork(2, 5)
    }

    f1 foreach (_ => println("callback called: f1 SUCCESS"))
    f2.failed foreach (_ => println("callback called: f2 FAILURE"))

    f1 onComplete {
      case Success(v) => println("future s1 completed SUCCESSFUL")
      case Failure(v) => println("future s1 completed FAILURE")
    }

    f2 onComplete {
      case Success(v) => println("future s1 completed SUCCESSFUL")
      case Failure(v) => println("future s1 completed FAILURE")
    }

    // Just for testing, no examination relevant
    Await.ready(f1, Duration.Inf)
    Await.ready(f2, Duration.Inf)
    Thread.sleep(100)
  }

  def futureComposition(): Unit = {
    val f1: Future[Int] = Future {
      compute(1, 5, 30)
    }
    val f2: Future[Int] = Future {
      compute(2, 3, 12)
    }

    println(s"f1: $f1")
    println(s"f2: $f2")

    // val sum: Future[Int]  = f1 flatMap(r1 => f2 map(r2 => combine(r1, r2)))
    // sum foreach(s => println(s"sum: $s"))

    val sum = for(r1 <- f1;
                  r2 <- f2) yield combine(r1,r2)
    sum foreach (sum => println(s"sum: $sum"))

    // Just for testing, no examination relevant
    Await.ready(f1, Duration.Inf)
    Await.ready(f2, Duration.Inf)
    Thread.sleep(1500)
  }

  def sequenceFutures(): Unit = {
  }

  println(s"availableProcessors=${Runtime.getRuntime.availableProcessors}")
  println(s"Main Thread Id: ${Thread.currentThread.getId}")
  println("==== sequentialInvocation ====")
  // sequentialInvocation()

  println("\n==== simpleFutures ====")
  // simpleFutures()

  println("\n==== futuresWithCallback ====")
  // futuresWithCallback()

  println("\n==== futureComposition ====")
  futureComposition()
}
