import java.util

import scala.concurrent.{Await, Future}
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.duration.Duration
import scala.util.{Failure, Random, Success}

/**
  * @author Thomas Herzog <herzog.thomas81@gmail.com>
  * @since 11/29/17
  */
object FuturesExamples extends App {

  val random: Random = new Random()

  def doWork(id: String): Unit = {
    Thread.sleep(random.nextInt(1000) + 1)
    println(s"#doWork#Thread#id: ${Thread.currentThread().getId} | Id: $id")
  }

  def doError(id: String): Unit = {
    Thread.sleep(random.nextInt(1000) + 1)
    throw new NumberFormatException("Number cannot be formatted")
  }

  def doWorkWithResult(id: String): String = {
    Thread.sleep(random.nextInt(1000) + 1)
    println(s"doWorkWithResult#Thread#id: ${Thread.currentThread().getId} | Id: $id")

    id
  }

  def doProduceNumber(id: String): Unit = {
    Thread.sleep(random.nextInt(1000) + 1)
    println(s"doWorkWithResult#Thread#id: ${Thread.currentThread().getId} | Id: $id")

    random.nextInt()
  }

  /**
    * Executes the two given tasks in parallel
    *
    * @param task1 the first task
    * @param task2 the secnd task
    */
  def doInParallel(task1: String => Unit, task2: String => Unit): Future[Unit] = {
    println("Starting: task1")
    val f1: Future[Unit] = Future {
      task1("task1")
    }
    println("Starting: task2")
    val f2: Future[Unit] = Future {
      task2("task2")
    }
    println("Waiting: task1, task2")
    for (_ <- f1; _ <- f2) yield {}
  }

  def doInParallelOverloadFor[U, V](f1: => Future[U], f2: => Future[V]): Future[(U, V)] ={
    for(u <- f1; v <- f2) yield {(u,v)}
  }

  def doInParallelOverloadFlatMap[U, V](f1: => Future[U], f2: => Future[V]): Future[(U, V)] ={
    f1 flatMap(x => f2.map(y => {(x,y)}))
  }

  def doInParallelOverloadFinder[A](col1: Iterable[A], col2: Iterable[A], finder: Iterable[A] => A): Future[(A,A)] ={
    doInParallelOverloadFor(Future{finder(col1)}, Future{finder(col2)})
  }

  def test_doInParallel_success(): Unit = {
    println("Start: test_doInParallel_success")

    try {
      val result = doInParallel(doWork, doWork)
      result onComplete {
        case Success(v) => println(s"Completed: SUCCESSFUL: $v")
        case Failure(v) => println(s"Completed: FAILURE:    $v")
      }
      Await.result(result, Duration.Inf)
      Thread.sleep(100)
    } catch {
      case t: Throwable => println(s"Exception was thrown: $t")
    }

    println("End: test_doInParallel_success")
  }

  def test_doInParallel_error(): Unit = {
    println("Start: test_error")

    try {
      val result = doInParallel(doWork, doError)
      result onComplete {
        case Success(v) => println(s"Completed: SUCCESSFUL: $v")
        case Failure(v) => println(s"Completed: FAILURE:    $v")
      }
      Await.result(result, Duration.Inf)
      Thread.sleep(100)
    } catch {
      case t: Throwable => println(s"Exception was thrown: $t")
    }

    println("End: test_error")
  }

  def test_doInParallelOverload_for_success(): Unit = {
    println("Start: test_doInParallelOverload_for_success")

    try {
      val result = doInParallelOverloadFor(Future{doWorkWithResult("f1")}, Future{doWorkWithResult("f2")})
      result onComplete {
        case Success(v) => println(s"Completed: SUCCESSFUL: $v")
        case Failure(v) => println(s"Completed: FAILURE:    $v")
      }
      Await.result(result, Duration.Inf)
      Thread.sleep(100)
    } catch {
      case t: Throwable => println(s"Exception was thrown: $t")
    }

    println("End: test_doInParallelOverload_for_success")
  }

  def test_doInParallelOverload_flat_map_success(): Unit = {
    println("Start: test_doInParallelOverload_flat_map_success")

    try {
      val result = doInParallelOverloadFlatMap(Future{doWorkWithResult("f1")}, Future{doWorkWithResult("f2")})
      result onComplete {
        case Success(v) => println(s"Completed: SUCCESSFUL: $v")
        case Failure(v) => println(s"Completed: FAILURE:    $v")
      }
      Await.result(result, Duration.Inf)
      Thread.sleep(100)
    } catch {
      case t: Throwable => println(s"Exception was thrown: $t")
    }

    println("End: test_doInParallelOverload_flat_map_success")
  }

  def test_doInParallelFindMaximum_success(): Unit ={
    println("Start: test_doInParallelFindMaximum_success")

    try {
      val seqNumbers = for(_ <- 1 to 10) yield { random.nextInt() }
      val parts = seqNumbers.splitAt(5)
      val result = doInParallelOverloadFinder(parts._1, parts._2, (iterable: Iterable[Int]) => {iterable.max})
      result onComplete {
        case Success(t) => {
          if(t._1 > t._2) {
            println(s"Maximum found: ${t._1}")
          }else{
            println(s"Maximum found: ${t._2}")
          }
        }
        case Failure(e) => println(s"Completed: Error: $e ")
      }

      Await.result(result, Duration.Inf)
      Thread.sleep(100)
    } catch {
      case t: Throwable => println(s"Exception was thrown: $t")
    }

    println("End: test_doInParallelFindMaximum_success")
  }

  test_doInParallelFindMaximum_success()
  /**
  println("===> Starting tests <===")
  test_doInParallel_success()
  println("========================")
  test_doInParallel_error()
  println("========================")
  test_doInParallelOverload_for_success()
  println("========================")
  test_doInParallelOverload_flat_map_success()
  println("===> End tests      <===")
    */
}
