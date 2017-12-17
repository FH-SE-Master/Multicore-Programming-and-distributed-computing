package mpv.futures

import scala.annotation.tailrec
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.duration.Duration
import scala.concurrent.{Await, Future, Promise}
import scala.util.{Failure, Random, Success}

/**
  * @author Thomas Herzog <herzog.thomas81@gmail.com>
  * @since 12/16/17
  */
object FutureAdvanced extends App {

  def random: Random = new Random()

  def createMaxFuture(list: List[Int], startIdx: Int, endIdx: Int): Future[Int] = {
    Future {
      list.slice(startIdx, endIdx).max
    }
  }

  def parallelMax1(list: List[Int], parts: Int): Future[Int] = {
    if (list == null || list.isEmpty || list.lengthCompare(parts) < 0) {
      Future {
        throw new IllegalArgumentException("List must not be null, or empty and parts < list.length")
      }
    } else {
      val size: Int = list.length / parts
      val sequence: Seq[Future[Int]] = (0 until parts).map(i => createMaxFuture(list, i * size, (i * size) + size))
      if (list.length % parts != 0) {
        sequence ++ Seq(createMaxFuture(list, size * parts, list.length))
      }

      for (result <- Future.sequence(sequence)) yield {
        result.max
      }
    }
  }

  def parallelMax2(list: List[Int], parts: Int): Future[Int] = {
    if (list == null || list.isEmpty || list.lengthCompare(parts) < 0) {
      Future {
        throw new IllegalArgumentException("List must not be null, or empty and parts < list.length")
      }
    } else {
      val size: Int = list.length / parts
      val sequence: List[Future[Int]] = (0 until parts).map(i => createMaxFuture(list, i * size, (i * size) + size)).toList
      if (list.length % parts != 0) {
        sequence ++ Seq(createMaxFuture(list, size * parts, list.length))
      }

      for (result <- futureSequence(sequence)) yield {
        result.max
      }
    }
  }

  def futureSequence(list: List[Future[Int]]): Future[List[Int]] = {
    @tailrec
    def futureSequenceRec(list: List[Future[Int]], acc: List[Int]): Future[List[Int]] = {
      if (list.lengthCompare(1) == 0) {
        Future {
          acc
        }
      } else {
        futureSequenceRec(list drop 1, acc :+ Await.result((list take 1).head, Duration.Inf))
      }
    }

    futureSequenceRec(list, List.empty[Int])
  }

  def doCompetitively[T](futures: List[Future[T]]): Future[T] = {
    val promise: Promise[T] = Promise[T]()
    futures.foreach(x => x onComplete (_ match {
      case Success(data) => if (promise.trySuccess(data)) {
        println(s"Success: Complete promise with: '$data'")
      };
      case Failure(ex) => println(s"Failure: '$ex'")
    }))
    promise.future
  }


  //region Test functions
  def test_parallelMax1_success(): Unit = {
    println("Start: test_parallelMax1_success")

    val seqNumbers = for (_ <- 1 to 10) yield {
      random.nextInt(1000)
    }

    println(s"List: $seqNumbers")
    println(s"Seq.max - Max:  ${seqNumbers.max}")
    val resultFuture = parallelMax1(seqNumbers.toList, 10)
    resultFuture onComplete {
      case Success(v) => println(s"Success - Max:  $v")
      case Failure(e) => println(s"Failure - Error: $e")
    }

    Await.result(resultFuture, Duration.Inf)

    println("End: test_parallelMax1_success")
  }

  def test_parallelMax1_error_part(): Unit = {
    println("Start: test_parallelMax1_error_part")
    val seqNumbers = for (_ <- 1 to 10) yield {
      random.nextInt(1000)
    }

    println(s"List: $seqNumbers")
    println(s"Seq.max - Max:  ${seqNumbers.max}")
    try {
      val resultFuture = parallelMax1(seqNumbers.toList, seqNumbers.length + 1)
      resultFuture onComplete {
        case Success(v) => println(s"Success - Max:  $v")
        case Failure(e) => println(s"Failure - Error: $e")
      }

      Await.result(resultFuture, Duration.Inf)
    } catch {
      case _ => println("Exception occurred in Future")
    }

    println("End: test_parallelMax1_error_part")
  }

  def test_parallelMax1_error_empty(): Unit = {
    println("Start: test_parallelMax1_error_empty")

    println(s"List: ${Seq.empty[Int]}")
    println(s"Seq.max - Max:  0")
    try {
      val resultFuture = parallelMax1(Seq.empty[Int].toList, 0)
      resultFuture onComplete {
        case Success(v) => println(s"Success - Max:  $v")
        case Failure(e) => println(s"Failure - Error: $e")
      }
      Await.result(resultFuture, Duration.Inf)
    } catch {
      case _ => println("Exception occurred in Future")
    }

    println("End: test_parallelMax1_error_empty")
  }

  def test_parallelMax2_success(): Unit = {
    println("Start: test_parallelMax2_success")

    val seqNumbers = for (_ <- 1 to 10) yield {
      random.nextInt(1000)
    }

    println(s"List: $seqNumbers")
    println(s"Seq.max - Max:  ${seqNumbers.max}")
    val resultFuture = parallelMax2(seqNumbers.toList, 10)
    resultFuture onComplete {
      case Success(v) => println(s"Success - Max:  $v")
      case Failure(e) => println(s"Failure - Error: $e")
    }

    Await.result(resultFuture, Duration.Inf)

    println("End: test_parallelMax2_success")
  }

  def test_parallelMax2_error_part(): Unit = {
    println("Start: test_parallelMax2_error_part")

    val seqNumbers = for (_ <- 1 to 10) yield {
      random.nextInt(1000)
    }

    println(s"List: $seqNumbers")
    println(s"Seq.max - Max:  ${seqNumbers.max}")
    try {
      val resultFuture = parallelMax2(seqNumbers.toList, seqNumbers.length + 1)
      resultFuture onComplete {
        case Success(v) => println(s"Success - Max:  $v")
        case Failure(e) => println(s"Failure - Error: $e")
      }

      Await.result(resultFuture, Duration.Inf)
    } catch {
      case _ => println("Exception occurred in Future")
    }

    println("End: test_parallelMax2_error_part")
  }

  def test_parallelMax2_error_empty(): Unit = {
    println("Start: test_parallelMax2_error_empty")

    println(s"List: ${Seq.empty[Int]}")
    println(s"Seq.max - Max:  0")
    try {
      val resultFuture = parallelMax2(Seq.empty[Int].toList, 0)
      resultFuture onComplete {
        case Success(v) => println(s"Success - Max:  $v")
        case Failure(e) => println(s"Failure - Error: $e")
      }
      Await.result(resultFuture, Duration.Inf)
    } catch {
      case _ => println("Exception occurred in Future")
    }

    println("Start: test_parallelMax2_error_empty")
  }

  def test_doCompetitively_success(): Unit = {
    println("Start: test_doCompetitively_success")

    for (i <- (1 to 10)) {
      println(s"Run $i:")
      val futures: List[Future[Int]] = (1 to 10).map(i => Future {
        Thread.sleep(1000 + random.nextInt(200))
        i
      }).toList

      try {
        val resultFuture = doCompetitively(futures)
        resultFuture onComplete {
          case Success(v) => println(s"Success - $v")
          case Failure(e) => println(s"Failure - Error: $e")
        }
        Await.result(resultFuture, Duration.Inf)
      } catch {
        case _ => println("Exception occurred in Future")
      }
    }

    println("Start: test_doCompetitively_success")
  }

  //endregion

  println("===> Starting tests <===")
  test_parallelMax1_success()
  println("========================")
  test_parallelMax1_error_part()
  println("========================")
  test_parallelMax1_error_empty()
  println("========================")
  test_parallelMax2_success()
  println("========================")
  test_parallelMax2_error_part()
  println("========================")
  test_parallelMax2_error_empty()
  println("========================")
  test_doCompetitively_success()
  println("===> End tests      <===")
}
