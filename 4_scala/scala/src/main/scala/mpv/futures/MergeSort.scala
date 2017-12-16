package mpv.futures

import scala.annotation.tailrec
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.duration.Duration
import scala.concurrent.{Await, Future}
import scala.math.Ordering
import scala.util.Random

/**
  * @author Thomas Herzog <herzog.thomas81@gmail.com>
  * @since 12/6/17
  */
object MergeSort extends App {

  def random: Random = new Random()

  val LIST_SIZE: Int = 20000
  val UPPER_BORDER: Int = LIST_SIZE * 10
  val RUNS = 2

  def printMeasure(s: String, ms: Long): Unit = {
    val formatter = java.text.NumberFormat.getIntegerInstance
    println("%.16s ... = %s ms".format(s, formatter.format(ms)))
  }

  def runWithTimerSeq[T](runs: Int)(block: => T): Long = {

    def measure(run: Int, block: => T): Long = {
      val start: Long = System.currentTimeMillis()
      val result = block
      val stop: Long = System.currentTimeMillis()
      val duration = stop - start
      printMeasure(s"$result", duration)

      duration
    }

    (1 to runs).map(i => measure(i, block)).sum / runs
  }

  def runWithTimerPar[T <: Future[_]](runs: Int)(block: => T): Long = {

    def measure(run: Int, block: => T): Long = {
      val start: Long = System.currentTimeMillis()
      val future = block
      val result = Await.result(future, Duration.Inf)
      val duration = System.currentTimeMillis() - start
      printMeasure(s"$result", duration)

      duration
    }

    (1 to runs).map(i => measure(i, block)).sum / runs
  }

  /**
    * Breaks down the list for the sorted merging and returns sorted result.
    *
    * @param list     the list to merge
    * @param ordering the implicit or explicit provided ordering
    * @tparam T the type of the list elements
    * @return the sorted list, depending on the ordering
    */
  def mergeSortSeq[T](list: List[T])(implicit ordering: Ordering[T]): List[T] = {
    val n = list.size / 2
    if (n == 0) {
      list
    }
    else {
      merge(mergeSortSeq(list take n)(ordering),
        mergeSortSeq(list drop n)(ordering))(ordering)
    }
  }

  /**
    * Breaks down the list for the sorted merging and returns sorted result.
    *
    * @param list     the list to sort
    * @param ordering the implicit or explicit ordering
    * @tparam T the type fo the list elements
    * @return the future holding the sorted list
    */
  def mergeSortPar[T](list: List[T], maxSize: Int = 600)(implicit ordering: Ordering[T]): Future[List[T]] = {
    val n = list.size / 2
    if (n == 0) {
      Future {
        list
      }
    } else if (list.lengthCompare(maxSize) < 0) {
      Future {
        merge(mergeSortSeq(list take n)(ordering),
          mergeSortSeq(list drop n)(ordering))(ordering)
      }
    }
    else {
      for (left <- mergeSortPar(list take n)(ordering);
           right <- mergeSortPar(list drop n)(ordering)) yield {
        merge(left, right)
      }
    }
  }

  def merge[T](x: List[T], y: List[T])(implicit ord: Ordering[T]): List[T] = {
    /** *
      * recursive merge
      *
      * @param x   the left hand list
      * @param y   the right hand list
      * @param acc used to combine the result, so that the recMerge is tail recursive
      * @return the sorted list depending on the provided ordering
      */
    @tailrec
    def recMerge(x: List[T], y: List[T], acc: List[T]): List[T] = {
      (x, y) match {
        case (Nil, Nil) => acc // Empty lists means accumulator is the result
        case (_, Nil) => acc ++ x // Only left list present, the merge to the right with accumulator
        case (Nil, _) => acc ++ y // Only right list present, the merge to the right with accumulator
        case (xh :: xt, yh :: yt) => // Pop the first element of the two lists
          if (ord.lteq(xh, yh)) // Compare them with lower equals
            recMerge(xt, y, acc :+ xh) // Call with new left list, popped element is in accumulator now
          else
            recMerge(x, yt, acc :+ yh) // Call with new right list, popped element is in accumulator now
      }
    }

    recMerge(x, y, List.empty) // First recursive call with empty accumulator list, which will represent the merged sorted list
  }

  def test_merge_explicit(): Unit = {
    println("Start: test_merge_explicit")

    def compare(x: Int, y: Int) = {
      //print(s"explicit compare: x: $x | y: $y, ")
      java.lang.Integer.compare(x, y)
    }

    val seqNumbers = for (_ <- 1 to LIST_SIZE) yield {
      random.nextInt(UPPER_BORDER)
    }

    val result = mergeSortSeq(seqNumbers.toList) {
      compare
    }

    println("%.16s".format(result))

    println("Start: test_merge_explicit")
  }

  def test_merge_custom_implicit(): Unit = {
    println("Start: test_merge_custom_implicit")

    implicit def reverse: Ordering[Int] = (x: Int, y: Int) => {
      //print(s"implicit compare: x: $x | y: $y, ")
      java.lang.Integer.compare(x, y)
    }

    val seqNumbers = for (_ <- 1 to LIST_SIZE) yield {
      random.nextInt(UPPER_BORDER)
    }

    val result = mergeSortSeq(seqNumbers.toList)

    println("%.16s".format(result))

    println("Start: test_merge_custom_implicit")
  }

  def test_merge_seq_timer(): Unit = {
    println("Start: test_merge_seq_timer")

    val seqNumbers = for (_ <- 1 to LIST_SIZE) yield {
      random.nextInt(UPPER_BORDER)
    }

    runWithTimerSeq(RUNS) {
      mergeSortSeq(seqNumbers.toList)
    }

    println("Start: test_merge_seq_timer")
  }

  def test_merge_par_timer(): Unit = {
    println("Start: test_merge_par_timer")

    val seqNumbers = for (_ <- 1 to LIST_SIZE) yield {
      random.nextInt(UPPER_BORDER)
    }

    runWithTimerPar(RUNS) {
      mergeSortPar(seqNumbers.toList, 500)
    }

    println("Start: test_merge_par_timer")
  }


  println("===> Starting tests <===")
  test_merge_explicit()
  println("========================")
  test_merge_custom_implicit()
  println("========================")
  test_merge_seq_timer()
  println("========================")
  test_merge_par_timer()
  println("===> End tests      <===")

  Thread.sleep(1000)
}
