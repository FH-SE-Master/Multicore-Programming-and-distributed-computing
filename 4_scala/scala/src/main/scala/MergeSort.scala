import scala.annotation.tailrec

/**
  * @author Thomas Herzog <herzog.thomas81@gmail.com>
  * @since 12/6/17
  */
object MergeSort extends App {

  def merge[T](x: List[T], y: List[T])(implicit ord: Ordering[T]): List[T] = {
    /***
      * recursive merge
      * @param x
      * @param y
      * @param acc used to combine the result, so that the recMerge is tail recursive
      * @return
      */
    @tailrec
    def recMerge(x: List[T], y: List[T], acc: List[T]): List[T] = {
      (x, y) match {
        case (Nil, Nil) => acc
        case (_, Nil) => acc ++ x
        case (Nil, _) => acc ++ y
        case (xh :: xt, yh :: yt) =>
          if (ord.lteq(xh, yh))
            recMerge(xt, y, acc :+ xh)
          else
            recMerge(x, yt, acc :+ yh)
      }
    }
    recMerge(x, y, List.empty)
  }


}
