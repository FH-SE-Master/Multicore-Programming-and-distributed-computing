package mpv.basics

/**
  * @author Thomas Herzog <herzog.thomas81@gmail.com>
  * @since 11/24/17
  */
object FuncProg extends App {

  println("===========: Basics of functional programming")


  // val numbers: Seq[Int] = Seq(1,2,3)
  val numbers: Seq[Int] = 1 to 3 // same as above

  // Common as in java
  // numbers.foreach(i => println(i))

  // Infix notation
  // numbers foreach(i => println(i))

  // single argument 1st
  // numbers foreach { i => println(i) }

  // single argument 2nd
  // numbers foreach (println(_))

  // single argument 3nd
  numbers foreach println

  // Assign odd numbers
  // var oddNumbers: Seq[Int] = numbers filter (_%2 != 0)

  // Chain calls on result
  println("===========: Odd numbers")
  numbers filter (_%2 != 0) foreach println

  // _*_ is not possible because second _ is considered to be the second argument and not the first again
  println("===========: X^2 of the odd numbers")
  numbers filter (_%2 != 0) map (i => i) foreach println

  // Save resulting sequence
  val oddSqr = numbers filter (_%2 != 0) map (i => i)
  // Self made sum
  // val reduceSum: Int = oddSqr reduce((acc, i) => acc+i)

  // ###########################################################
  // Aggregator functions
  // ###########################################################
  val reduceSum: Int = oddSqr.sum
  println(s"reduceSum: $reduceFold")

  val reduceFold: Int = oddSqr.foldLeft(0)((acc,i)=> acc+i)
  println(s"reduceFold: $reduceFold")

  // currying, two parameter lists where the first one returns a function, which the second parameter list is applied to
  val foldLeftStr: String = (1 to 3).foldLeft("0")((acc, i) => s"f($acc,$i)")
  println(s"foldLeftStr: $foldLeftStr")

  val foldRightStr: String = (1 to 3).foldRight("0")((acc, i) => s"f($acc,$i)")
  println(s"foldRightStr: $foldRightStr")

  val func: ((Int, String) => String) => String = (1 to 3).foldRight("0")
  var funcRes = func((acc, i) => s"f($acc,$i)")
  println(s"func: $func")
  println(s"funcRes: $funcRes")
}
