package mpv.basics.actors

import akka.actor.{Actor, ActorRef, ActorSystem, PoisonPill, Props, Terminated}
import mpv.basics.actors.SimplePrimeCalculator.{Find, Found}

import scala.concurrent.{Await, Future}
import scala.concurrent.duration.Duration

/**
  * Companion object, contains static methods of the class SimplePrimeCalculator
  */
object SimplePrimeCalculator {

  // Case class is a ordinary class, but very good integrated in pattern matching
  case class Find(lower: Int, upper: Int)

  case class Found(lower: Int, upper: Int, primes: Seq[Int])

  def isPrime(n: Int): Boolean = {
    val upper = math.sqrt(n + 0.5).toInt
    (2 to upper) forall (i => (n % i) != 0)
  }

  def name: String = {
    classOf[SimplePrimeCalculator].getName
  }
}

/**
  * @author Thomas Herzog <herzog.thomas81@gmail.com>
  * @since 12/7/17
  */
class SimplePrimeCalculator extends Actor {

  import SimplePrimeCalculator._

  // Here we define a partial function
  // Can receive anything, therefore we use case to match received message
  override def receive: Receive = {
    case Find(lower, upper) =>
      val primes = (lower to upper) filter isPrime
      // notify sender about receive
      sender() ! Found(lower, upper, primes)
  }
}

/**
  * This actor manages other actors
  */
class SimplePrimeCalculatorMain extends Actor {

  // Will be done in main constructor (class MainActor() extends Actor)
  val calculator = context.actorOf(Props[SimplePrimeCalculator], SimplePrimeCalculator.name)
  calculator ! Find(2, 100)
  calculator ! Find(1000, 2000)
  calculator ! "Some invalid message"
  calculator ! PoisonPill // Will kill the actor when it handles this message

  context.watch(calculator)

  override def receive = {
    case Found(lower, upper, primes) =>
      println(s"SimplePrimeCalculator#isPrime($lower, $upper) = { ${primes.mkString(",")} }")
    case Terminated(_) =>
      context.system.terminate()
  }

  override def unhandled(message: Any): Unit = {
    println(s"Unknown message received: $message")
    super.unhandled(message)
  }

}

object SimplePrimeCalculatorApp extends App {
  println("============= SimplePrimeCalculatorApp =============")

  val find = Find(2, 100)

  val system = ActorSystem("SimplePrimeCalculatorAppSystem")
  // val calculator: ActorRef = system.actorOf(Props[SimplePrimeCalculator], SimplePrimeCalculator.name) // classOf (Overload of Props, same as Props(classOf[...])) scala operator, where we can check if class#isAssignableForm()
  val calculator: ActorRef = system.actorOf(Props[SimplePrimeCalculatorMain])

  // MAinActor has a case Terminated(_) which will terminate the actor system

  // Actor system would otherwise run forever
  //Thread.sleep(500) // Be careful here, actor couldn't be done yet
  //val done: Future[Terminated] = system.terminate()
  //Await.ready(done, Duration.Inf)
}
