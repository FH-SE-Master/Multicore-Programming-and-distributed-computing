package mpv.basics.actors

import java.util.concurrent.TimeUnit

import akka.actor.SupervisorStrategy.Restart
import akka.actor.{Actor, ActorRef, ActorSystem, OneForOneStrategy, Props, SupervisorStrategy, Terminated}
import akka.pattern.pipe
import mpv.basics.actors.PrimeCalculator._
import mpv.util.LoggingUtil
import mpv.util.RangeUtil._

import scala.collection.mutable
import scala.concurrent.Future
import scala.concurrent.duration._
import scala.util.Random

/**
  * Companion object, contains static methods of the class PrimeCalculator
  */
object PrimeCalculator {

  // Case class is a ordinary class, but very good integrated in pattern matching
  case class Failed(lower: Int, upper: Int)

  case class Found(lower: Int, upper: Int, primes: Seq[Int])

  def isPrime(n: Int): Boolean = {
    val upper = math.sqrt(n + 0.5).toInt
    (2 to upper) forall (i => (n % i) != 0)
  }

  def name: String = {
    classOf[SimplePrimeCalculator].getName
  }
}

abstract class BaseActor extends Actor {

  override def receive: PartialFunction[Any, Unit] = PartialFunction.empty

  override def postRestart(reason: Throwable): Unit = {
    println(s"${self.path.name}: RESTARTED: reason: '$reason'")
    super.postRestart(reason)
  }

  override def unhandled(message: Any): Unit = {
    println(s"${self.path.name}: UNHANDLED: MESSAGE RECEIVED")
    super.unhandled(message)
  }
}

class PrimeCalculator(lower: Int, upper: Int, maxDuration: FiniteDuration) extends BaseActor {
  import context.dispatcher

  private val MAX_WORKERS: Int = Runtime.getRuntime.availableProcessors()
  private val aggregatedPrimes = mutable.Set.empty[Int]
  private val completed = mutable.Set.empty[ActorRef]
  private var nWorkers = 0

  def this(lower: Int, upper: Int) = this(lower, upper, 100.millis)

  object Timeout

  createChildren()
  registerForDeathwath()
  registerForTimeout()

  private def createChildren(): Unit = {
    for ((lower, upper) <- splitIntoIntervals(lower, upper, MAX_WORKERS)) {
      context.actorOf(Props(classOf[PrimeFinder], lower, upper), s"${classOf[PrimeFinder].getSimpleName}_${lower}_$upper")
      nWorkers = context.children.size
    }
  }

  private def registerForDeathwath(): Unit = {
    context.children foreach context.watch
  }

  private def registerForTimeout(): Unit = {
    context.system.scheduler.scheduleOnce(maxDuration, self, Timeout)
  }

  override def supervisorStrategy: OneForOneStrategy = OneForOneStrategy(2) {
    case _: ArithmeticException =>
      Restart
    case ex =>
      SupervisorStrategy.defaultStrategy.decider(ex)
  }

  override def receive: PartialFunction[Any, Unit] = {
    case primes: Seq[Int] =>
      aggregatedPrimes ++= primes
      completed += sender()
      if (completed.size >= nWorkers) {
        context.parent ! Found(lower, upper, aggregatedPrimes.toSeq)
        context.stop(self)
      }
    case Timeout =>
      println(s"${self.path.name}: TIMEOUT: by scheduler")
      context.parent ! Failed(lower, upper)
      context stop self

    case Terminated(actor) =>
      // 1. Destroyed by us, because ended successfully (not needed to be handled)
      // 2. Destroyed by supervisor in case of an error
      if (!(completed contains actor)) {
        println(s"${actor.path.name}: STOPPED: by supervisor")
        context.parent ! Failed(lower, upper)
        context stop self
      }
  }
}

class PrimeFinder(lower: Int, upper: Int) extends BaseActor {

  // Is a execution context, and contains the fork join pool
  import context.dispatcher

  // Executed in constructor, bad to do synchronously, therefore asynchronously
  Future {
    simulateFailure(0.2) {
      (lower to upper) filter isPrime
    }
  } pipeTo self // pipe to this actor when done

  // Auxiliary constructors
  def this(upper: Int) = this(2, upper)

  private def simulateFailure[T](probability: Double)(body: => T): T = {
    val result = body
    if (Random.nextDouble() <= probability) {
      throw new ArithmeticException("Computation failed")
    }

    result
  }

  override def receive: PartialFunction[Any, Unit] = {
    case primes: Seq[_] =>
      println(s"${self.path.name}: RECEIVE: ($lower, $upper) = { ${primes.mkString(",")} }")
      // Notify parent actor about found primes
      context.parent ! primes
      // Stop this actor
      context.stop(self)
    case akka.actor.Status.Failure(ex) =>
      println(s"${self.path.name}: FAILED: '$ex'")
      throw ex
  }
}

class PrimeCalculatorMain extends BaseActor {

  //context.actorOf(Props(classOf[PrimeCalculator], 2, 100), classOf[PrimeCalculator].getSimpleName + "_1")
  context.actorOf(Props(classOf[PrimeCalculator], 1000, 2000, FiniteDuration.apply(100, TimeUnit.MILLISECONDS)), classOf[PrimeCalculator].getSimpleName + "_2")

  // So that we receive terminate messages
  context.children foreach context.watch

  override def receive: PartialFunction[Any, Unit] = {
    case Found(lower, upper, primes) =>
      println(s"${self.path.name}: RECEIVED: ($lower, $upper) = { ${primes.mkString(",")} }")
    case Failed(lower, upper) =>
      println(s"${self.path.name}: FAILED: ($lower, $upper)")
    case Terminated(_) =>
      context.system.terminate()
  }
}


object PrimeCalculatorApp extends App {

  println("============= PrimeCalculatorApp =============")

  val system = ActorSystem("PrimeCalculatorAppSystem", LoggingUtil.loggingLevel("OFF"))
  val calculator: ActorRef = system.actorOf(Props[PrimeCalculatorMain], s"${classOf[PrimeCalculatorMain].getSimpleName}")
}
