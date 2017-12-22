package mpv.actor

import akka.actor.Actor

import scala.concurrent.duration._

/**
  * @author Thomas Herzog <herzog.thomas81@gmail.com>
  * @since 12/7/17
  */
abstract class BaseActor(maxDuration: FiniteDuration) extends Actor {

  // Defaults to 100 millis for all actors if they don't define timeout
  def this() = this(100.millis)

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