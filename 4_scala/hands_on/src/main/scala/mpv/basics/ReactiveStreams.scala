package mpv.basics

import akka.{Done, NotUsed}
import akka.actor.ActorSystem
import akka.stream.ActorMaterializer
import akka.stream.scaladsl.{Flow, Keep, RunnableGraph, Sink, Source}

import scala.concurrent.duration.Duration
import scala.concurrent.{Await, Future}

/**
  * At the end of the stream the exceptions will have to be handled, because they are not thrown within the stream.
  *
  * @author Thomas Herzog <herzog.thomas81@gmail.com>
  * @since 11/24/17
  */
object ReactiveStreams extends App {

  // If implicit parameter is required, these one will be used
  implicit val system = ActorSystem("mySystem")
  implicit val materializer = ActorMaterializer()

  def quickStart(): Unit = {
    // Describe the graph nodes
    val source: Source[Int, NotUsed] = Source(1 to 3)
    val flow: Flow[Int, Int, NotUsed] = Flow[Int].map(_ * 2)
    val sink: Sink[Int, Future[Done]] = Sink.foreach[Int](i => print(s"$i "))

    // val source1: Source[Int, NotUsed] = source.via(flow)
    // val sink2 = Sink[Int, NotUsed] = flow.to(sink)

    // Build graph with the nodes
    val graph: RunnableGraph[Future[Done]] = source.via(flow).toMat(sink)(Keep.right)
    val done: Future[Done] = graph.run

    Await.ready(done, Duration.Inf)
  }

  def primeTwins(): Unit = {
    def isPrime(n: Int): Boolean = {
      val upper = math.sqrt(n).toInt + 1
      (2 to upper) forall (i => n % i != 0)
    }

    val source: Source[Int, NotUsed] = Source(1 to 100)
    val printPairSink = Sink.foreach[(Int, Int)](pair => print(s"$pair "))

    val oddPairSource: Source[(Int, Int), NotUsed] = source.map(2 * _ + 1).map(i => (i, i + 2))
    val twinPairSource: Source[(Int, Int), NotUsed] = oddPairSource.async.filter(pair => isPrime(pair._1) && isPrime(pair._2))
    val done: Future[Done] = twinPairSource.runWith(printPairSink)

    Await.ready(done, Duration.Inf)
  }

  println("===============: Reactive streams")

  println("---------------: quickstart")
  // quickStart()

  println("---------------: primetwins")
  primeTwins()

  system.terminate()
}
