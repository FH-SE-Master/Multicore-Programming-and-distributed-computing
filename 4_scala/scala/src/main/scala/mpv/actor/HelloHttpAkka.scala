package mpv.actor

import akka.actor.ActorSystem
import akka.http.scaladsl.Http
import akka.http.scaladsl.model.{HttpRequest, HttpResponse, StatusCode}
import akka.http.scaladsl.unmarshalling.Unmarshal
import akka.stream.ActorMaterializer

import scala.concurrent.duration.Duration
import scala.concurrent.{Await, ExecutionContextExecutor, Future}

object Action extends Enumeration {
  val START, STOP = Value
}

abstract class HttpAkkaResponse(val url: String) {}

class HttpAkkaSuccessResponse(override val url: String, val content: String) extends HttpAkkaResponse(url) {}

class HttpAkkaErrorResponse(override val url: String, val exc: Throwable) extends HttpAkkaResponse(url) {}

/**
  * @author Thomas Herzog <herzog.thomas81@gmail.com>
  * @since 12/25/17
  */
class HelloHttpAkka {

  implicit val system = ActorSystem("test_helloHttpAkkaSystem")
  implicit val materializer = ActorMaterializer()
  implicit val executionContext: ExecutionContextExecutor = system.dispatcher


  def download(url: String): Future[HttpAkkaResponse] = {
    Future {
      println(s"Thread#id: '${Thread.currentThread().getId}' | url: $url")
      try {
        val urlCallFuture: Future[HttpResponse] = Http().singleRequest(HttpRequest(uri = url))
        val result: HttpResponse = Await.result(urlCallFuture, Duration.Inf)
        val status: StatusCode = result.status
        if (status.isFailure()) {
          new HttpAkkaErrorResponse(url, new IllegalStateException(s"ERROR: Download failed for status: '$status'"))
        }
        new HttpAkkaSuccessResponse(url, Await.result(Unmarshal(result.entity).to[String], Duration.Inf))
      } catch {
        case t: Throwable => new HttpAkkaErrorResponse(url, t)
      }
    }
  }

  def terminate(): Unit = {
    system.terminate()
  }
}


/**
  * @author Thomas Herzog <herzog.thomas81@gmail.com>
  * @since 12/25/17
  */
object HelloHttpAkkaApplication extends App {

  implicit val system = ActorSystem("test_helloHttpAkkaSystem")
  implicit val executionContext: ExecutionContextExecutor = system.dispatcher

  def test_helloHttpAkka_success(): Unit = {
    val httpAkka = new HelloHttpAkka()
    try {
      val responseFuture = httpAkka.download("http://www.google.at")
      val response = Await.result(responseFuture, Duration.Inf)
      println(s"Ended: ${response.getClass.getSimpleName} \r\n url: ${response.url} \r\n content: ${response.asInstanceOf[HttpAkkaSuccessResponse].content}")
    } catch {
      case t: Throwable => println(s"Exception: $t")
    } finally {
      httpAkka.terminate()
    }
  }

  def test_helloHttpAkka_error(): Unit = {
    val httpAkka: HelloHttpAkka = new HelloHttpAkka()
    try {
      val responseFuture = httpAkka.download("bubu://www...google.at")
      val response = Await.result(responseFuture, Duration.Inf)
      println(s"Ended: ${response.getClass.getSimpleName} \r\n url: ${response.url} \r\n content: ${response.asInstanceOf[HttpAkkaErrorResponse].exc}")
    } catch {
      case t: Throwable => println(s"Exception: $t")
    } finally {
      httpAkka.terminate()
    }
  }

  test_helloHttpAkka_success()
  test_helloHttpAkka_error()
}
