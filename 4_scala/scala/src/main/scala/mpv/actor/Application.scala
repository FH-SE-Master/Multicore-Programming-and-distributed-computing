package mpv.actor

import java.net.URL

import akka.actor.{Actor, ActorRef, ActorSystem, Props}
import akka.http.scaladsl.Http
import akka.http.scaladsl.model.{HttpRequest, HttpResponse, StatusCode}
import akka.http.scaladsl.unmarshalling.Unmarshal
import akka.stream.ActorMaterializer

import scala.collection.mutable
import scala.concurrent.duration.{Duration, FiniteDuration, _}
import scala.concurrent.{Await, ExecutionContextExecutor, Future}
import scala.util.matching.Regex

/**
  * @author Thomas Herzog <herzog.thomas81@gmail.com>
  * @since 12/25/17
  */
object Application extends App {

  implicit val system = ActorSystem("ApplicationHttpAkka")
  implicit val materializer = ActorMaterializer()
  implicit val executionContext: ExecutionContextExecutor = system.dispatcher

  val MAX_DOWNLOAD_WORKER = 512

  object Action extends Enumeration {
    val START, STOP = values
  }

  object State extends Enumeration {
    val STOPPED, RUNNING, WAIT_FOR_STOP = values
  }

  object WebPage {
    val ANCHOR_TAG_PATTERN = "(?i)<a ([^>]+)>.+?</a>".r
    val HREF_ATTR_PATTERN = """\s*(?i)href\s*=\s*(?:"([^"]*)"|'([^']*)'|([^'">\s]+))\s*""".r
  }

  case class WebPage(level: Int, url: String, var content: String) {

    import WebPage._

    def toAbsoluteUrl(link: String): String = {
      if (link.startsWith("http"))
        link
      else {
        val base = new URL(url)
        new URL(new URL(s"${base.getProtocol}://${base.getHost}"), link).toString
      }
    }

    def findAbsoluteLinks(): Iterator[String] = findLinks() map toAbsoluteUrl

    def findLinks(): Iterator[String] = {
      for {
        anchor <- ANCHOR_TAG_PATTERN.findAllMatchIn(content)
        HREF_ATTR_PATTERN(dquot, quot, bare) <- anchor.subgroups
      } yield if (dquot != null) dquot
      else if (quot != null) quot
      else bare
    }

    def contains(pattern: Regex): Boolean = pattern.findFirstIn(this.content).isDefined

    def cleanUrlForActor(): String = {
      url.replace("/", "_$_").replace("\\", "_#_")
    }
  }

  /**
    * @author Thomas Herzog <herzog.thomas81@gmail.com>
    * @since 12/7/17
    */
  abstract class BaseActor(maxDuration: FiniteDuration) extends Actor {

    object Timeout

    // Defaults to 100 millis for all actors if they don't define timeout
    def this() = {
      this(100.millis)
      context.system.scheduler.scheduleOnce(maxDuration, self, Timeout)
    }

    override def receive: PartialFunction[Any, Unit] = {
      case Timeout =>
        println(s"${self.path.name}: TIMEOUT: by scheduler")
    }

    override def postRestart(reason: Throwable): Unit = {
      println(s"${self.path.name}: RESTARTED: reason: '$reason'")
      super.postRestart(reason)
    }

    override def unhandled(message: Any): Unit = {
      println(s"${self.path.name}: UNHANDLED: MESSAGE RECEIVED: $message")
      super.unhandled(message)
    }

    protected def registerForTimeout(): Unit = {
      context.children foreach context.watch
    }
  }

  class Crawler(rootUrl: String, maxDepth: Int, pattern: Regex) extends BaseActor(10.millis) {
    var owner: ActorRef = _
    var visitedPages = mutable.Set.empty[WebPage]
    var visitedUrls = mutable.Set.empty[String]
    var currentUrls = mutable.Set.empty[String]
    var state: State.ValueSet = State.STOPPED

    def download(worker: ActorRef, msg: (Int, String)): Unit = {
      currentUrls += msg._2
      println(s"DOWNLOAD: '${self.path}', Level: '${msg._1}', Url: ${msg._2}")
      worker ! msg
    }

    def handle(webPage: WebPage): Unit = {
      println(s"Self: ${self.path} \r\n Url: ${webPage.url}")
      visitedPages += webPage
      visitedUrls += webPage.url
      currentUrls -= webPage.url
      val newUrls: mutable.Set[String] = (mutable.Set.empty[String] ++ webPage.findAbsoluteLinks()) --= visitedUrls

      if (newUrls.nonEmpty) {
        if ((webPage.level + 1) <= maxDepth) {
          if (context.children.size < MAX_DOWNLOAD_WORKER) {
            val availableCount = MAX_DOWNLOAD_WORKER - context.children.size
            var actualCount: Int = newUrls.size
            if (newUrls.size > availableCount) {
              actualCount = availableCount
            }

            val workers: Seq[ActorRef] = (1 to actualCount).map(idx => system.actorOf(Props(classOf[Downloader]), s"${classOf[Downloader].getName}_${webPage.cleanUrlForActor()}_$idx"))
            registerForTimeout()
            val workerIt = workers.iterator
            for (url <- newUrls) {
              if (workerIt.hasNext) {
                download(workerIt.next(), (webPage.level + 1, url))
              }
            }
          }
        } else {
          println(s"MaxDepth: $maxDepth reached, therefore stopping")
          state = State.WAIT_FOR_STOP
        }
      }
    }

    override def receive: PartialFunction[Any, Unit] = {
      case Action.START =>
        owner = sender()
        state = State.RUNNING
        val worker = system.actorOf(Props(classOf[Downloader]), s"${classOf[Downloader].getName}_root_url")
        registerForTimeout()
        download(worker, (1, rootUrl))
      case webPage: WebPage =>
        handle(webPage)
        if(currentUrls.isEmpty) {
          owner ! visitedUrls
        }
      case Action.STOP =>
        println(s"STOPPING: actor: ${self.path}")
        context.parent ! Action.STOP
      case Timeout =>
        context.children.foreach(context.stop)
        owner ! visitedUrls
      case _ => super.receive
    }
  }

  /**
    * @author Thomas Herzog <herzog.thomas81@gmail.com>
    * @since 12/25/17
    */
  class Downloader extends BaseActor {

    override def receive: PartialFunction[Any, Unit] = {
      case (level: Int, url: String) =>
        println(s"Thread#id: '${Thread.currentThread().getId}' | url: ($level, $url)")
        try {
          val urlCallFuture: Future[HttpResponse] = Http().singleRequest(HttpRequest(uri = url))
          val result: HttpResponse = Await.result(urlCallFuture, Duration.Inf)
          val status: StatusCode = result.status
          if (status.isFailure()) {
            result.discardEntityBytes()
            context.sender() ! WebPage(level, url, s"ERROR: Download failed for status: '$status'")
          }
          context.sender() ! WebPage(level, url, Await.result(Unmarshal(result.entity).to[String], Duration.Inf))
        } catch {
          case t: Throwable => context.sender() ! WebPage(level, url, t.getMessage)
        } finally {
          self ! Action.STOP
        }
      case _ => super.receive
    }
  }

  class ApplicationDownloaderMain extends BaseActor {
    val downloader: ActorRef = system.actorOf(Props(classOf[Downloader]), s"${classOf[Downloader].getName}")

    override def receive: PartialFunction[Any, Unit] = {
      case Action.START =>
        downloader ! "http://www.google.at"
      case Action.STOP =>
        context.stop(downloader)
        context.stop(self)
        context.system.terminate()
      case WebPage(_, url, _) =>
        println(s"Url: $url \r\n content: ")
        self ! Action.STOP
      case _ => super.receive
    }
  }

  class ApplicationCrawlerMain extends BaseActor {
    val crawler: ActorRef = system.actorOf(Props(classOf[Crawler], "https://www.fh-ooe.at/campus-hagenberg/", 3, ".*.".r), s"${classOf[Crawler].getName}")

    registerForTimeout()

    override def receive: PartialFunction[Any, Unit] = {
      case Action.START =>
        crawler ! Action.START
      case pages: mutable.Set[WebPage] =>
        println(s"Crawled Pages: $pages")
        context.stop(crawler)
        context.stop(self)
        context.system.terminate()
      case _ => super.receive
    }
  }

  //val app: ActorRef = system.actorOf(Props(classOf[ApplicationDownloaderMain]), s"${classOf[ApplicationDownloaderMain].getName}")
  //app ! Action.START

  val app: ActorRef = system.actorOf(Props(classOf[ApplicationCrawlerMain]), s"${classOf[ApplicationCrawlerMain].getName}")
  app ! Action.START
}
