package mpv.actor

import java.net.{MalformedURLException, URL}
import java.util.UUID
import java.util.concurrent.TimeUnit

import akka.actor.SupervisorStrategy.{Resume, Stop}
import akka.actor.{Actor, ActorRef, ActorSystem, OneForOneStrategy, PoisonPill, Props, Terminated}
import akka.http.scaladsl.Http
import akka.http.scaladsl.model.{HttpRequest, HttpResponse, StatusCode}
import akka.http.scaladsl.unmarshalling.Unmarshal
import akka.stream.ActorMaterializer

import scala.collection.mutable
import scala.concurrent.duration.{Duration, _}
import scala.concurrent.{Await, ExecutionContextExecutor, Future}
import scala.util.Random
import scala.util.matching.Regex

/**
  * @author Thomas Herzog <herzog.thomas81@gmail.com>
  * @since 12/25/17
  */
object Application extends App {

  implicit val system = ActorSystem("ApplicationHttpAkka")
  implicit val materializer = ActorMaterializer()
  implicit val executionContext: ExecutionContextExecutor = system.dispatcher

  val MAX_DOWNLOAD_WORKER = 30

  object Action extends Enumeration {
    val START, STOP = Value
  }

  class DownloadException(private val message: String = "",
                          private val cause: Throwable = None.orNull)
    extends RuntimeException(message, cause)

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
        try {
          val base = new URL(url)
          new URL(new URL(s"${base.getProtocol}://${base.getHost}"), link).toString
        } catch {
          case _ => null
        }
      }
    }

    def findAbsoluteLinks(): Iterator[String] = findLinks() map toAbsoluteUrl filter (url => url != null)

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
    * The base actor which is the base for all implemented actors and provides common
    * functionality.
    *
    * @author Thomas Herzog <herzog.thomas81@gmail.com>
    * @since 12/7/17
    */
  abstract class BaseActor extends Actor {

    object Timeout

    override def receive: PartialFunction[Any, Unit] = PartialFunction.empty[Any, Unit]

    override def postRestart(reason: Throwable): Unit = {
      println(s"${self.path.name}: RESTARTED: reason: '$reason'")
      super.postRestart(reason)
    }

    override def unhandled(message: Any): Unit = {
      println(s"${self.path.name}: UNHANDLED: MESSAGE RECEIVED: $message")
      super.unhandled(message)
    }
  }

  /**
    * The web crawler which searches the web pages provided by links of the given root url
    * and searches for a pattern in the web pages
    *
    * @param owner    the actor owning this crawler
    * @param rootUrl  the root url where to start
    * @param maxDepth the maximum depth of to search links
    * @param pattern  the search pattern for the web pages
    */
  class Crawler(owner: ActorRef, rootUrl: String, maxDepth: Int, pattern: Regex) extends BaseActor {
    val random: Random = new Random()
    var visitedPages = mutable.Set.empty[WebPage]
    var openUrls = mutable.Set.empty[(Int, String)]
    var visitedUrls = mutable.Set.empty[String]
    var currentUrls = mutable.Set.empty[String]

    /**
      * Downloads the web page by sending to Downloader
      *
      * @param worker the downloader worker
      * @param msg    the ms to be sent
      */
    def download(worker: ActorRef, msg: (Int, String)): Unit = {
      currentUrls += msg._2
      println(s"DOWNLOAD: '${self.path}', Level: '${msg._1}', Url: ${msg._2}")
      worker ! msg
    }

    /**
      * Creates the workers for downloading the web pages
      *
      * @param requestWorkers the count of workers which are requested
      * @return the created workers, an empty list if no workers could be created
      */
    def createWorkers(requestWorkers: Int): Iterable[ActorRef] = {
      val availableCount = MAX_DOWNLOAD_WORKER - context.children.size
      var actualCount: Int = requestWorkers
      if (requestWorkers > availableCount) {
        actualCount = availableCount
      }
      println(s"actualCount: $actualCount | children: ${context.children.size}")
      if (actualCount > 0) {
        val workers: Seq[ActorRef] = (1 to actualCount).map(_ => context.actorOf(Props(classOf[Downloader]), s"${classOf[Downloader].getName}_${UUID.randomUUID().toString}"))
        workers foreach context.watch
        workers
      } else {
        Iterable.empty[ActorRef]
      }
    }

    /**
      * Tries to download the open url where no worker was available
      */
    def workOnOpen(): Unit = {
      openUrls = openUrls.filter(t => !currentUrls.contains(t._2)).filter(t => !visitedUrls.contains(t._2))
      if (openUrls.nonEmpty) {
        val workers: Iterable[ActorRef] = createWorkers(openUrls.size)
        if (workers.nonEmpty) {
          val workerIt = workers.iterator
          for (tuple <- openUrls) {
            if (workerIt.hasNext) {
              download(workerIt.next(), tuple)
            }
          }
        }
      }
    }

    /**
      * Hanlde the downloaded web page
      *
      * @param webPage the web page to handle
      */
    def handle(webPage: WebPage): Unit = {
      println(s"Self: ${self.path} \r\n Url: ${webPage.url}")
      visitedUrls += webPage.url
      currentUrls -= webPage.url
      val newUrls: mutable.Set[String] = (mutable.Set.empty[String] ++ webPage.findAbsoluteLinks()) --= visitedUrls
      var newOpenUrls = mutable.Set.empty[(Int, String)]

      if (pattern.findFirstIn(webPage.content).nonEmpty) {
        visitedPages += webPage
      }
      if (((webPage.level + 1) > maxDepth) || (context.children.size >= MAX_DOWNLOAD_WORKER) || newUrls.isEmpty) {
        if (openUrls.nonEmpty) {
          workOnOpen()
        }
      }
      else {
        val workers: Iterable[ActorRef] = createWorkers(newUrls.size)
        val workerIt = workers.iterator
        for (url <- newUrls) {
          val tuple = (webPage.level + 1, url)
          if (workerIt.hasNext) {
            download(workerIt.next(), tuple)
          } else {
            newOpenUrls += tuple
          }
        }
        // Assume workers are left, because no new open url has been added
        openUrls ++= newOpenUrls
        if (newOpenUrls.isEmpty) {
          workOnOpen()
        }
      }
    }

    override def supervisorStrategy: OneForOneStrategy = OneForOneStrategy(maxNrOfRetries = 2, withinTimeRange = Duration.apply(100, TimeUnit.MILLISECONDS)) {
      case _: MalformedURLException => Resume
      case _: DownloadException => Resume
      case _: IllegalStateException => Resume
      case _ => Stop
    }

    override def receive: PartialFunction[Any, Unit] = {
      case Action.START =>
        println(s"RECEIVE START: actor: ${self.path} from ${sender().path}")
        val worker = system.actorOf(Props(classOf[Downloader]), s"${classOf[Downloader].getName}_root_url")
        context.watch(worker)
        download(worker, (0, rootUrl))
      case webPage: WebPage =>
        println(s"RECEIVE WebPage: actor: ${self.path} from ${sender().path}")
        handle(webPage)
      case Action.STOP =>
        println(s"'RECEIVE STOP: openUrl: ${openUrls.size}, visitedUrl: ${visitedUrls.size}, currentUrl: ${currentUrls.size} actor: ${self.path} from ${sender().path}")
        context.children foreach (child => child ! PoisonPill)
        owner ! visitedUrls
      case Terminated(actor) =>
        println(s"'RECEIVE Terminated: children: ${context.children.size} parent: ${self.path} child-actor: ${actor.path}")
        if (openUrls.nonEmpty) {
          workOnOpen()
        } else if (context.children.isEmpty) {
          self ! Action.STOP
        }
    }
  }

  /**
    * The downloader actor.
    *
    * @author Thomas Herzog <herzog.thomas81@gmail.com>
    * @since 12/25/17
    */
  class Downloader extends BaseActor {

    override def receive: PartialFunction[Any, Unit] = {
      case (level: Int, url: String) =>
        try {
          println(s"Thread#id: '${Thread.currentThread().getId}' | url: ($level, $url)")
          val urlCallFuture: Future[HttpResponse] = Http().singleRequest(HttpRequest(uri = new URL(url).toExternalForm))
          val result: HttpResponse = Await.result(urlCallFuture, Duration.Inf)
          val status: StatusCode = result.status
          if (status.isFailure()) {
            context.sender() ! WebPage(level, url, s"ERROR: Download failed for status: '$status'")
          }
          context.sender() ! WebPage(level, url, Await.result(Unmarshal(result.entity).to[String], Duration.Inf))
          result.discardEntityBytes()
        } catch {
          case t: MalformedURLException =>
            println(s"EXCEPTION: ${self.path}, $t")
            throw t
          case t: Throwable => println(s"EXCEPTION: ${self.path}, $t")
            throw new DownloadException(s"Download failed: ${t.getMessage}", t)
        } finally {
          context.stop(self)
        }
      case Timeout =>
        println(s"'RECEIVE Timeout: actor: ${self.path} from ${sender().path}")
        self ! Action.STOP
      case Terminated(actor) =>
        println(s"'RECEIVE Terminated: parent: ${self.path} child-actor: ${actor.path}")
    }
  }

  class ApplicationDownloaderMain extends BaseActor {
    val downloader: ActorRef = system.actorOf(Props(classOf[Downloader]), s"${classOf[Downloader].getName}")

    context.system.scheduler.scheduleOnce(240.seconds, self, Timeout)

    override def receive: PartialFunction[Any, Unit] = {
      case Action.START =>
        downloader ! "http://www.google.at"
      case Action.STOP =>
        context.stop(downloader)
        context.stop(self)
        Await.result(Http().shutdownAllConnectionPools(), Duration.apply(5, TimeUnit.SECONDS))
        materializer.shutdown()
        Await.result(context.system.terminate(), Duration.apply(5, TimeUnit.SECONDS))
      case WebPage(_, url, _) =>
        println(s"Url: $url \r\n content: ")
        self ! Action.STOP
      case Terminated(actor) =>
        println(s"'RECEIVE Terminated: parent: ${self.path} child-actor: ${actor.path}")
    }
  }

  class ApplicationCrawlerMain extends BaseActor {
    val crawler: ActorRef = system.actorOf(Props(classOf[Crawler], self, "https://www.fh-ooe.at/campus-hagenberg/", 7, ".*.".r), s"${classOf[Crawler].getName}")

    context.system.scheduler.scheduleOnce(240.seconds, self, Timeout)

    override def supervisorStrategy: OneForOneStrategy = OneForOneStrategy(maxNrOfRetries = 2, withinTimeRange = Duration.apply(100, TimeUnit.MILLISECONDS)) {
      case _ => Resume
    }

    override def receive: PartialFunction[Any, Unit] = {
      case Action.START =>
        crawler ! Action.START
      case pages: mutable.Set[WebPage] =>
        println(s"===============================================")
        println(s"Page Count: ${pages.size}")
        println(s"Pages     :")
        pages foreach println
        println(s"===============================================")
        context.stop(crawler)
        context.stop(self)
        Await.result(Http().shutdownAllConnectionPools(), Duration.apply(10, TimeUnit.SECONDS))
        materializer.shutdown()
        Await.result(context.system.terminate(), Duration.apply(10, TimeUnit.SECONDS))
      case Timeout =>
        println(s"'RECEIVE Timeout: actor: ${self.path} from ${sender().path}")
        crawler ! Action.STOP
        // Graceful shutdown not possible when aborting, because of stream errors of akka http
      case Terminated(actor) =>
        println(s"'RECEIVE Terminated: parent: ${self.path} child-actor: ${actor.path}")
    }
  }

  //val app: ActorRef = system.actorOf(Props(classOf[ApplicationDownloaderMain]), s"${classOf[ApplicationDownloaderMain].getName}")
  //app ! Action.START

  val app: ActorRef = system.actorOf(Props(classOf[ApplicationCrawlerMain]), s"${
    classOf[ApplicationCrawlerMain].getName
  }")
  app ! Action.START
}
