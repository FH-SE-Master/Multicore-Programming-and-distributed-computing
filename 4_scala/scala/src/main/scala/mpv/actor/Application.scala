package mpv.actor

import java.net.URL
import java.util
import java.util.concurrent.TimeUnit
import java.util.{Collections, UUID}

import akka.actor.SupervisorStrategy.{Escalate, Resume, Stop}
import akka.actor.{Actor, ActorRef, ActorSystem, OneForOneStrategy, PoisonPill, Props, Terminated}
import akka.http.scaladsl.Http
import akka.http.scaladsl.model.{HttpRequest, HttpResponse, StatusCode}
import akka.http.scaladsl.unmarshalling.Unmarshal
import akka.stream.ActorMaterializer

import scala.collection.mutable
import scala.concurrent.duration.{Duration, _}
import scala.concurrent.{Await, ExecutionContextExecutor, Future, Promise}
import scala.util.matching.Regex
import scala.util.{Failure, Random, Success}

/**
  * @author Thomas Herzog <herzog.thomas81@gmail.com>
  * @since 12/25/17
  */
object Application extends App {

  // region Traits, Companion Objects, case classes and so on
  implicit val system: ActorSystem = ActorSystem("ApplicationHttpAkka")
  implicit val materializer: ActorMaterializer = ActorMaterializer()
  implicit val executionContext: ExecutionContextExecutor = system.dispatcher

  var searchEngines: util.Set[String] = Collections.synchronizedSet(new util.HashSet[String]())

  object Timeout

  object Action extends Enumeration {
    val START, STOP, HARD_SHUTDOWN, GRACEFUL_SHUTDOWN, REMOVE = Value
  }


  object WebPage {
    val ANCHOR_TAG_PATTERN: Regex = "(?i)<a ([^>]+)>.+?</a>".r
    val HREF_ATTR_PATTERN: Regex = """\s*(?i)href\s*=\s*(?:"([^"]*)"|'([^']*)'|([^'">\s]+))\s*""".r
  }

  class DownloadException(private val message: String = "",
                          private val cause: Throwable = None.orNull)
    extends RuntimeException(message, cause)

  trait TimeUtil {
    def differenceInMillis(thenMillis: Long): Long = {
      System.currentTimeMillis() - thenMillis
    }

    def differenceInSeconds(thenMillis: Long): Long = {
      differenceInMillis(thenMillis) / 1000
    }
  }

  trait SearchEngine {
    def search(url: String, pattern: Regex, depth: Int = 0,
               maxDuration: FiniteDuration = 3.seconds): Future[Set[WebPage]]

    def shutdown(): Future[Terminated]
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

  // endregion

  /**
    * The base actor which is the base for all implemented actors and provides common
    * functionality.
    *
    * @author Thomas Herzog <herzog.thomas81@gmail.com>
    * @since 12/7/17
    */
  abstract class BaseActor extends Actor {

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
  class Crawler(owner: ActorRef, rootUrl: String, maxDepth: Int, maxWorkers: Int, pattern: Regex) extends BaseActor {
    val random: Random = new Random()
    val maxDownloadDuration: FiniteDuration = 1.second

    var visitedPages = mutable.Set.empty[WebPage]
    var queuedUrls = mutable.Set.empty[(Int, String)]
    var foundUrls = mutable.Set.empty[String]
    var downloadingUrls = mutable.Set.empty[String]
    var skipProcessing: Boolean = false

    /**
      * Downloads the web page by sending to Downloader
      *
      * @param worker the downloader worker
      * @param msg    the ms to be sent
      */
    def download(worker: ActorRef, msg: (Int, String)): Unit = {
      downloadingUrls += msg._2
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
      val availableCount = maxWorkers - context.children.size
      var actualCount: Int = requestWorkers
      if (requestWorkers > availableCount) {
        actualCount = availableCount
      }
      // If workers can be created for downloading
      if (actualCount > 0) {
        val workers: Seq[ActorRef] = (1 to actualCount).map(_ => context.actorOf(Props(classOf[Downloader], maxDownloadDuration), s"${classOf[Downloader].getName}_${UUID.randomUUID().toString}"))
        workers foreach context.watch
        workers
      }
      else {
        Iterable.empty[ActorRef]
      }
    }

    /**
      * Tries to download the open url where no worker was available
      */
    def downloadOpenUrls(): Unit = {
      if (!skipProcessing) {
        queuedUrls = queuedUrls.filter(t => !downloadingUrls.contains(t._2)).filter(t => !foundUrls.contains(t._2))
        if (queuedUrls.nonEmpty) {
          val workers: Iterable[ActorRef] = createWorkers(queuedUrls.size)
          if (workers.nonEmpty) {
            val workerIt = workers.iterator
            for (tuple <- queuedUrls) {
              if (workerIt.hasNext) {
                download(workerIt.next(), tuple)
              }
            }
          }
        }
      }
    }

    /**
      * Crawls the links of the given web page
      *
      * @param webPage the web page to handle
      */
    def crawlWebPage(webPage: WebPage): Unit = {
      println(s"Self: ${self.path} \r\n Url: ${webPage.url}")
      foundUrls += webPage.url
      downloadingUrls -= webPage.url

      if (pattern.findFirstIn(webPage.content).nonEmpty) {
        visitedPages += webPage
      }

      if (!skipProcessing) {
        val newUrls: mutable.Set[String] = (mutable.Set.empty[String] ++ webPage.findAbsoluteLinks()) --= foundUrls
        var newOpenUrls = mutable.Set.empty[(Int, String)]

        if (((webPage.level + 1) > maxDepth) || newUrls.isEmpty) {
          if (queuedUrls.nonEmpty) {
            downloadOpenUrls()
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
          queuedUrls ++= newOpenUrls
          if (newOpenUrls.isEmpty && context.children.size < maxWorkers) {
            downloadOpenUrls()
          }
        }
      }
    }

    override def supervisorStrategy: OneForOneStrategy = OneForOneStrategy(maxNrOfRetries = 2, withinTimeRange = Duration.apply(100, TimeUnit.MILLISECONDS)) {
      case _: DownloadException => Stop
      case _ => Escalate
    }

    override def receive: PartialFunction[Any, Unit] = {
      // Starts the crawler by downloading the root url web page
      case Action.START =>
        println(s"RECEIVE START: actor: ${self.path} from ${sender().path}")
        val worker = system.actorOf(Props(classOf[Downloader], maxDownloadDuration), s"${classOf[Downloader].getName}_${UUID.randomUUID().toString}")
        context.watch(worker)
        download(worker, (0, rootUrl))
      // Stops by sending the result to the owner actor
      case Action.STOP =>
        println(s"'RECEIVE STOP: openUrl: ${queuedUrls.size}, visitedUrl: ${foundUrls.size}, currentUrl: ${downloadingUrls.size} actor: ${self.path} from ${sender().path}")
        owner ! foundUrls
      // Soft way, waits for ongoing downloads to finish and handles all the open results, before returning
      case Action.GRACEFUL_SHUTDOWN =>
        skipProcessing = true
      // Hard way, which mostly ends in error, when shutting down the actor system
      case Action.HARD_SHUTDOWN =>
        println(s"'RECEIVE Timeout: actor: ${self.path} from ${sender().path}")
        context.stop(self)
        context.children foreach (child => child ! PoisonPill)
        owner ! Action.STOP
      // Receives a downloaded web page
      case webPage: WebPage =>
        println(s"RECEIVE WebPage: actor: ${self.path} from ${sender().path}")
        crawlWebPage(webPage)
      // When this crawler timeouts we stop gracefully, because termination ends mostly in an error
      case Timeout =>
        self ! Action.GRACEFUL_SHUTDOWN
      // Called each t ime a download actor terminates
      case Terminated(actor) =>
        println(s"'RECEIVE Terminated: children: ${context.children.size}, openUrl: ${queuedUrls.size}, visitedUrl: ${foundUrls.size}, currentUrl: ${downloadingUrls.size}, parent: ${self.path} child-actor: ${actor.path}")
        if (!skipProcessing && queuedUrls.nonEmpty) {
          downloadOpenUrls()
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
  class Downloader(maxDuration: FiniteDuration = 10.seconds) extends BaseActor {

    // Restrict download duration
    context.system.scheduler.scheduleOnce(maxDuration, self, Timeout)

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
          case t: Throwable => println(s"EXCEPTION: ${self.path}, $t")
            throw new DownloadException(s"Download failed: ${t.getMessage}", t)
        } finally {
          context.stop(self)
        }
      case Timeout =>
        println(s"'RECEIVE Timeout: actor: ${self.path} from ${sender().path}")
        context.stop(self)
    }
  }

  class ApplicationDownloaderMain extends BaseActor {
    val downloader: ActorRef = system.actorOf(Props(classOf[Downloader], 10.seconds), s"${classOf[Downloader].getName}")

    context.system.scheduler.scheduleOnce(240.seconds, self, Timeout)

    override def receive: PartialFunction[Any, Unit] = {
      case Action.START =>
        downloader ! "http://www.google.at"
      case Action.GRACEFUL_SHUTDOWN =>
        context.stop(downloader)
        context.stop(self)
        Await.result(Http().shutdownAllConnectionPools(), Duration.apply(5, TimeUnit.SECONDS))
        materializer.shutdown()
        Await.result(context.system.terminate(), Duration.apply(5, TimeUnit.SECONDS))
      case WebPage(_, url, _) =>
        println(s"Url: $url \r\n content: ")
        self ! Action.GRACEFUL_SHUTDOWN
      case Terminated(actor) =>
        println(s"'RECEIVE Terminated: parent: ${self.path} child-actor: ${actor.path}")
    }
  }

  class ApplicationCrawlerMain extends BaseActor {
    val crawler: ActorRef = system.actorOf(Props(classOf[Crawler], self, "https://www.fh-ooe.at/campus-hagenberg/", 7, 32, ".*.".r), s"${classOf[Crawler].getName}")

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

  /**
    * Application for testing the search engine which uses crawlers to search for texts in linked web pages
    */
  class ApplicationSearchEngine(maxWorkers: Int) extends BaseActor with SearchEngine with TimeUtil {

    var crawler: ActorRef = _
    var shutdownPromise: Promise[Terminated] = Promise()
    var requests: util.Map[String, (ActorRef, Promise[Set[WebPage]])] = new util.HashMap()
    var shutdownInProgress: Boolean = false
    var counter: Int = 0

    /**
      * Performs the system shutdown which will stop the actor system as well
      */
    private def systemShutdown(): Unit = {
      context.stop(self)
      searchEngines remove self.path.name
      // Last one turns of the light :)
      if (searchEngines.isEmpty) {
        Await.ready(Http().shutdownAllConnectionPools(), Duration.apply(5, TimeUnit.SECONDS))
        materializer.shutdown()
        Await.ready(context.system.terminate(), Duration.apply(5, TimeUnit.SECONDS))
      }
      shutdownInProgress = false

      shutdownPromise.success(null)
    }

    override def search(url: String, pattern: Regex, depth: Int, maxDuration: FiniteDuration): Future[Set[WebPage]] = {
      // Create crawler for search
      counter += 1
      val crawler: ActorRef = system.actorOf(Props(classOf[Crawler], self, url, depth, maxWorkers, pattern), s"${self.path.name}_${classOf[Crawler].getName}_$counter")
      // Register timeout
      context.system.scheduler.scheduleOnce(maxDuration, crawler, Timeout)
      // watch teh crawler
      context.watch(crawler)
      // Put crawler into map
      requests.put(crawler.path.name, (crawler, Promise()))

      // Send start request to crawler
      crawler ! Action.START
      // Return promise future
      requests.get(crawler.path.name)._2.future
    }

    override def shutdown(): Future[Terminated] = {
      // Send break request to crawlers
      requests forEach ((_, value) => value._1 ! Action.GRACEFUL_SHUTDOWN)
      // mark shutdown in progress
      shutdownInProgress = true
      // Return promise future
      shutdownPromise.future
    }

    override def supervisorStrategy: OneForOneStrategy = OneForOneStrategy(maxNrOfRetries = 2, withinTimeRange = Duration.apply(1, TimeUnit.SECONDS)) {
      case _ => Resume
    }

    override def receive: PartialFunction[Any, Unit] = {
      case "test_multiple_downloads_in_time" => test_multiple_downloads_in_time()
      case "test_multiple_downloads_timeout" => test_multiple_downloads_timeout()
      case Action.GRACEFUL_SHUTDOWN =>
        shutdownInProgress = true
      case pages: mutable.Set[WebPage] =>
        println(s"'RECEIVE result: actor: ${self.path} from ${sender().path}")
        context.stop(sender())
        // Set promise success, after result returned
        if (requests.containsKey(sender().path.name)) {
          requests.remove(sender().path.name)._2.success(pages.toSet)
        }
      case Timeout =>
        println(s"'RECEIVE Timeout: actor: ${self.path} from ${sender().path}")
        // This actor timeouts, all searches must stop, but engine stays alive
        requests forEach ((_, value) => value._1 ! Timeout)
      case Terminated(actor) =>
        println(s"'RECEIVE Terminated: parent: ${self.path} child-actor: ${actor.path}")
        // Fail promise if search terminates without having returned result before
        if (requests.containsKey(actor.path.name)) {
          requests.remove(actor.path.name)._2.failure(new IllegalStateException("Search did not return any result"))
        }
        // System shuts down, after all searches have stopped and shutdown request is present
        if (shutdownInProgress && requests.isEmpty) {
          systemShutdown()
        }
    }

    def test_multiple_downloads_in_time(): Unit = {
      val now = System.currentTimeMillis()
      search("https://www.fh-ooe.at/campus-hagenberg/", ".*.".r, 3, 5.minutes) onComplete {
        case Success(data) =>
          println(s"test_multiple_downloads_in_time\n\t 1. Search completed: \n\t Url: 'https://www.fh-ooe.at/campus-hagenberg/' \n\t size: ${data.size}  \n\t result: $data  \n\t seconds: ${differenceInSeconds(now)}")
        case Failure(ex) =>
          println(s"test_multiple_downloads_in_time\n\t 1. Search failed: $ex  \n\t seconds: ${differenceInSeconds(now)}")
      }
      search("https://www.google.at/", ".*.".r, 3, 5.minutes) onComplete {
        case Success(data) =>
          println(s"test_multiple_downloads_in_time\n\t 2. Search completed: \n\t Url: 'https://www.google.at/' \n\t size: ${data.size}  \n\t result: $data  \n\t seconds: ${differenceInSeconds(now)}")
        case Failure(ex) =>
          println(s"test_multiple_downloads_in_time\n\t 2. Search failed: $ex  \n\t seconds: ${differenceInSeconds(now)}")
      }
    }

    def test_multiple_downloads_timeout(): Unit = {
      val now = System.currentTimeMillis()

      search("https://www.fh-ooe.at/campus-hagenberg/", ".*.".r, 10, 5.seconds) onComplete {
        case Success(data) =>
          println(s"test_multiple_downloads_timeout\n\t 1. Search completed: \n\t Url: 'https://www.fh-ooe.at/campus-hagenberg/' \n\t size: ${data.size} \n\t result: $data \n\t seconds: ${differenceInSeconds(now)}")
        case Failure(ex) =>
          println(s"test_multiple_downloads_timeout\n\t 1. Search failed: $ex  \n\t seconds: ${differenceInSeconds(now)}")
      }
      search("https://www.google.at/", ".*.".r, 10, 10.seconds) onComplete {
        case Success(data) => println(s"test_multiple_downloads_timeout\n\t 2. Search completed: \n\t Url: 'https://www.google.at/' \n\t size: ${data.size} \n\t result: $data \n\t seconds: ${differenceInSeconds(now)}")
        case Failure(ex) => println(s"test_multiple_downloads_timeout\n\t 2. Search failed: $ex  \n\t seconds: ${differenceInSeconds(now)}")
      }
    }
  }

  val app1: ActorRef = system.actorOf(Props(classOf[ApplicationSearchEngine], 8), s"${
    classOf[ApplicationSearchEngine].getName
  }_test_multiple_downloads_in_time")
  searchEngines add app1.path.name
  app1 ! "test_multiple_downloads_in_time"
  app1 ! Action.GRACEFUL_SHUTDOWN

  val app2: ActorRef = system.actorOf(Props(classOf[ApplicationSearchEngine], 8), s"${
    classOf[ApplicationSearchEngine].getName
  }_test_multiple_downloads_timeout")
  searchEngines add app2.path.name
  app2 ! "test_multiple_downloads_timeout"
  app2 ! Action.GRACEFUL_SHUTDOWN
}
