package mpv.akka

import java.nio.file.{FileSystems, Path}

import akka.NotUsed
import akka.actor.ActorSystem
import akka.stream.{ActorMaterializer, IOResult}
import akka.stream.alpakka.file.scaladsl.Directory
import akka.stream.scaladsl.{FileIO, Flow, Framing, Sink, Source}
import akka.util.ByteString
import java.nio.file.Paths

import scala.concurrent.{Await, Future}
import scala.concurrent.Future
import scala.concurrent.duration.Duration
import scala.util.{Failure, Success}
import scala.util.matching.Regex
import scala.concurrent.ExecutionContext.Implicits.global

/**
  * @author Thomas Herzog <herzog.thomas81@gmail.com>
  * @since 12/17/17
  */
object AkkaStreams extends App {

  //region Shell methods
  class MatchingLine(path: Path, lineNr: Long, line: String) {
    override def toString: String = {
      s"File: ${path.toFile.getAbsolutePath}, lineNr: $lineNr, line: $line"
    }
  }

  def showThreadMetadata(func: String): Unit = {
    println(s"func: '$func' on thread '${Thread.currentThread().getId}'")
  }

  def filesAndDirectories(directory: String): Source[Path, NotUsed] = {
    showThreadMetadata("filesAndDirectories")
    val path: Path = FileSystems.getDefault.getPath(directory)
    println(s"Walk directory: '${path.toFile.getAbsolutePath}'")
    Directory.walk(path)
  }

  def matchingFiles(path: String, fileExt: Seq[String]): Source[Path, NotUsed] = {
    def extractExtension(path: Path): String = {
      val split = path.getFileName.toString.split('.')
      if (split.length > 0) {
        return "." + split.last
      }
      ""
    }

    filesAndDirectories(path)
      .filter(path => {
        val file = path.toFile
        file.isFile && file.canRead
      })
      .filter(path => fileExt.contains(extractExtension(path)))
  }

  def matchingLines(path: Path, pattern: Regex): Source[MatchingLine, Future[IOResult]] = {
    showThreadMetadata("matchingLines")
    FileIO.fromPath(path)
      .via(Framing.delimiter(ByteString(System.lineSeparator()), Int.MaxValue, allowTruncation = true).map(_.utf8String))
      .zipWithIndex
      .filter((str: (String, Long)) => pattern.findFirstIn(str._1).isDefined)
      .map(x => new MatchingLine(path, x._2, x._1))
  }

  def grep(dir: String, ext: Seq[String], pattern: Regex, parallel: Boolean = false): Source[MatchingLine, NotUsed] = {
    if (parallel)
      matchingFiles(dir, ext).async.flatMapConcat(x => matchingLines(x, pattern))
    else
      matchingFiles(dir, ext).flatMapConcat(x => matchingLines(x, pattern))
  }

  //endregion
  implicit val system: ActorSystem = ActorSystem("mySystem")
  implicit val materializer: ActorMaterializer = ActorMaterializer()
  val CPU_COUNT = Runtime.getRuntime.availableProcessors()
  val FORMATTER = java.text.NumberFormat.getIntegerInstance

  def execute[T](f: Future[T]) {
    val start: Long = System.currentTimeMillis()
    f onComplete {
      case Success(_) => println(s"-> SUCCESS")
      case Failure(ex) => println(s"-> FAILURE: EXCEPTION: $ex")
    }
    Await.ready(f, Duration.Inf)
    val duration = System.currentTimeMillis() - start
    println("Duration: %s ms".format(FORMATTER.format(duration)))
  }

  def test_filesAndDirectories(): Unit = {
    println("Start: test_filesAndDirectories")

    execute(filesAndDirectories("./").runWith(Sink.foreachParallel[Path](CPU_COUNT)(println(_))))

    println("End: test_filesAndDirectories")
  }

  def test_matchingFiles(): Unit = {
    println("Start: test_matchingFiles")

    execute(matchingFiles("./", List(".scala", ".sbt")).runWith(Sink.foreachParallel[Path](CPU_COUNT)(println(_))))

    println("End: test_matchingFiles")
  }

  def test_grep(): Unit = {
    println("Start: test_grep")

    execute(grep("./", List(".scala"), new Regex("import .*")).runWith(Sink.foreachParallel[MatchingLine](CPU_COUNT)(println(_))))

    println("Start: test_grep")
  }

  println("===> Starting tests <===========================")
  test_filesAndDirectories()
  println("================================================")
  test_matchingFiles()
  println("================================================")
  test_grep()
  Thread.sleep(1000)
  println("===> End tests <===========================")

  Thread.sleep(1000)
  system.terminate()
}
