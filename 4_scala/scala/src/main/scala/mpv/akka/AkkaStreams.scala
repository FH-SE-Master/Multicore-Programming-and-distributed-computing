package mpv.akka

import java.nio.file.{FileSystem, FileSystems, Path, Paths}

import akka.NotUsed
import akka.stream.scaladsl.Source

import scala.util.matching.Regex

/**
  * @author Thomas Herzog <herzog.thomas81@gmail.com>
  * @since 12/17/17
  */
object AkkaStreams extends App {

  class MatchingLine(path: Path, lineNr: Int) {
  }

  def filesAndDirectories(path: String): Source[Path, NotUsed] = {
    def filesAndDirectoriesRec(path: String, fs: FileSystem, acc: List[Path]): List[Path] = {
      if (path != null) {
        val currentPath = fs.getPath(path)
        val currentFile = currentPath.toFile
        println(s"Visiting: $currentPath")

        if (currentFile.isDirectory && currentFile.listFiles() != null) {
          for (file <- currentFile.listFiles()) {
            return filesAndDirectoriesRec(file.getAbsolutePath, fs, acc :+ currentPath)
          }
        }
        return filesAndDirectoriesRec(null, fs, acc :+ currentPath)
      }

      acc
    }

    Source(filesAndDirectoriesRec(path, FileSystems.getDefault, List.empty[Path]))
  }

  def matchingFiles(path: String, fileExt: Seq[String]): Source[Path, NotUsed] = {
    filesAndDirectories(path).filter(p => fileExt.contains(p.getFileName.toString.split('.').last))
  }

  def matchingLines(fileName: String, regex: Regex): Source[MatchingLine, NotUsed] = {
    val matchingLines: List[MatchingLine] = List.empty[MatchingLine]
    val file = Paths.get(fileName).toFile
    if (file.isFile) {
      var lineNr = 0
      for (line <- io.Source.fromFile(fileName).getLines) {
        lineNr += 1
        if (regex.findFirstIn(line).isDefined) {
          matchingLines :+ new MatchingLine(Paths.get(line), lineNr)
        }
      }
    }

    Source(matchingLines)
  }

  def grep(path: String, fileExt: Seq[String], pattern: Regex): Source[MatchingLine, NotUsed] = {
    null
  }

  def test_filesAndDirectories_success(): Unit = {
    val paths = filesAndDirectories("/home/het/repositories/github/FH-SE-Master/Multicore-Programming-and-distributed-computing/4_scala/")
  }

  test_filesAndDirectories_success()
}
