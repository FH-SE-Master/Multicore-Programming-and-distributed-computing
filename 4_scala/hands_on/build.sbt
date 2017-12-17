name := "Scala"

version := "0.1"

scalaVersion := "2.12.4"

val akkaVersion = "2.5.7"

libraryDependencies += "com.typesafe.akka" % "akka-actor_2.12" % akkaVersion
libraryDependencies += "com.typesafe.akka" % "akka-stream_2.12" % akkaVersion