name := "mvp-ue02"
version := "0.1"
scalaVersion := "2.12.4"

val akkaVersion = "2.5.7"
val akkaHttpVersion = "10.0.11"
val akkaFileVersion="0.15"
val jacksonJsonVersion = "10.1.0-RC1"

libraryDependencies += "com.typesafe.akka" % "akka-actor_2.12" % akkaVersion
libraryDependencies += "com.typesafe.akka" % "akka-stream_2.12" % akkaVersion
libraryDependencies += "com.typesafe.akka" %% "akka-http-core" % akkaHttpVersion
libraryDependencies += "com.lightbend.akka" %% "akka-stream-alpakka-file" % akkaFileVersion
libraryDependencies += "com.typesafe.akka" %% "akka-http-jackson" % jacksonJsonVersion
