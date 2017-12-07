name := "mvp-ue02"

version := "1.0"

scalaVersion := "2.12.4"

val akkaVersion = "2.5.7"
val akkaHttpVersion = "10.0.10"

libraryDependencies += "com.typesafe.akka" %% "akka-actor_2.12" % akkaVersion
libraryDependencies += "com.typesafe.akka" %% "akka-stream_2.12" % akkaVersion
libraryDependencies += "com.typesafe.akka" %% "akka-http-core_2.12" % akkaHttpVersion
