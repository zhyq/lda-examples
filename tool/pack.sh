# ./pack.sh idf
rm ${1}.jar
#scalac -classpath /usr/local/spark-1.6.1-bin-hadoop2.6/lib/spark-assembly-1.6.1-hadoop2.6.0.jar ${1}".scala"
#scalac -classpath :/usr/local/spark-2.0.0-bin-hadoop2.7/jars/*  ${1}".scala"
scalac -classpath :/usr/local/spark-2.0.0-bin-hadoop2.7/jars/*:/home/research/hanleyzhang/tool/jars/scopt_2.10-3.2.0.jar  ${1}".scala"

jar -cvf ${1}".jar" *.class
rm *.class
