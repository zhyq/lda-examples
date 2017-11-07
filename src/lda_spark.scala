/*
 * LDA training
 * zhangyunquan@hotmail.com
 * date:20161116
 * ver 1.0
 */

// scalastyle:off println
//package org.apache.spark.examples.mllib

import org.apache.log4j.{Level, Logger}
import org.apache.spark.broadcast.Broadcast
import scopt.OptionParser
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel, RegexTokenizer, StopWordsRemover}
import org.apache.spark.ml.linalg.{Vector => MLVector}
import org.apache.spark.mllib.clustering.{DistributedLDAModel, EMLDAOptimizer, LDA, OnlineLDAOptimizer}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, SparkSession}
import java.util.NoSuchElementException


object LDATopModel {

  case class Docs(
                   id:String="",
                   content:String=""
                 )

  /**
    *
    * @param corpusDir [string] 语料输入 支持多个路径，用逗号可隔开
    * @param k [int] 聚类数目
    * @param modelDir [string] 保存和加载模型的路径
    * @param actionMode [string] "train" 训练 "predict" 预测
    * @param vocabDir [string] 保存和加载词典的路径
    * @param topicDistributionsDir [string]
    * @param topTopicsPerDocument [string] 
    * @param topKTopicsPerDocument [int]
    * @param topicWordsDir
    * @param vocabSize [int]支持训练的词的总大小
    * @param stopwordFile [string]停止词路径
    * @param algorithm [string] "em" EMLDAOptimizer "online" onlineLDAOptimizer
    * @param maxWordForEachTopic
    * @param maxIterations [int] 最大的执行结点数
    * @param docConcentration
    * @param topicConcentration
    * @param checkpointDir
    * @param checkpointInterval
    */
  case class Params(
                     corpusDir: Seq[String] = Seq.empty,
                     k: Int = 20,
                     modelDir: String = "",
                     vocabDir: String = "",
                     topicDistributionsDir: String = "",
                     topicWordsDir : String = "",
                     actionMode : String = "train",
                     maxWordForEachTopic : Int = 10,
                     maxIterations: Int = 10,
                     docConcentration: Double = 5,
                     topicConcentration: Double = 5,
                     topTopicsPerDocument : String = "",
                     topKTopicsPerDocument : Int = 1,
                     vocabSize: Int = 10000,
                     stopwordFile: String = "",
                     algorithm: String = "em",
                     checkpointDir: Option[String] = None,
                     checkpointInterval: Int = 10
                   )

  // 参数解析
  def paramParser(args: Array[String]):Params = {
    val defaultParams = Params()
    val parser = new OptionParser[Params]("InfosecLDA") {
      head("Infosec nlp LDA: LDA app for plain text data.")
      opt[Int]("k")
        .text(s"number of topics. default: ${defaultParams.k}")
        .action((x, c) => c.copy(k = x))
      opt[String]("modelDir")
        .text(s"modelDir. Save or load model path." +
          s" default: ${defaultParams.modelDir}")
        .action((x, c) => c.copy(modelDir = x))
      opt[String]("vocabDir")
        .text(s"vocabDir. Save or load vocab path." +
          s" default: ${defaultParams.vocabDir}")
        .action((x, c) => c.copy(vocabDir = x))

      opt[String]("actionMode")
        .text(s"actionMode. predict or train." +
          s" default: ${defaultParams.actionMode}")
        .action((x, c) => c.copy(actionMode = x))

      opt[String]("topicDistributionsDir")
        .text(s"topicDistributionsDir. save the result of the top documents for each topic and the corresponding weight of the topic in the documents " +
          s" default: ${defaultParams.topicDistributionsDir}")
        .action((x, c) => c.copy(topicDistributionsDir = x))
      opt[String]("topicWordsDir")
        .text(s"topicWordsDir. Save the result of topics arrays of eact terms weight." +
          s" default: ${defaultParams.topicWordsDir}")
        .action((x, c) => c.copy(topicWordsDir = x))

      opt[Int]("maxIterations")
        .text(s"number of iterations of learning. default: ${defaultParams.maxIterations}")
        .action((x, c) => c.copy(maxIterations = x))
      opt[Int]("maxWordForEachTopic")
        .text(s"number of words for each topic. default: ${defaultParams.maxWordForEachTopic}")
        .action((x, c) => c.copy(maxWordForEachTopic = x))
      opt[Double]("docConcentration")
        .text(s"amount of topic smoothing to use (> 1.0) (-1=auto)." +
          s"  default: ${defaultParams.docConcentration}")
        .action((x, c) => c.copy(docConcentration = x))
      opt[Double]("topicConcentration")
        .text(s"amount of term (word) smoothing to use (> 1.0) (-1=auto)." +
          s"  default: ${defaultParams.topicConcentration}")
        .action((x, c) => c.copy(topicConcentration = x))
      opt[Int]("vocabSize")
        .text(s"number of distinct word types to use, chosen by frequency. (-1=all)" +
          s"  default: ${defaultParams.vocabSize}")
        .action((x, c) => c.copy(vocabSize = x))
      opt[String]("stopwordFile")
        .text(s"filepath for a list of stopwords. Note: This must fit on a single machine." +
          s"  default: ${defaultParams.stopwordFile}")
        .action((x, c) => c.copy(stopwordFile = x))
      opt[String]("algorithm")
        .text(s"inference algorithm to use. em and online are supported." +
          s" default: ${defaultParams.algorithm}")
        .action((x, c) => c.copy(algorithm = x))
      opt[String]("checkpointDir")
        .text(s"Directory for checkpointing intermediate results." +
          s"  Checkpointing helps with recovery and eliminates temporary shuffle files on disk." +
          s"  default: ${defaultParams.checkpointDir}")
        .action((x, c) => c.copy(checkpointDir = Some(x)))
      opt[Int]("checkpointInterval")
        .text(s"Iterations between each checkpoint.  Only used if checkpointDir is set." +
          s" default: ${defaultParams.checkpointInterval}")
        .action((x, c) => c.copy(checkpointInterval = x))

      
      opt[String]("topTopicsPerDocument")
        .text(s"topTopicsPerDocument. Save the result of top K topics per Document." +
          s"default: ${defaultParams.topTopicsPerDocument}")
        .action((x, c) => c.copy(topTopicsPerDocument = x))
      
      opt[Int]("topKTopicsPerDocument")
        .text(s"topKTopicsPerDocument " +
          s" default: ${defaultParams.topKTopicsPerDocument}")
        .action((x, c) => c.copy(topKTopicsPerDocument = x))
      
      opt[String]("corpusDir")
        .text("corpusDir paths (directories) to plain text corpora." +
          "  Each text file line should hold 1 document.")
        .action((x, c) => c.copy(corpusDir = c.corpusDir :+ x))
    }
    parser.parse(args , defaultParams)  match {
      case Some(params) => return params
      case _ => return defaultParams
    }
  }


  def main(args: Array[String]) {
    val params = paramParser(args)
    if(params.actionMode == "predict"){
      predict(params)
    }
    else {
      train(params)
    }
  }



  //预测模型
  def predict(params:Params):Unit={
    val conf = new SparkConf().setAppName(s"LDA pridict with $params").set("spark.hadoop.validateOutputSpecs","false")
    val sc = new SparkContext(conf)
    Logger.getRootLogger.setLevel(Level.WARN)

    val documents = predictpreprocess(sc,params)

    //predict
    val disTrainModel = DistributedLDAModel.load(sc,params.modelDir)
    val localModel = disTrainModel.toLocal
    val predictResult = localModel.topicDistributions(documents)
    predictResult.map({case(id,vec)=>id.toString + " " + vec.toString}).saveAsTextFile(params.topicDistributionsDir)
  }

  // 训练
  def train(params: Params): Unit = {
    val conf = new SparkConf().setAppName(s"InfosecLDA with $params").set("spark.hadoop.validateOutputSpecs","false")
    val sc = new SparkContext(conf)

    Logger.getRootLogger.setLevel(Level.WARN)

    // 1 Load documents, and prepare them for LDA.
    val preprocessStart = System.nanoTime()
    val (corpus, vocabArray, actualNumTokens) =
    preprocess(sc, params.corpusDir, params.vocabSize, params.stopwordFile)
    corpus.cache()
    val actualCorpusSize = corpus.count()
    val actualVocabSize = vocabArray.length
    val preprocessElapsed = (System.nanoTime() - preprocessStart) / 1e9

    println()
    println(s"Corpus summary:")
    println(s"\t Training set size: $actualCorpusSize documents")
    println(s"\t Vocabulary size: $actualVocabSize terms")
    println(s"\t Training set size: $actualNumTokens tokens")
    println(s"\t Preprocessing time: $preprocessElapsed sec")
    println()

    //2  Run LDA.
    val lda = new LDA()

    val optimizer = params.algorithm.toLowerCase match {
      case "em" => new EMLDAOptimizer
      // add (1.0 / actualCorpusSize) to MiniBatchFraction be more robust on tiny datasets.
      case "online" => new OnlineLDAOptimizer().setMiniBatchFraction(0.05 + 1.0 / actualCorpusSize)
      case _ => throw new IllegalArgumentException(
        s"Only em, online are supported but got ${params.algorithm}.")
    }

    lda.setOptimizer(optimizer)
      .setK(params.k)
      .setMaxIterations(params.maxIterations)
      .setDocConcentration(params.docConcentration)
      .setTopicConcentration(params.topicConcentration)
      .setCheckpointInterval(params.checkpointInterval)
    if (params.checkpointDir.nonEmpty) {
      sc.setCheckpointDir(params.checkpointDir.get)
    }
    val startTime = System.nanoTime()
    val ldaModel = lda.run(corpus)
    val elapsed = (System.nanoTime() - startTime) / 1e9

    println(s"Finished training LDA model.  Summary:")
    println(s"\t Training time: $elapsed sec")

    //3 save model

    if(params.modelDir != ""){
      ldaModel.save(sc,params.modelDir)
    }

    //4 save vocab
    if(params.vocabDir != "") {
      saveVocab(vocabArray, sc, params)
    }
    //5 doc 用 topic
    if (ldaModel.isInstanceOf[DistributedLDAModel] && params.topicDistributionsDir != "") {
      val distLDAModel = ldaModel.asInstanceOf[DistributedLDAModel]
      val avgLogLikelihood = distLDAModel.logLikelihood / actualCorpusSize.toDouble
      println(s"\t Training data average log likelihood: $avgLogLikelihood")
      println()

      val distRdd = distLDAModel.topicDistributions
      distRdd.map({case(id,vec)=>id.toString + " " + vec.toString}).saveAsTextFile(params.topicDistributionsDir)
    }


    //6 Print the topics, showing the top-weighted terms for each topic.
    if(params.topicWordsDir != "") {
      val topicIndices = ldaModel.describeTopics(maxTermsPerTopic = params.maxWordForEachTopic)
      val topics = topicIndices.map { case (terms, termWeights) =>
        terms.zip(termWeights).map { case (term, weight) => (vocabArray(term.toInt), weight) }
      }
      val indexTopics = topics.zipWithIndex;
      sc.parallelize(indexTopics).map(_.swap).map { case (topic, words) => words.map({ case (word, weight) => topic.toString + " " + word.toString + " " + weight.toString }) }.map(x=>x.mkString).saveAsTextFile(params.topicWordsDir)

      println(s"${params.k} topics:")
      topics.zipWithIndex.foreach { case (topic, i) =>
        println(s"TOPIC $i")
        topic.foreach { case (term, weight) =>
          println(s"$term\t$weight")
        }
        println()
      }

    }

    //7 print topTopicsPerDocument
    if(ldaModel.isInstanceOf[DistributedLDAModel] && params.topTopicsPerDocument != "") {
        val distLDAModel = ldaModel.asInstanceOf[DistributedLDAModel]
        val tmpLda = distLDAModel.topTopicsPerDocument(params.topKTopicsPerDocument).map {
            f =>
            (f._1, f._2 zip f._3)
            }.map(f => s"${f._1}\t${f._2.map(k => k._1 + "\t" + k._2).mkString("\t")}").repartition(1).saveAsTextFile(params.topTopicsPerDocument)
        //val tmpLda = distLDAModel.topTopicsPerDocument(params.topKTopicsPerDocument).map({case (docId , topics) => topics.map({case(indice,weight) => docId.toString +" " + indice.toString + " " + weight.toString})}).saveAsTextFile(params.topKTopicsPerDocument)
    }
    sc.stop()
  }

  def saveVocab(vocab:Array[String],sc:SparkContext,params: Params)  {
    //val indexVocab = vocab.zipWithIndex
    var indexVocab:Array[Tuple2[String,Int]]= new Array (vocab.length)
    var index=0
    for (i <- 0 until vocab.length){
      indexVocab(index)= (vocab(i),i)
      index = index+1
    }
    sc.parallelize(indexVocab).map(_.toString().replaceAll("[()]","").replaceAll("[,]"," ")).saveAsTextFile(params.vocabDir)
  }


  /**
    * Load documents, tokenize them, create vocabulary, and prepare documents as term count vectors.
    *
    * @return (corpus, vocabulary as array, total token count in corpus)
    */
  private def preprocess(
                          sc: SparkContext,
                          paths: Seq[String],
                          vocabSize: Int,
                          stopwordFile: String): (RDD[(Long, Vector)], Array[String], Long) = {

    val spark = SparkSession
      .builder
      //  .sparkContext(sc)
      .getOrCreate()
    import spark.implicits._

  
    val df = sc.textFile(paths.mkString(",")).filter(line=>line.split(" |\t").length >= 2).map(_.split(" |\t",2)).map(attributes=>Docs(attributes(0).trim,attributes(1))).toDF("did","docs")
    val customizedStopWords: Array[String] = if (stopwordFile.isEmpty) {
      Array.empty[String]
    } else {
      val stopWordText = sc.textFile(stopwordFile).collect()
      stopWordText.flatMap(_.stripMargin.split("\\s+"))
    }
    val tokenizer = new RegexTokenizer()
      .setInputCol("docs")
      .setOutputCol("rawTokens")
    val stopWordsRemover = new StopWordsRemover()
      .setInputCol("rawTokens")
      .setOutputCol("tokens")
    stopWordsRemover.setStopWords(stopWordsRemover.getStopWords ++ customizedStopWords)
    val countVectorizer = new CountVectorizer()
      .setVocabSize(vocabSize)
      .setInputCol("tokens")
      .setOutputCol("features")

    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, stopWordsRemover, countVectorizer))

    val model = pipeline.fit(df)
    val documents = model.transform(df)
      .select("did","features")
      .rdd
      //.map { case Row(features: MLVector) =>Vectors.fromML(features) }
      .map { case Row(id,features: MLVector) =>(id.toString.toLong, Vectors.fromML(features)) }
    //.zipWithIndex()
    //.map(_.swap)

    (documents,
      model.stages(2).asInstanceOf[CountVectorizerModel].vocabulary,  // vocabulary
      documents.map(_._2.numActives).sum().toLong) // total token count
  }

  def predictpreprocess(sc: SparkContext,params: Params):(RDD[(Long, Vector)])={
    //load vocabulary

    val vocabtext = sc.textFile(params.vocabDir).map(line => line.split(" "))
    val arr:Array[(String,String)]= sc.textFile(params.vocabDir).map(line => (line.split(" ")(0),line.split(" ")(1))).collect()
    val vocab:Map[String,String] = arr.toMap
    val vocab_broad = sc.broadcast(vocab)

    // transform
    val spark = SparkSession
      .builder
      .getOrCreate()
    import spark.implicits._
    val df = sc.textFile(params.corpusDir.mkString(",")).filter(line=>line.split(" |\t").length >= 2).map(_.split(" |\t",2)).map(attributes=>Docs(attributes(0).trim,attributes(1))).toDF("id","content")
    val documents = df
      .select("id","content")
      .rdd
      .map { case Row(id,content: String) =>(id.toString.toLong, trans(content,params.vocabSize,vocab_broad))}
    return documents
  }

  // convert text to vector
  def trans(str:String,vocabsize:Int,vocab_broad: Broadcast[Map[String,String]]):Vector={
    // 文章的文字数组 中国 美国 上海

    val text_array = str.split(" ")
    val index_array =new Array[Int] (text_array.length)
    //初始化为-1 为0 为导致误以为是vocab数组的下标为0的词
    for (i <- 0 until text_array.length ) {
      index_array(i) = -1;
    }
    val regex="""^\d+$""".r
    //文章的数字数组  中国 美国 上海 对应的 vocab 下标
    for (i <- 0 until text_array.length){
      try {
        if (regex.findFirstMatchIn(vocab_broad.value(text_array(i))) != None) {
          if (vocab_broad.value(text_array(i)).toInt >= 0 && vocab_broad.value(text_array(i)).toInt < vocabsize)
            index_array(i) = vocab_broad.value(text_array(i)).toInt
        }
      }catch{
        case e: NoSuchElementException => println(s"not in map ")
      }
    }

    //文章的向量
    val vector_array = new Array[Double] (vocabsize)
    for(arg <- index_array){
      if(arg >=0) {
        vector_array(arg) += 1
      }
    }

    val dv:Vector = Vectors.dense(vector_array)
    val dv2:Vector = dv.toSparse
    /*
    println("use map:")
    println(vocab_broad.value)
    println(s" doc:$str")
    println(s"text_array")
    for(arg <- text_array){
      print(s" $arg")
    }
    println(" ")
    println(s"index_array ")
    for(arg <- index_array){
      print(s" $arg")
    }
    println(" ")
    println(s"vector_array ")
    for(arg <- vector_array){
      print(s" $arg")
    }
    println(" ")
    println(s"vector $dv")
    val dv1: Vector = Vectors.dense(1,2,3,4,7) 
    */
    return dv2
  }
}
// scalastyle:on println
