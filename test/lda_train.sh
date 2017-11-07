#!/usr/bin/env sh
cur_path=$(cd `dirname $0`; pwd)

#spark-submit --class segment --master yarn-cluster --num-executors 40 --driver-memory 30g --executor-memory 10g --executor-cores 15 --driver-java-options "-Dspark.akka.frameSize=520" \
#$cur_path/segment.jar  \
#hdfs://cluster-infosec-base//dwtmp/user/hanleyzhang/custom_topic_test2 \
#hdfs://cluster-infosec-base/user/infosec/user/infosec/textprocess/data/vocab/FC_20160724_wash_segout_trim_vocab_spark \
#hdfs://cluster-infosec-base/user/infosec/user/infosec/textprocess/out/new715 \
#hdfs://cluster-infosec-base/user/infosec/user/infosec/textprocess/out/new_google_715

modelDir="/dwtmp/user/hanleyzhang/zhishu/lda_model"
vocabDir="/dwtmp/user/hanleyzhang/zhishu/lda_vocab"
topicDistributionsDir="/dwtmp/user/hanleyzhang/zhishu/lda_topic_distribution"
topicWordsDir="/dwtmp/user/hanleyzhang/zhishu/lda_topic_words"
topTopicsPerDocument="/dwtmp/user/hanleyzhang/zhishu/topTopicsPerDocument"
source /usr/local/spark-2.0.0-bin-hadoop2.7/conf/spark-env.sh
#/usr/local/spark-2.0.0-bin-hadoop2.7/bin/spark-submit --class LDAExample --master yarn-cluster --queue root.platform.radar \
 #/home/research/hanleyzhang/sparktextprocess/spark_topmodle/LDATopModel.jar \
hadoop fs -rm -r $modelDir $vocabDir $topicDistributionsDir $topicWordsDir $topTopicsPerDocument
/usr/local/spark-2.0.0-bin-hadoop2.7/bin/spark-submit --class LDATopModel --master yarn-cluster --num-executors 60 --driver-memory 30g --executor-memory 20g --executor-cores 8 \
--jars /home/research/hanleyzhang/tool/jars/scopt_2.10-3.2.0.jar \
 /home/research/hanleyzhang/code/LDATopModel.jar \
 --actionMode "train" \
 --topicConcentration 1.1 \
 --maxWordForEachTopic 200 \
 --modelDir $modelDir \
 --vocabDir $vocabDir \
 --topicDistributionsDir $topicDistributionsDir \
 --topicWordsDir $topicWordsDir \
 --k 10 \
 --vocabSize 64285 \
 --topKTopicsPerDocument 1 \
 --topTopicsPerDocument $topTopicsPerDocument \
 --stopwordFile /dwtmp/user/hanleyzhang/zhishu/stopwords \
 --corpusDir /dwtmp/user/hanleyzhang/zhishu/segment 
 #--corpusDir /dwtmp/user/hanleyzhang/zhishu/gugong_segment 
 #--corpusDir /dwtmp/user/hanleyzhang/zhishu/rand_segment
# --corpusDir /dwtmp/user/hanleyzhang/zhishu/segment 
