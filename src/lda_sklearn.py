#!/usr/bin/env python

from gensim import corpora
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation,NMF

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()
n_samples = 2000
n_features = 1000
n_components = 10
n_top_words = 20


with open('gugong1.txt') as f:
    text = f.read()


corpus = [text]
cntVector = CountVectorizer()
#tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
#                                   max_features=n_features,
#                                   stop_words='english')
#tfidf = tfidf_vectorizer.fit_transform(corpus)
cntTf = cntVector.fit_transform(corpus)
lda = LatentDirichletAllocation(n_topics = 20, learning_offset=50., random_state=0)
docers = lda.fit_transform(cntTf)
feature_name = cntVector.get_feature_names()
print_top_words(lda,feature_name,n_top_words)
#print docers
#print lda.components_

