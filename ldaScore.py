from __future__ import print_function, division

__author__ = 'amrit'

import sys

sys.dont_write_bytecode = True
from random import shuffle, seed
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer
import copy
from sklearn.decomposition import LatentDirichletAllocation

ROOT = os.getcwd()
seed(1)
np.random.seed(1)
from gensim import corpora
from gensim.models.ldamulticore import LdaMulticore
from gensim.models import LdaModel as LMSingle

useLdaVEM = False
repetitions = 10


def calculate(topics=[], lis=[], count1=0):
    count = 0
    for i in topics:
        if i in lis:
            count += 1
    if count >= count1:
        return count
    else:
        return 0


def recursion(topic=[], index=0, count1=0):
    count = 0
    global data
    # print(data)
    # print(topics)
    d = copy.deepcopy(data)
    d.pop(index)
    for l, m in enumerate(d):
        # print(m)
        for x, y in enumerate(m):
            if calculate(topics=topic, lis=y, count1=count1) != 0:
                count += 1
                break
                # data[index+l+1].pop(x)
    return count


data = []


def jaccard(a, score_topics=[], term=0, runs=9):
    global data
    data = score_topics
    j_score = []
    for i, j in enumerate(data):
        for l, m in enumerate(j):
            sum = recursion(topic=m, index=i, count1=term)
            if sum != 0:
                j_score.append(sum / float(runs))
    if len(j_score) == 0:
        j_score = [0]
    Y = sorted(j_score)
    return Y[int(len(Y) / 2)]


def get_top_words(model, feature_names, n_top_words, i=0):
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        li = []
        for j in topic.argsort()[:-n_top_words - 1:-1]:
            li.append(feature_names[j].encode('ascii', 'ignore'))
        topics.append(li)
    return topics


def readfile1(filename=''):
    dict = []
    with open(filename, 'r') as f:
        for doc in f.readlines():
            try:
                row = doc.lower().strip()
                dict.append(row)
            except:
                pass
    return dict


def _test_LDA(data_samples=[],
              term=7,
              random_state=1,
              max_iter=100,
              runs=10,
              n_components=None,
              doc_topic_prior=None,
              topic_word_prior=None):
    topics = []

    for i in range(runs):
        shuffle(data_samples)
        tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
        tf = tf_vectorizer.fit_transform(data_samples)
        lda1 = LatentDirichletAllocation(max_iter=max_iter,
                                         learning_method='online',
                                         random_state=random_state,
                                         n_components=n_components,
                                         doc_topic_prior=doc_topic_prior,
                                         topic_word_prior=topic_word_prior)
        lda1.fit_transform(tf)
        tf_feature_names = tf_vectorizer.get_feature_names()
        topics.append(get_top_words(lda1, tf_feature_names, term, i=i))

    return topics


def prepare_model(data_samples,
                  topic_count,
                  iteration_count,
                  random_state,
                  doc_topic_prior,
                  topic_word_prior,
                  ldaMultiWorkers):
    dictionary = corpora.Dictionary(data_samples)
    dictionary.filter_extremes(no_below=20, no_above=0.2, keep_n=20000)
    corpus = [dictionary.doc2bow(text) for text in data_samples]

    if (ldaMultiWorkers > 0):
        print('LDA MultiCore')
        return LdaMulticore(corpus,
                            num_topics=topic_count,
                            iterations=iteration_count,
                            id2word=dictionary,
                            passes=10,
                            workers=ldaMultiWorkers,
                            random_state=random_state,
                            alpha=doc_topic_prior,
                            eta=topic_word_prior)
    else:
        print('LDA Single')
        return LMSingle(corpus=corpus,
                        num_topics=topic_count,
                        iterations=iteration_count,
                        id2word=dictionary,
                        random_state=random_state,
                        passes=10,
                        alpha=doc_topic_prior,
                        eta=topic_word_prior)


def estimate_model_stability(self, topic_count, iteration_count, repeat=10):
    sum = 0.0
    model1 = self.prepare_model(topic_count, iteration_count)
    for i in range(0, repeat):  # create repeat models
        model2 = self.prepare_model(topic_count, iteration_count)
        topic_similarity_score = self.get_jaccard_similarity(model1, model2)
        sum += topic_similarity_score
        model1 = model2
    avg_stability = sum / repeat
    return avg_stability


def get_jaccard_similarity(model1, model2):
    (differences, annotation) = model1.diff(model2, distance="jaccard", num_words=10, diagonal=True,
                                            annotation=True)
    avg_score = sum(differences) / len(differences)
    return avg_score


def ldascore(*x, **r):
    l = np.asarray(x)
    n_components = l[0]['n_components']
    doc_topic_prior = l[0]['doc_topic_prior']  # theta/alpha
    topic_word_prior = l[0]['topic_word_prior']  # beta/eta
    max_iter = r['max_iter']
    data_samples = r['data_samples']
    random_state = r['random_state']


    if useLdaVEM:
        return ldaVemScore(data_samples,
                           doc_topic_prior,
                           max_iter,
                           n_components,
                           r,
                           random_state,
                           topic_word_prior)
    else:
        return ldaGensimScore(data_samples,
                              doc_topic_prior,
                              max_iter,
                              n_components,
                              r,
                              random_state,
                              topic_word_prior)


def ldaGensimScore(data_samples, doc_topic_prior, max_iter, n_components, r, random_state, topic_word_prior):
    ldaMultiWorkers = r['ldaMultiWorkers']
    print("Attempting to calculate stability score")
    sum_stability = 0.0
    stability_score = 0.0
    prev_model = prepare_model(data_samples=data_samples,
                               ldaMultiWorkers=ldaMultiWorkers,
                               topic_count=n_components,
                               iteration_count=max_iter,
                               random_state=random_state,
                               doc_topic_prior=doc_topic_prior,
                               topic_word_prior=topic_word_prior)
    for repeat in range(0, repetitions):
        shuffle(data_samples)
        model = prepare_model(data_samples=data_samples,
                              ldaMultiWorkers=ldaMultiWorkers,
                              topic_count=n_components,
                              iteration_count=max_iter,
                              random_state=random_state,
                              doc_topic_prior=doc_topic_prior,
                              topic_word_prior=topic_word_prior)

        topic_similarity_score = get_jaccard_similarity(model, prev_model)
        sum_stability += topic_similarity_score
        prev_model = model
        stability_score = sum_stability / repetitions

        print("LDA Score\n\tRepetition: ["
              + str(repeat)
              + "],\tAccumulated Stability: ["
              + str(stability_score)
              + "]")
    return stability_score


def ldaVemScore(data_samples, doc_topic_prior, max_iter, n_components, r, random_state, topic_word_prior):
    term = r['term']
    topics = _test_LDA(data_samples=data_samples,
                       term=int(term),
                       random_state=random_state,
                       max_iter=max_iter,
                       runs=repetitions,
                       n_components=n_components,
                       doc_topic_prior=doc_topic_prior,
                       topic_word_prior=topic_word_prior)
    return jaccard(n_components, score_topics=topics, term=int(term), runs=repetitions - 1)
