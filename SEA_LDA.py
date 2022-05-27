import multiprocessing
import warnings

warnings.simplefilter("ignore")
import html
import csv

import re
import nltk
import openpyxl
from nltk.stem.snowball import SnowballStemmer
from gensim import corpora, models
from gensim.models.ldamulticore import LdaMulticore
from gensim.models import LdaModel as LMSingle
from gensim.models.coherencemodel import CoherenceModel
import argparse
import pyLDAvis
import pyLDAvis.gensim_models as ldvis
import pandas as pd
import random
import gensim
from gensim.models import EnsembleLda

# create English stop words list

stemmer = SnowballStemmer("english")

my_stopwords = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out',
                'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such',
                'into', 'however', 'every', 'like', 'want', 'fine', 'one', 'two', 'make', 'thing', 'every', 'able'
                'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each',
                'the', 'work', 'set', 'get', 'similar', 'change', 'must', 'above', 'both', 'need',
                'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me',
                'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'could', 'would', 'our', 'their', 'while',
                'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and',
                'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over',
                'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too',
                'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my',
                'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than', 'xa', 'use']

stemmed_stopwords = []

for i in my_stopwords:
    stemmed_stopwords.append(stemmer.stem(i))

domain_terms = ['graphql', 'apollo', 'application', 'app', 'service', 'code', 'gql', 'data', 'object', 'project',
                'schema', 'return', 'name', 'run', 'implement', 'call', 'api', 'file', 'write', 'follow', 'new',
                'update', 'generate', 'class', 'user']

stemmed_domain_terms = []
for i in domain_terms:
    stemmed_domain_terms.append(stemmer.stem(i))

def compute_average_diagonal(matrix, dimension):
    sum =0.0
    for i in range(0, dimension):
        sum += matrix[i][i]
    average =sum/dimension
    return  average

def sent_to_words(sentences):
    for sentence in sentences:
        yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))


def replace_bigram(texts):
    bigram = gensim.models.Phrases(texts, min_count=20, threshold=10)
    mod = [bigram[sent] for sent in texts]
    return mod


def stem_tokens(tokens):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def remove_stopwords(tokens):
    stopped_tokens = [i for i in tokens if not i in stemmed_stopwords]
    return stopped_tokens


def remove_domainterms(tokens):
    newtokens = [i for i in tokens if not i in stemmed_domain_terms]  # remove domain terms
    return newtokens


def tokenize(text):
    tokens = nltk.word_tokenize(text)
    tokens = [w.lower() for w in tokens if w.isalpha() == True or '_' in w]  # lower case, remove number, punctuation
    return tokens


url_regex = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')


def remove_url(s):
    return url_regex.sub(" ", s)


def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    cleantext = html.unescape(cleantext)
    return cleantext


def cleanup_text(text):
    # comments=text
    text = str(text).encode('ascii', 'ignore').decode('ascii')
    text = cleanhtml(text)  # clean html
    text = remove_url(text)  # remove url
    return text


def preprocess_text(text):
    # comments=text
    tokens = tokenize(text)
    tokens = stem_tokens(tokens)
    tokens = remove_stopwords(tokens)
    tokens = remove_domainterms(tokens)
    return tokens


def get_random_number():
    return random.randint(0, 50000)


class LDADocument:
    def __init__(self, id, posttype, body):
        self.id = id
        self.posttype = posttype
        self.body = body


class SEALDAModel:
    def __init__(self, training_data=None,  use_multicore=True, coherence=0.6, core=-1, iterations=50,
                 model_file="graphql-posts.xlsx"):
        self.use_multicore = use_multicore
        self.target_coherence = coherence
        self.fileprefix="graphql"
        if core ==-1:
            self.workers = multiprocessing.cpu_count()
        else: self.workers=core

        self.iterations = iterations

        if (training_data is None):
            self.training_data = self.read_data_from_oracle(model_file)
        else:
            self.training_data = training_data

        self.prepare_training_data()

    def find_best_model(self,min_topic, max_topic, num_iterations):
        best_model =None
        best_cv=0.0
        best_score_topic_count =0
        score_chart_data =dict()
        for topic_count in range(min_topic,max_topic):
            best_score_for_n_topic=0

            for iteration in range(0,num_iterations):
                print("Num topics: "+ str(topic_count) +"\n Iteration: "+ str(iteration))
                model =self.prepare_model(topic_count)
                coherence = self.compute_coherence(model)
                if coherence>best_score_for_n_topic:
                    best_score_for_n_topic=coherence
                    if coherence> best_cv:
                        best_model=model
                        best_cv=coherence
                        best_score_topic_count=topic_count
                print("CV score: "+ str(coherence))
                print("Best CV score: " + str(best_cv) +" achieved with "+str(best_score_topic_count)+ " topics.")
            score_chart_data[topic_count] =best_score_for_n_topic
        self.model=best_model
        return (best_model, best_score_topic_count, best_cv, score_chart_data)



    def get_model(self):
        return self.model

    def visualize(self):
        lda_display = ldvis.prepare(self.model, self.corpus, self.dictionary)
        pyLDAvis.save_html(lda_display, self.fileprefix + ".html")
        #pyLDAvis.display(lda_display)

    def print_topics(self):
        print(self.model.print_topics(num_words=10))

    def prepare_training_data(self):
        training_documents = []
        document_ids = []
        print("Preparing data for training..")
        for document in self.training_data:
            doc = cleanup_text(document.body)
            training_documents.append(doc)
            document_ids.append(document.id)
        self.document_ids = document_ids

        doc_collection = []
        for text in training_documents:
            collection = preprocess_text(text)
            doc_collection.append(collection)

        self.token_collection = replace_bigram(doc_collection)

        self.dictionary = corpora.Dictionary(self.token_collection)
        self.dictionary.filter_extremes(no_below=20, no_above=0.2, keep_n=20000)
        self.corpus = [self.dictionary.doc2bow(text) for text in self.token_collection]
        print("Finished data cleanup..")

    def prepare_model(self, topic_count):
        if (self.use_multicore):
            print('LDA MultiCore')
            ldamodel = LdaMulticore(self.corpus,
                                     num_topics=topic_count,
                                     id2word=self.dictionary,
                                     passes=15,
                                     workers=self.workers,
                                     alpha='symmetric',
                                     random_state=get_random_number(),
                                     eta='auto')

        else:
            ldamodel = LMSingle(corpus=self.corpus,
                                num_topics=self.num_topics,
                                id2word=self.dictionary,
                                random_state=get_random_number(),
                                passes=50,
                                alpha='auto',
                                eta='auto',
                                iterations=self.iterations)
        return ldamodel

    def compute_coherence(self,model):
        coherencemodel = CoherenceModel(model=model, dictionary=self.dictionary, texts=self.token_collection,
                                        topn=15,
                                        coherence='c_v')
        value = coherencemodel.get_coherence()
        return value

    def get_jaccard_similarity(self, model1, model2):
        (differences, annotation)=model1.diff(model2, distance="jaccard", num_words=10, diagonal=True, annotation=True)
        #print(differences)
        avg_score = sum(differences)/len(differences)
        return  avg_score


    def estimate_model_stability(self, topic_count, repeat=10):
        sum =0.0
        model1 = self.prepare_model(topic_count)
        for i in range(0,repeat): #create repeat models
            model2 =self.prepare_model(topic_count)
            topic_similarity_score =self.get_jaccard_similarity(model1,model2)
            sum += topic_similarity_score
            model1 =model2
        avg_stability=sum/repeat
        return  avg_stability



    def create_ensemble_model(self, topic_count):
        return EnsembleLda(
            epsilon=0.1,
        corpus=self.corpus,
        id2word=self.dictionary,
        num_topics=topic_count,
        passes=15,
        num_models=10,
        topic_model_class='ldamulticore',
        ensemble_workers=self.workers,
        distance_workers=self.workers)

    def read_data_from_oracle(self,file):
        dataframe = pd.read_excel(file)
        model_data = []
        print("Reading data from oracle..")
        for index, row in dataframe.iterrows():
            post_id = row["Id"]
            post_type = row["PostTypeId"]
            post_body = row["Body"]

            document = LDADocument(post_id, post_type, post_body)
            model_data.append(document)
        return model_data

    def classify_documents(self):
        df_topic_sents_keywords = self.format_topics_sentences()

        df_dominant_topic = df_topic_sents_keywords.reset_index()
        df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Original_id']

        # Show
        df_dominant_topic.to_csv(self.fileprefix + "-document-to-topic.csv")

    def format_topics_sentences(self):
        # Init output
        sent_topics_df = pd.DataFrame()

        # Get main topic in each document
        for i, row in enumerate(self.model[self.corpus]):
            row = sorted(row, key=lambda x: (x[1]), reverse=True)
            # Get the Dominant topic, Perc Contribution and Keywords for each document
            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:  # => dominant topic
                    wp = self.model.show_topic(topic_num)
                    topic_keywords = ", ".join([word for word, prop in wp])
                    sent_topics_df = sent_topics_df.append(
                        pd.Series([int(topic_num), round(prop_topic, 4), topic_keywords]), ignore_index=True)
                else:
                    break
        sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

        # Add original ids to the end of the output
        contents = pd.Series(self.document_ids)
        sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
        return (sent_topics_df)

    def get_topic(self, text):
        comment = preprocess_text(text)
        feature_vector = self.vectorizer.transform([comment]).toarray()
        topic_class = self.model.predict(feature_vector)
        return topic_class

    def get_topic_collection(self, texts):
        predictions = []
        for text in texts:
            comment = preprocess_text(text)
            feature_vector = self.vectorizer.transform([comment]).toarray()
            topic_class = self.model.predict(feature_vector)
            predictions.append(topic_class)

        return predictions


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='LDA Model')
    parser.add_argument('--multicore', type=bool, help='Is Multicore', default=True)
    parser.add_argument('--mintopic', type=int, help='Minimium number of Topics', default=6)
    parser.add_argument('--maxtopic', type=int, help='Maximum number of Topics', default=20)
    parser.add_argument('--core', type=int, help='CPU Threads', default=-1)
    parser.add_argument('--iteration', type=int, help='Number of iterations', default=25)

    args = parser.parse_args()
    print("args: " + args.__str__())

    multi_core = args.multicore
    mintopic = args.mintopic
    maxtopic = args.maxtopic
    num_core = args.core
    iterations = args.iteration

    lda_trainer = SEALDAModel(use_multicore=multi_core,core=num_core, iterations=iterations)



    (best_model, best_score_topic_count, best_cv,score_chart) =lda_trainer.find_best_model(mintopic,maxtopic, iterations)

    best_model.print_topics()
    print("Best CV score: "+ str(best_cv))
    csv_file = "cv_chart.csv"
    try:
        with open(csv_file, 'w') as csvfile:
            csvfile.write("topic_count, cv_score\n")
            for key, value in score_chart.items():
                csvfile.write(str(key) + ","+ str (value)+"\n")
    except IOError:
        print("I/O error")
    best_model.classify_documents()
    best_model.visualize()

    estimated_stability = lda_trainer.estimate_model_stability(best_score_topic_count)
    print(estimated_stability)

