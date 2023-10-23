import multiprocessing
import warnings
from xml.etree import ElementTree

from pandas.compat import numpy

from LDADE import LDADE, UserTestConfig

warnings.simplefilter("ignore")
import html

import re
import nltk
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
from collections import OrderedDict

# create English stop words list

stemmer = SnowballStemmer("english")

my_stopwords = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out',
                'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such',
                'into', 'however', 'every', 'like', 'want', 'fine', 'one', 'two', 'make', 'thing', 'every', 'able',
                'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each',
                'the', 'work', 'set', 'get', 'similar', 'change', 'must', 'above', 'both', 'need', 'also',
                'may', 'themselves', 'much', 'maybe', 'instead'
                'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me',
                'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'could', 'would', 'our', 'their', 'while',
                'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and',
                'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over',
                'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too',
                'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my',
                'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than', 'xa', 'use', 'might']

stemmed_stopwords = []

for i in my_stopwords:
    stemmed_stopwords.append(stemmer.stem(i))

domain_terms = ['graphql', 'apollo', 'gql', 'something', 'look', 'see', 'find', 'way', 'different', 'differ', 'foo',
                'bar', 'baz', 'pet', 'example', 'call', 'follow', 'possible', 'know', 'c', 'value', 'api', 'seem',
                'question', 'case', 'item', 'true', 'false', 'found', 'issue', 'problem', 'node', 'post', 'request',
                'response', 'variable', 'div', 'page', 'app', 'server', 'client', 'send', 'let', 'think', 'require',
                'null', 'object', 'result', 'number', 'success', 'input', 'book', 'import', 'export', 'boolean',
                'author', 'title', 'int', 'default', 'e', 'person', 'async', 'age', 'country', 'city', 'address', 'r'
                'song', 'artist', 'episode', 'friend', 'movie', 'employee', 'human', 'customer', 'vehicle', 'game'
                'document', 'var', 'player', 'p', 'todo', 'req', 'res', 'go', 'actual', 'url', 'uri', 'print', 'run',
                'contain', 'add', 'provide', 'pm', 'err', 'article', 'first', 'list', 'show', 'inside', 'path',
                'detail', 'implement', 'context', 'text', 'action', 'description', 'time', 'file', 'mutate',
                'mutation', 'user', 'option', 'create', 'price', 'application']

# Refactor
# domain_terms = ['name', 'string', 'public', 'int', 'x', 'y', 'f', 'n', 'b', 'c', 'j', 'g', 'v', 'h', 'i', 'see',
#                 'something', 'might', 'z', 'k', 'i', "I", 'l', 'r', 'e', 'q', 'refactor', 'end', 'move', 'way',
#                 'find']

stemmed_domain_terms = []

for i in domain_terms:
    stemmed_domain_terms.append(stemmer.stem(i))


def compute_average_diagonal(matrix, dimension):
    sum = 0.0
    for i in range(0, dimension):
        sum += matrix[i][i]
    average = sum / dimension
    return average


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
    def __init__(self, training_data=None, use_multicore=True, num_core=-1,
                 model_name="graphql", source_file_format = "xlsx"):

        self.use_multicore = use_multicore
        self.model_name = model_name
        if num_core == -1:
            self.workers = multiprocessing.cpu_count()
        else:
            self.workers = num_core

        file_format = self.model_name + "-posts." + source_file_format
        if (training_data is None):
            if source_file_format.__eq__("xlsx"):
                self.training_data = self.read_data_from_xlsx(file_format)
            else:
                self.training_data = self.read_data_from_xml(file_format)
        else:
            self.training_data = training_data

        self.prepare_training_data()

    def find_best_model(self,
                        min_topic,
                        max_topic,
                        step_topic,
                        min_iteration,
                        max_iteration,
                        step_iteration,
                        repetitions):
        index = 1
        current_best_model = None
        current_best_cv = 0.0
        current_best_iteration_count = 0
        current_best_topic_count = 0
        current_best_estimated_stability = 0.0

        # Determine best model [iterations, topics, cv and stability]
        for iteration_count in range(min_iteration, max_iteration, step_iteration):
            for topic_count in range(min_topic, max_topic, step_topic):
                (self.model, cv, estimated_stability) = self.find_single_model(topic_count,
                                                                               repetitions,
                                                                               iteration_count)
                if cv > current_best_cv:
                    current_best_cv = cv
                    current_best_model = self.model
                    current_best_topic_count = topic_count
                    current_best_iteration_count = iteration_count
                    current_best_estimated_stability = estimated_stability

                print("\t Iterations: ["
                      + str(iteration_count)
                      + "]\tNum Topics: ["
                      + str(topic_count)
                      + "],\tCV score: ["
                      + str(cv) + "]")

                current_cv_chart_data = str(iteration_count) \
                                       + "," + str(topic_count) \
                                       + "," + str(cv) \
                                       + "_" + str(estimated_stability) \
                                       + "_" + str(self.get_topics())

                csv_file = self.model_name + "-cv_chart.csv"
                try:
                    with open(csv_file, 'a') as csvfile:
                        csvfile.write(current_cv_chart_data + "\n")
                except IOError:
                    print("I/O error")

                iteration_topic_name = str(iteration_count) + "_" + str(topic_count)
                self.classify_documents(iteration_topic_name)
                self.visualize(iteration_topic_name)
                index += 1

        print("Best num topics: " + str(current_best_topic_count))
        print("Best iterations: " + str(current_best_iteration_count))
        print("Best stability: " + str(current_best_estimated_stability))
        print("Best cv: " + str(current_best_cv))

        # Code to run LDA DE

        # config = UserTestConfig()
        # config["data_samples"] = self.token_collection
        # config["learners_para_bounds"] = [(current_best_topic_count, current_best_topic_count), (0.1, 1), (0.01, 1)]
        # learner_para = OrderedDict()
        # learner_para["n_components"] = current_best_topic_count  # topics
        # learner_para["doc_topic_prior"] = 0.1  # alpha
        # learner_para["topic_word_prior"] = 0.01  # beta
        # config["learners_para"] = learner_para
        #
        # config["ldaMultiWorkers"] = self.workers
        # config["max_iter"] = current_best_iteration_count
        #
        # (ind, fit) = LDADE(config)
        # print("Index" + str(ind))
        # print("Fit" + str(fit))

        return current_best_model, current_best_topic_count, current_best_iteration_count, current_best_cv,\
            current_best_estimated_stability

    def find_single_model(self, topic_count, repetitions=25, iteration_count=1000):
        current_best_model = None
        current_best_cv = 0
        sum_stability = 0.0
        estimated_stability = 0.0
        prev_model = self.prepare_model(topic_count, iteration_count)
        for repeat in range(0, repetitions):
            self.model = self.prepare_model(topic_count, iteration_count)
            cv = self.compute_coherence(self.model)
            if cv > current_best_cv:
                current_best_cv = cv
                current_best_model = self.model
            topic_similarity_score = self.get_jaccard_similarity(self.model, prev_model)
            sum_stability += topic_similarity_score
            prev_model = self.model
            estimated_stability = sum_stability / repetitions

            print("\t Repetition: ["
                  + str(repeat)
                  + "],\tCV score: "
                  + str(cv)
                  + "],\tAccumulated Stability: ["
                  + str(estimated_stability)
                  + "]")

        return current_best_model, current_best_cv, estimated_stability

    def get_model(self):
        return self.model

    def visualize(self, postfix):
        lda_display = ldvis.prepare(self.model, self.corpus, self.dictionary)
        pyLDAvis.save_html(lda_display, self.model_name + "_" + postfix + ".html")
        # pyLDAvis.display(lda_display)

    def get_topics(self):
        return self.model.print_topics(num_words=10)

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

    def prepare_model(self, topic_count, iteration_count):
        if (self.use_multicore):
            print('LDA MultiCore')
            ldamodel = LdaMulticore(self.corpus,
                                    num_topics=topic_count,
                                    iterations=iteration_count,
                                    id2word=self.dictionary,
                                    passes=10,
                                    workers=self.workers,
                                    random_state=get_random_number(),
                                    alpha='symmetric',
                                    eta='auto')
        else:
            ldamodel = LMSingle(corpus=self.corpus,
                                num_topics=topic_count,
                                iterations=iteration_count,
                                id2word=self.dictionary,
                                random_state=get_random_number(),
                                passes=10,
                                alpha='auto',
                                eta='auto')
        return ldamodel

    def compute_coherence(self, model):
        coherencemodel = CoherenceModel(model=model, dictionary=self.dictionary, texts=self.token_collection,
                                        topn=15,
                                        coherence='c_v')
        value = coherencemodel.get_coherence()
        return value

    def get_jaccard_similarity(self, model1, model2):
        (differences, annotation) = model1.diff(model2, distance="jaccard", num_words=10, diagonal=True,
                                                annotation=True)
        # print(differences)
        avg_score = sum(differences) / len(differences)
        return avg_score

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

    def read_data_from_xlsx(self, file):
        dataframe = pd.read_excel(file)
        model_data = []
        print("Reading data from: " + file)
        for index, row in dataframe.iterrows():
            post_id = row["Id"]
            post_type = row["PostTypeId"]
            post_body = row["Body"]
            document = LDADocument(post_id, post_type, post_body)
            model_data.append(document)
        return model_data

    def read_data_from_xml(self, file):

        model_data = []
        print("Reading data from: " + file)

        xml_iter = ElementTree.iterparse(file, ('start', 'end'))
        for event, elem in xml_iter:
            tag = elem.tag
            if event == "start" and tag == "row":
                post_id = elem.attrib["Id"]
                post_body = elem.attrib["Body"]
                post_type = elem.attrib["PostTypeId"]
                document = LDADocument(post_id, post_type, post_body)
                model_data.append(document)
        return model_data

    def classify_documents(self, postfix):
        df_topic_sentences_keywords = self.format_topics_sentences()
        df_dominant_topic = df_topic_sentences_keywords.reset_index()
        df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Original_id']

        # Show
        df_dominant_topic.to_csv(self.model_name + "_" + postfix + "-document-to-topic.csv")

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

    parser = argparse.ArgumentParser(description='SEA LDA Model')

    # Configs -- Cores
    parser.add_argument('--multicore', type=bool, help='Is multicore', default=True)
    parser.add_argument('--numcore', type=int, help='CPU threads', default=-1)

    # Configs -- Topics
    parser.add_argument('--mintopic', type=int, help='Minimum number of topics', default=5)
    parser.add_argument('--maxtopic', type=int, help='Maximum number of topics', default=51)
    parser.add_argument('--steptopic', type=int, help='Step to increase the number of topics', default=5)

    # Configs -- Iterations
    parser.add_argument('--miniteration', type=int, help='Minimum number of iterations', default=500)
    parser.add_argument('--maxiteration', type=int, help='Maximum number of iterations', default=2001)
    parser.add_argument('--stepiteration', type=int, help='Maximum number of iterations', default=500)

    # Configs -- Misc
    parser.add_argument('--repetitions', type=int, help='Number of passes ', default=25)
    parser.add_argument('--modelname', type=str, help='Name of the model', default="graphql")
    parser.add_argument('--sourcefileformat', type=str, help='Format of the source file containing posts', default="xlsx")

    args = parser.parse_args()
    print("args: " + args.__str__())

    # Create cv chart file with top headers
    csv_file = args.modelname + "-cv_chart.csv"
    try:
        with open(csv_file, 'w') as csvfile:
            csvfile.write("iteration_count, topic_count, cv_score, stability, topics\n")
    except IOError:
        print("I/O error")

    # Start SEAL-LDA
    sea_lda = SEALDAModel(use_multicore=args.multicore,
                          num_core=args.numcore,
                          model_name=args.modelname,
                          source_file_format=args.sourcefileformat)

    (best_model, best_topic_count, best_iteration_count, best_cv, best_estimated_stability) = \
        sea_lda.find_best_model(min_topic=args.mintopic,
                                max_topic=args.maxtopic,
                                step_topic=args.steptopic,
                                min_iteration=args.miniteration,
                                max_iteration=args.maxiteration,
                                step_iteration=args.stepiteration,
                                repetitions=args.repetitions)
