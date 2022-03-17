
import numpy as np
import csv
#import treetaggerwrapper as ttw
import pprint as pp

import nltk
# nltk.download('punkt')


from nltk.probability import FreqDist
import string


import spacy
from spacy import displacy
from sklearn.feature_extraction.text import CountVectorizer


#
def load_corpus(path_file, path_corpus) -> dict:
    """Chargement des fichiers et des annotations:
    """

    # Initialisation du dictionnaire qui contient des annotations:
    corpus = []
    labels = []

    with open(path_file, 'r') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        next(reader, None)
        for row in reader:
            doc_id = row[0].split()[1].strip()
            doc = get_text_by_doc_id(doc_id, path_corpus)

            labels.append(row[1])
            corpus.append(doc)

    return corpus, labels


def get_text_by_doc_id(doc_id, path_corpus) -> str:

    with open(path_corpus+doc_id+'.txt', 'r', encoding='utf8') as doc:
        return doc.read()


def load_stopwords(path_file):
    """https://countwordsfree.com/stopwords/french
    """
    with open(path_file, 'r', encoding='utf8') as csvfile:
        return [w.strip() for w in csvfile]

"""
def tokenization(text, lang='fr', stopwords=[], punctuations=''):

    tagger = ttw.TreeTagger(TAGLANG=lang)

    tags = tagger.tag_text(text.replace('’', "'"))

    return zip(*[tag.split() for tag in tags if tag.split()[2] not in stopwords and tag.split()[2] not in punctuations])
"""

def words_frequency(words):
    return nltk.FreqDist([w for w in words])


def most_common_words(words, n=3):

    occ = words_frequency(words)

    return occ.most_common(n)


def trigramme(words):
    return [(i, j, k) for (i, j, k) in zip(*[words[i:] for i in range(3)])]


def concat_docs_by_label(labels: list, docs: list):

    data = {}

    for label, doc in zip(labels, docs):
        data[label] = data.get(label, '') + '\n' + doc

    return data


nlp = spacy.load('fr_core_news_lg')


def get_ner(text):
    doc = nlp(text)
    return [x.text for x in doc.ents if x.label_ in ['ORG', 'PER', 'LOC']]


def get_avr_len_doc_by_label(labels, docs):

    data = {}
    count_docs = {}

    for cls, doc in zip(labels, docs):
        words = list(tokenization(doc))[0]
        count_docs[cls] = count_docs.get(cls, 0) + 1
        data[cls] = data.get(cls, 0) + len(words)

    for cls, n in count_docs.items():
        data[cls] = round(data[cls] / n, 2)

    return data


def main():
    # nom du fichier annotations corpus (dev)
    path_corpus_dir = './content/corpus/'

    path_stopwords_file = './content/stop_words_french.txt'

    path_file_tsv = './content/annotation_corpus_dev.tsv'

    #chargement du corpus
    corpus, labels = load_corpus(path_file_tsv, path_corpus_dir)


    """
    TODO: découpage du corpus (x_train,y_train, x_test,y_test)
    
    """

    x_train = corpus
    y_train = labels



    #vectorisation du corpus
    vectorizer = CountVectorizer()
    x_train_vectorized = vectorizer.fit_transform(x_train)
    #print(vectorizer.get_feature_names_out())

    print(x_train_vectorized.toarray())

    print(x_train_vectorized.shape)

    







    




if __name__ == "__main__":
    main()
