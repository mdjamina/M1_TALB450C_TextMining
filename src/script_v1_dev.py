
import numpy as np
import csv
import treetaggerwrapper as ttw
import pprint as pp

import nltk
#nltk.download('punkt')


from nltk.probability import FreqDist
import string


import spacy
from spacy import displacy


#
def load_corpus(path_file,path_corpus) -> dict:
    """Chargement des fichiers et des annotations:
    """

    #Initialisation du dictionnaire qui contient des annotations:
    corpus = []
    labels = []

    with open(path_file, 'r') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        next(reader, None)
        for row in reader:
            doc_id = row[0].split()[1].strip()
            doc = get_text_by_doc_id(doc_id,path_corpus)

            labels.append(row[1])
            corpus.append(doc)

    return corpus,labels



def get_text_by_doc_id(doc_id, path_corpus) -> str:

    with open(path_corpus+doc_id+'.txt','r',encoding='utf8') as doc:
        return doc.read()


def load_stopwords(path_file):
    """https://countwordsfree.com/stopwords/french
    """
    with open(path_file,'r',encoding='utf8') as csvfile:
        return [w.strip() for w in csvfile]


def tokenization(text,lang='fr',stopwords=[], punctuations='' ):

    tagger = ttw.TreeTagger(TAGLANG=lang)

    tags = tagger.tag_text(text.replace('’',"'"))
   
    return zip(*[tag.split() for tag in tags if tag.split()[2] not in stopwords and tag.split()[2] not in punctuations ] )   


        
def words_frequency(words):
    return nltk.FreqDist([w for w in words])


def most_common_words(words, n=3):

    occ = words_frequency(words)

    return occ.most_common(n)


def trigramme(words):
    return [(i,j,k) for (i,j,k) in zip(*[words[i:] for i in range(3)])]


def concat_docs_by_label(labels:list,docs:list):

    data = {}

    for label,doc in zip(labels,docs):
        data[label] = data.get(label,'')+ '\n' + doc
    
    return data


nlp = spacy.load('fr_core_news_lg')

def get_ner(text):
    doc = nlp(text)
    return [x.text for x in doc.ents if x.label_ in ['ORG','PER','LOC']]
      

def get_avr_len_doc_by_label(labels,docs):

    data = {}
    count_docs = {}

    for cls,doc in zip(labels,docs):
        words =list(tokenization(doc))[0]
        count_docs[cls] = count_docs.get(cls,0) + 1
        data[cls] = data.get(cls,0) + len(words) 
    
    for cls,n in count_docs.items():
        data[cls] = round(data[cls] / n,2)

    return data



def main():
    #nom du fichier annotations corpus (dev)
    path_corpus_dir = './content/corpus/'

    path_stopwords_file = './content/stop_words_french.txt'

    path_file_tsv_dev = './content/annotation_corpus_dev.tsv'

    dev_corpus, dev_labels = load_corpus(path_file_tsv_dev,path_corpus_dir)


    stopwords = load_stopwords(path_stopwords_file)

    punctuations = string.punctuation + '’«»'

     

    data = concat_docs_by_label(dev_labels,dev_corpus)


    data_most_common_by_labels = {}
    data_most_common_lemma_by_labels = {}
    data_most_common_trigram_by_labels = {}
    data_most_common_entity_by_labels = {}
    data_avr_docs_len_by_labels ={}


    for cls,doc in data.items():
        words,pos,lemma =list(tokenization(doc,stopwords=stopwords,punctuations=punctuations))

        #3 mots les plus fréquents (hors stopwords) 
        data_most_common_by_labels[cls] = most_common_words(words,3)

        #Lemmes les plus fréquents pour chaque classe
        data_most_common_lemma_by_labels[cls] = most_common_words(lemma,5)

        words =list(tokenization(doc))[0]

        #5 trigrammes les plus fréquents pour chaque classe
        data_most_common_trigram_by_labels[cls] = most_common_words(trigramme(words),5)

        #Entités nommées les plus fréquentes pour chaque
        #classe (PER-ORG-LOC uniquement)
        data_most_common_entity_by_labels[cls] = most_common_words(get_ner(doc),5)

    
    
    #Longueur moyenne d’un document p/chaque classe
    data_avr_docs_len_by_labels = get_avr_len_doc_by_label(dev_labels,dev_corpus)


        





    


       
    pp.pprint(data_avr_docs_len_by_labels)


    

    
    





    







if __name__ == "__main__":
    main()


