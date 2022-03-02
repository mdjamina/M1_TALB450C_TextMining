import numpy as np
import csv



def load_annotations(path_file) -> dict:
    """
        Chargement du fichier des annotations csv
    """

    #Initialisation du dictionnaire qui contient des annotations:
    corpus = {}

    with open(path_file, 'r') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        next(reader, None)
        for row in reader:
            key = row[0].split()[1].strip()
            value = row[1].strip()
            corpus[key] = value
    
    return corpus

#
def load_corpus(path_file) -> dict:
    """Chargement du fichier des annotations:
    """
    
    #Initialisation du dictionnaire qui contient des annotations:
    corpus = {}

    with open(path_file, 'r') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        next(reader, None)
        for row in reader:
            value = row[0].split()[1].strip()
            key = row[1]
            corpus[key] = corpus.get(key,[]) +  [value]

    return corpus


def train_test_split(data:dict, test_size:int=10,dev_size:int=None):
    """
    Divisez le corpus en sous-ensembles d'entraînement, de développement et de test aléatoires.


    Parameters
    ----------
    data: le corpus

    test_size: int, default=10
        représente le percentage d'échantillons de test.

    dev_size: int, default=None
        représente le percentage d'échantillons de dev.
        Si (dev_size=None) => train_size = 100-(test_size)
        Sinon train_size = 100 - (test_size + dev_size)


    """

    # échantillonnage des données

    # corpus d'entrainnement 
    train ={}

    # corpus du dev 
    dev = {}

    # corpus du teste 
    test = {}


    for classe,lst_docs in data.items():
  
        #pour chaque classe
        #calcul du nombre du document de 10%
        n = round( len(lst_docs)*test_size/100) 
        #print('classe:',classe,' tot doc=',len(lst_docs),' n=',n)

        #selection aléatoire des n doc_id 
        x = list( np.random.choice(lst_docs, n, replace=False))

        #ajout de la classe et l'échantillon doc_id dans le corpus de dev
        test[classe] = x

        # suppression des doc_id déjà selectionnés
        lst_docs = list(set(lst_docs) - set(x))

        if dev_size is not None:
  
            #selection aléatoire des n doc_id 
            x = list(np.random.choice(lst_docs, n, replace=False))

            #ajout de la classe et l'échantillon doc_id dans le corpus de test
            dev[classe] = x

            # suppression des doc_id déjà selectionnés
            lst_docs = list(set(lst_docs) - set(x))


        #ajout du reste des doc_ids (80%) dans le corpus train
        train[classe] = lst_docs
    

    return (train,test,dev)


def get_text_by_doc_id(doc_id, directory) -> str:

    with open(directory+doc_id+'.txt','r',encoding='utf8') as doc:
        return doc.read()


def get_text(list_doc_id:list, directory) -> str:
    
    return '\n'.join([get_text_by_doc_id(doc_id,directory) for doc_id in list_doc_id ])
     

def main():
    #nom du fichier annotations corpus (dev)
    path_corpus_dir = './content/corpus/'
    #path_file_tsv = './content/annotation_corpus.tsv'

    #corpus = load_corpus(path_file_tsv)

    #(corpus_train,corpus_test,corpus_dev) = train_test_split(corpus,dev_size=10)

    path_file_tsv_dev = './content/annotation_corpus_dev.tsv'

    corpus_dev = load_corpus(path_file_tsv_dev)

    print(corpus_dev)
 
    text = get_text(corpus_dev['politique'],path_corpus_dir)

    print(text[:500])


if __name__ == "__main__":
    main()


