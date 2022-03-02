import numpy as np
import csv


def dict_to_file(data, path_file_tsv):

  with open(path_file_tsv,'w',newline='') as tsv:
    writer = csv.writer(tsv,delimiter='\t')
    writer.writerow(['DocId','Classe'])
    for key,value in data.items():
      for e in value:
        writer.writerow([e,key])


def train_test_split(path_file_tsv, test_size:int=10,dev_size:int=10):
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

    data={}


    #chargement du fichier des annotations
    with open(path_file_tsv, 'r') as tsvfile:
      reader = csv.reader(tsvfile, delimiter='\t')
      next(reader, None) 
      for row in reader:    
        value = row[0].split()[1].strip()
        key = row[1].strip()
        data[key] = data.get(key,[]) +  [value]


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
        n_test = round( len(lst_docs)*test_size/100) 
        n_dev = round( len(lst_docs)*dev_size/100) 
        #print('classe:',classe,' tot doc=',len(lst_docs),' n=',n)

        #selection aléatoire des n doc_id 
        x = list( np.random.choice(lst_docs, n_test, replace=False))

        #ajout de la classe et l'échantillon doc_id dans le corpus de dev
        test[classe] = x

        # suppression des doc_id déjà selectionnés
        lst_docs = list(set(lst_docs) - set(x))

        if dev_size != None:

            #calcul du nombre du document de 10%
            

  
            #selection aléatoire des n doc_id 
            x = list(np.random.choice(lst_docs, n_dev, replace=False))

            #ajout de la classe et l'échantillon doc_id dans le corpus de test
            dev[classe] = x

            # suppression des doc_id déjà selectionnés
            lst_docs = list(set(lst_docs) - set(x))


        #ajout du reste des doc_ids (80%) dans le corpus train
        train[classe] = lst_docs


    outputfile = path_file_tsv.split('.')[0] + '_{}.tsv'

    dict_to_file(train,outputfile.format('train'))
    dict_to_file(test,outputfile.format('test'))
    dict_to_file(dev,outputfile.format('dev'))
    

    return (train,test,dev)


 

def main():
    #nom du fichier annotations corpus (dev)
    path_corpus_dir = './content/corpus/'
    path_file_tsv = 'content/annotation_corpus.tsv'



    (corpus_train,corpus_test,corpus_dev) = train_test_split(path_file_tsv,dev_size=10)

    print(corpus_dev)
 
    


if __name__ == "__main__":
    main()
