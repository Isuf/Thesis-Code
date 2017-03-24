''''

      Author:  Isuf Deliu 
      Date :  24.02.2017
      Place: NTNU Gjovik, Norway 

      The following code implements Paragraph Vectors based on Gensim library.
      Two Main Methods: 
      1) build_d2v_model (file_location, model_name, do_train="True")
      2) run_classification(file_location, labels_location, Dataset_name="my_dataset", do_train="True", show_progress="True")

      Code to Run in Main for Classification:
      dl_doc2vec.run_classification(forum_data_File,forum_labels_file,"Spam Collection","True")

      *** Check the Paramters to Doc2Vec 

'''

from gensim.models import doc2vec

#For classificaition
from sklearn.model_selection import train_test_split
import numpy as np
import Utils
from time import time
import logging

import ShallowClassifiers.clf_helper as shclf

'''

Builds a doc2vec model. Accept the location of the file as input
the File needs to be preprocessed first, as no preprocessing is performed here

'''

def build_d2v_model(file_location, model_name, do_train="True"):

    documents =doc2vec.TaggedLineDocument(file_location)
     
    if (do_train=="True"):

        model = doc2vec.Doc2Vec(documents,size=200, window=5, min_count=3, workers=8, iter=20)
        model.save(model_name)
   
    else:
        model = doc2vec.Doc2Vec.load(model_name)
        model.init_sims(replace=True)


    return model



'''
 Generates the necessary arrays for further classification or clustering
 generates X_train, X_test, y_train, y_test from doc2vec arrays 
'''
def make_d2v_model_ready_for_use(model,data_all, labels_all, test_data_size=0.33, rnd_state=0):
    
    num_examples= len(data_all)
    vector_dimensionality= len(model.docvecs[0])

    #Create training data
    data_arrays = np.zeros((num_examples,vector_dimensionality))
    label_arrays = np.zeros(num_examples)

    for i in range(num_examples):
        prefix_train_pos = str(i)
        data_arrays[i] = model.docvecs[int(prefix_train_pos)]
        label_arrays[i] = labels_all[i]

    X_train, X_test, y_train, y_test = train_test_split(data_arrays, label_arrays, test_size=test_data_size, random_state=rnd_state)

    return X_train, X_test, y_train, y_test


'''
      Classification with Doc2vec vectors as features. The method "make_d2v_model_ready_for_use" is first called 
      to prepare the vectors for classification. The classifier(s) to be used are decided in "classify" method in help_funtions.py

'''
def run_classification(file_location, labels_location, Dataset_name="my_dataset", do_train="True", show_progress="True"):

    if(show_progress=="True"):
        logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')

    t0 = time()
    data, labels= Utils.get_data_and_labels(file_location,  labels_location, Dataset_name)

    duration = time() - t0
    print("Data loaded  in %fs " % (duration))

    doc2vec_model = build_d2v_model(file_location, Dataset_name, do_train)
    scores =[]
    for i in range(0,10):
        X_train, X_test, y_train, y_test = make_d2v_model_ready_for_use (doc2vec_model, data, labels,rnd_state=i)
        results,s = shclf.classify(X_train,X_test,y_train,y_test)
        scores.append(s)

    print("Accuracy: " + str(round(np.mean(scores)*100,3)))


#if __name__ == "__main__" :

#     run_classification(file_location, labels_location, Dataset_name="my_dataset")

