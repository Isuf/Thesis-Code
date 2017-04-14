# For features
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline

#For classificaition
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np
import Utils
from time import time
import logging

#import ShallowClassifiers.clf_helper as clfh
import ShallowClassifiers.clf_helper as clfh
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix


def print_settings(param):
    print("Settings : \n\n")
    print("##################################################################################################")
    method_description=""
    if(param["ngrams_bow"]["method"]=="ngrams"):
        method_description=param["ngrams_bow"]["feature_level"] + " level ngrams"
    else: 
        method_description=param["ngrams_bow"]["method"]

    print("Method: " + method_description)
    if(param["ngrams_bow"]["method"]=="ngrams"):
      print("ngram_range: ("+str(param["ngrams_bow"]["min_ngrams"])+ " ," + str(param["ngrams_bow"]["max_ngrams"]) + ")")
    print("##################################################################################################")

    print("\n")

def construct_tfidf_features(X_train_raw, X_test_raw):
   
     # Tfidf vectorizer:
    #   - Strips out 'stop words'
    #   - Filters out terms that occur in more than half of the docs (max_df=0.5)
    #   - Filters out terms that occur in only one document (min_df=2).
    #   - Selects the 10,000 most frequently occuring words in the corpus.
    #   - Normalizes the vector (L2 norm of 1.0) to normalize the effect of 
    #     document length on the tf-idf values. 
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=10000,
                                 min_df=2, stop_words='english',
                                 use_idf=True)

    # Build the tfidf vectorizer from the training data ("fit"), and apply it to extract features from text data as well
    X_train_tfidf = vectorizer.fit_transform(X_train_raw)
    X_test_tfidf = vectorizer.transform(X_test_raw)

    print("  Number of tfidf features: %d" % X_train_tfidf.get_shape()[1])
    return X_train_tfidf, X_test_tfidf
    
def construct_LSA_features(X_train_tfidf, X_test_tfidf, number_of_components=100):
    t0=time()
    # Project the tfidf vectors onto the first 100 principal components.
    svd = TruncatedSVD(number_of_components)
    lsa = make_pipeline(svd, Normalizer(copy=False))
 
    # Run SVD on the training data, then project the training data.
    X_train_lsa = lsa.fit_transform(X_train_tfidf)

    # Run SVD on the training data, then project the training data.
    X_train_lsa = lsa.fit_transform(X_train_tfidf)

    print("  done in %.3fsec" % (time() - t0))

    explained_variance = svd.explained_variance_ratio_.sum()
    print("  Explained variance of the SVD step: {}%".format(int(explained_variance * 100)))


    # Now apply the transformations to the test data as well.
    X_test_lsa = lsa.transform(X_test_tfidf)

    return X_train_lsa, X_test_lsa

def construct_ngram_features(X_train, X_test, use_hashing=False,max_num_features=10000, feature_level="char", min_ngram=1, max_ngram=1):
    
    t0 = time()
    if (use_hashing ==True):
        vectorizer = HashingVectorizer(stop_words='english', non_negative=True,
                                        n_features=max_num_features,
                                        analyzer=feature_level,
                                        ngram_range=(min_ngram,max_ngram))
        

        X_train = vectorizer.transform(X_train)
         
    else:
        vectorizer = HashingVectorizer(stop_words='english', non_negative=True,
                                        n_features=max_num_features,
                                        analyzer=feature_level,
                                        ngram_range=(min_ngram,max_ngram))

         #vectorizer = TfidfVectorizer( stop_words='english', 
         #                              sublinear_tf=True, max_df=0.95) 

        X_train = vectorizer.fit_transform(X_train)

    print("Extracting features from the test data using the same vectorizer")
    t0 = time()
    X_test = vectorizer.transform(X_test)
    duration = time() - t0
    print("done in %fs " % (duration))
    print("n_samples: %d, n_features: %d" % X_test.shape)
    print()

    return X_train,X_test

def construct_bow_features(X_train,X_test, use_hashing=False,max_num_features=10000, feature_level="char", min_ngram=1, max_ngram=1):
    t0 = time()
    if use_hashing==True: 
        vectorizer = HashingVectorizer(stop_words='english', non_negative=True,
                                       n_features=max_num_features)

        X_train = vectorizer.transform(X_train)
    else:
        vectorizer = TfidfVectorizer(stop_words='english', non_negative=True,
                                     n_features=max_num_features)  # analyzer not accepted as paramter??? )

        X_train = vectorizer.fit_transform(X_train)

    print("Extracting features from the test data using the same vectorizer")
    t0 = time()
    X_test = vectorizer.transform(X_test)
    duration = time() - t0
    #print("done in %fs at %0.3fMB/s" % (duration, data_test_size_mb / duration))
    print("done in %fs " % (duration))
    print("n_samples: %d, n_features: %d" % X_test.shape)
    print()

    return X_train,X_test


def run_document_classification(data, labels, feature_construction_method, parameters, show_progress=True):

    #Define Input parameters 
    max_num_features = parameters["ngrams_bow"]["max_num_features"]
    feature_level = parameters["ngrams_bow"]["feature_level"]
    min_ngrams = parameters["ngrams_bow"]["min_ngrams"]
    max_ngrams = parameters["ngrams_bow"]["max_ngrams"]
    print_settings(parameters)
    
    if(show_progress==True):
       logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')

    #STEP 1: The data 
    data = np.array(data)
    labels = np.array(labels)

    #STEP 2: 10-Fold Cross Validation of Data
    kf = KFold(n_splits=10,shuffle=True)

    scores=[]
    precision=[]
    recall = []  
    f1 = []

    for train_index, test_index in kf.split(data):
    
        # Get the data for the current fold (10fold CV) 
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        print("Extracting features from the training data using a sparse vectorizer")

        if feature_construction_method =="bow":   # Bag-of-Words as features
            X_train,X_test=construct_bow_features(X_train,X_test,False,max_num_features,feature_level,min_ngrams,max_ngrams)

        elif feature_construction_method =="ngrams": # n-grams as features 
            X_train,X_test=construct_ngram_features(X_train,X_test,False,max_num_features,feature_level,min_ngrams,max_ngrams)
       
        elif feature_construction_method =="lsa":
             X_train_tfidf, X_test_tfidf = construct_tfidf_features(X_train, X_test)
             X_train,X_test=construct_LSA_features(X_train_tfidf,X_test_tfidf)
        
        else:
             print("The method name is wrong ...")


        #Training/Classification
        results ,s= clfh.classify(X_train,X_test,y_train,y_test)
        y_pred=results[0][5]
        y_test=results[0][4]
        scores.append(s)

        f1.append(f1_score(y_test, y_pred, average="macro"))
        precision.append(precision_score(y_test, y_pred, average="macro"))
        recall.append(recall_score(y_test, y_pred, average="macro"))
        print(classification_report(y_test, y_pred))
 
    print("Average Accuracy: " +  str(round(np.mean(scores)*100,2)))
    print("Average Precision: " + str(round(np.mean(precision)*100,3)))
    print("Average Recall: " +    str(round(np.mean(recall)*100,3)))
    print("Average F1 score: " + str(round(np.mean(f1)*100,3)))


    
    

 
