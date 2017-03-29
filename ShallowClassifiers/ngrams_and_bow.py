# For features
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer


#For classificaition
from sklearn.model_selection import train_test_split
import numpy as np
import Utils
from time import time
import logging

#import ShallowClassifiers.clf_helper as clfh
import ShallowClassifiers.clf_helper as clfh

def print_settings(param):
    print("Settings : \n\n")
    print("Method: " + param["ngrams_bow"]["method"])
    print( param["ngrams_bow"]["ngram_unit"] + "level ngrams")
    print("ngram_range: ("+str(param["ngrams_bow"]["min_ngrams"])+ " ," + str(param["ngrams_bow"]["max_ngrams"]) + ")")
    print("\n")

def feature_construction_ngrams(X_train,X_test,param):
    t0 = time()
    if param["ngrams_bow"]["use_hashing"]==True: 
         vectorizer = HashingVectorizer(stop_words='english', non_negative=True,
                                        n_features=param["ngrams_bow"]["n_features"],
                                        analyzer=param["ngrams_bow"]["ngram_unit"],
                                        ngram_range=(param["ngrams_bow"]["min_ngrams"],param["ngrams_bow"]["max_ngrams"]))

         X_train = vectorizer.transform(X_train)
    else:

         #vectorizer = TfidfVectorizer( stop_words='english', 
         #                              sublinear_tf=True, max_df=0.95,
         #                              analyzer=param["ngrams_bow"]["ngram_unit"],
         #                              ngram_range=(param["ngrams_bow"]["min_ngrams"],param["ngrams_bow"]["max_ngrams"]))
         vectorizer = TfidfVectorizer(
                                     analyzer=param["ngrams_bow"]["ngram_unit"],
                                      ngram_range=(param["ngrams_bow"]["min_ngrams"],param["ngrams_bow"]["max_ngrams"]))

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


def feature_construction_BOW(X_train,X_test,param):
    t0 = time()
    if param["ngrams_bow"]["use_hashing"]==True: 
        vectorizer = HashingVectorizer(stop_words='english', non_negative=True,
                                       n_features=param["ngrams_bow"]["n_features"])

        X_train = vectorizer.transform(X_train)
    else:
        vectorizer = TfidfVectorizer(stop_words='english',
                                     sublinear_tf=True, max_df=0.5)  # analyzer not accepted as paramter??? )

        X_train = vectorizer.fit_transform(X_train)
    

    duration = time() - t0
    print("done in %fs " % (duration))#, data_train_size_mb / duration))
    print("n_samples: %d, n_features: %d" % X_train.shape)
    print()

    print("Extracting features from the test data using the same vectorizer")
    t0 = time()
    X_test = vectorizer.transform(X_test)
    duration = time() - t0
    #print("done in %fs at %0.3fMB/s" % (duration, data_test_size_mb / duration))
    print("done in %fs " % (duration))
    print("n_samples: %d, n_features: %d" % X_test.shape)
    print()

    return X_train,X_test

def run_classification(file_location, labels_location, param,Dataset_name="my_dataset", show_progress="True",test_data_size=0.1):
    print_settings(param)

    if(show_progress=="True"):
        logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')


 
    # Step 1 : Data Load
    t0 = time()
    data= Utils.loadfile(file_location)
    labels = labels_location
    #X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_data_size, random_state=0)
    duration = time() - t0
    print("Data loaded  in %fs " % (duration))

    data = np.array(data)
    labels = np.array(labels)

    #data=np.random.permutation(data)
    ### END_OF DATA_LOAD #######################

    from sklearn.model_selection import KFold

    for i in range (1):

        kf = KFold(n_splits=10,shuffle=True, random_state=i)

        scores=[]
        for train_index, test_index in kf.split(data):
                X_train, X_test = data[train_index], data[test_index]
                y_train, y_test = labels[train_index], labels[test_index]

                print("Extracting features from the training data using a sparse vectorizer")

                if param["ngrams_bow"]["method"] =="bow":
                   X_train,X_test=feature_construction_BOW(X_train,X_test,param)

                elif param["ngrams_bow"]["method"] =="ngrams":
                   X_train,X_test=feature_construction_ngrams(X_train,X_test,param)
                else:
                    print("The method name is wrong ...")


                #2 Training/Classification
                results ,s= clfh.classify(X_train,X_test,y_train,y_test)
                scores.append(s)
         
    print("Accuracy: " + str(round(np.mean(scores)*100,3)))