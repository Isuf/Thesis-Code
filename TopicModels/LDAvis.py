from __future__ import print_function
from time import time
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

from sklearn.model_selection import train_test_split
import pyLDAvis
import pyLDAvis.sklearn 
pyLDAvis.enable_notebook()





def build_LDA_model(dataset, n_topics=10, n_top_words=10,n_features = 10000, test_data_size = 0.3,show_topics=True):


    # 1. Data Load 
    train_samples, heldout_samples = train_test_split(dataset,test_size=test_data_size, random_state=0)

    n_samples = len(train_samples)
         
    # 2. Feature Extaction from raw data
    # Use tf (raw term count) features for LDA.
    print("Extracting tf features for LDA...")
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=1,
                                    max_features=n_features,
                                    stop_words='english')

    t0 = time()
    tf = tf_vectorizer.fit_transform(train_samples)
    print("done in %0.3fs." % (time() - t0))

    
    print("Fitting LDA models with tf features, "
          "n_samples=%d and n_features=%d..."
          % (n_samples, n_features))

    # 3. Build the model
    lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
                                    learning_method='online',
                                    learning_offset=50.,
                                    random_state=0)

    # 4. Train the model
    t0 = time()
    lda.fit(tf)
    print("done in %0.3fs." % (time() - t0))

    if(show_topics==True):

        #5 Show the topics
        print("\nTopics in LDA model:")
        tf_feature_names = tf_vectorizer.get_feature_names()
        print_top_words(lda, tf_feature_names, n_top_words)

    return lda, tf,tf_vectorizer


''' Loads a file and saves its content as a list 
       [ "this is test 1", " this is test 2"]
'''
def loadfile(fileName): 
    with open(fileName,encoding="utf8",errors='ignore') as f:
         content = f.readlines()
         content = [x for x in content] 
    return content

dataset_locations ={ 
    "une": ("D:\\Tema NTNU\\Data\\Nulled\\Deliu\\positive.txt","")
    }
train_data_samples = loadfile(dataset_locations["une"][0])

num_topics = 10 
num_top_words =20
num_features = 30000
heldout_data_size=0.1

# Run LDA     
lda,tf,tf_vectorizer=build_LDA_model(train_data_samples,
                                         n_topics=num_topics, n_top_words=num_top_words,
                                         n_features = num_features, test_data_size=heldout_data_size,
                                         show_topics=False)

pyLDAvis.sklearn.prepare(lda, tf, tf_vectorizer)