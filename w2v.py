from __future__ import print_function
#from gensim.models import word2vec
from os.path import join, exists, split
import os
import numpy as np
import gensim
#https://github.com/alexander-rakhlin/CNN-for-Sentence-Classification-in-Keras/blob/master/data_helpers.py
def build_word2vec(num_sentences,max_sentence_len,data, vocabulary_inv, name="Google",binary_format=True,vector_size=300, min_word_count=1, context=5 ):
    a=5
    model_dir = 'D:\Tema NTNU\Data\Vector_Models'
    model_name="{:s}_word2vec_{:d}".format(name,vector_size)
    model_name = join(model_dir, model_name+ (".bin" if binary_format else ".txt"))
    if exists(model_name):
        model =gensim.models.KeyedVectors.load_word2vec_format(model_name, binary=binary_format)   
        print('Loading existing Word2Vec model \'%s\'' % split(model_name)[-1])
    else:
        #sentences = [sent["text"] for sent in data]
        sentences= [sent["text"].split() for sent in data]
        model =gensim.models.word2vec.Word2Vec(sentences, size=vector_size,min_count=min_word_count,window=context)
       
        # If we don't plan to train the model any further, calling 
        # init_sims will make the model much more memory-efficient.
        model.init_sims(replace=True)

        # Saving the model for later use. You can load it later using Word2Vec.load()
        if not exists(model_dir):
            os.mkdir(model_dir)
        print('Saving Word2Vec model \'%s\'' % split(model_name)[-1])
        model.wv.save_word2vec_format(model_name,binary=binary_format)
        #model.wv.save_word2vec_format(parameters["word2vec"]["model_name"], binary=True)
    
    embedding_weights=[]
    embedding_weights = np.zeros(shape=(len(vocabulary_inv)+1, vector_size), dtype='float32')            
    embedding_weights[0] = np.zeros(vector_size, dtype='float32')
    word_idx_map = dict()
    i=1
    wordInModel=0
    for word in vocabulary_inv: 
        if word not in model:
           rand_vector = np.random.uniform(-0.25, 0.25, vector_size)
           embedding_weights[i]=rand_vector
        else:
            wordInModel +=1
            embedding_weights[i] = model[word]
        word_idx_map[word] = i
        i += 1
    print("Number of words already in word2vec model :" + str(wordInModel))
    ## add unknown words
    #embedding_weights = [np.array([model[w] if w in model \
    #                                   else np.random.uniform(-0.25, 0.25, model.vector_size) \
    #                               for w in vocabulary_inv])]
    return embedding_weights,word_idx_map


#def train_word2vec(sentence_matrix, vocabulary_inv,
#                   num_features=300, min_word_count=1, context=5):
#    """
#    Trains, saves, loads Word2Vec model
#    Returns initial weights for embedding layer.
   
#    inputs:
#    sentence_matrix # int matrix: num_sentences x max_sentence_len
#    vocabulary_inv  # dict {str:int}
#    num_features    # Word vector dimensionality                      
#    min_word_count  # Minimum word count                        
#    context         # Context window size 
#    """
#    model_dir = 'models'
#    model_name = "{:d}features_{:d}minwords_{:d}context".format(num_features, min_word_count, context)
#    model_name = join(model_dir, model_name)
#    if exists(model_name):
#        embedding_model = word2vec.Word2Vec.load(model_name)
#        print('Loading existing Word2Vec model \'%s\'' % split(model_name)[-1])
#    else:
#        # Set values for various parameters
#        num_workers = 2  # Number of threads to run in parallel
#        downsampling = 1e-3  # Downsample setting for frequent words

#        # Initialize and train the model
#        print('Training Word2Vec model...')
#        sentences = [[vocabulary_inv[w] for w in s] for s in sentence_matrix]
#        embedding_model = word2vec.Word2Vec(sentences, workers=num_workers, \
#                                            size=num_features, min_count=min_word_count, \
#                                            window=context, sample=downsampling)

#        # If we don't plan to train the model any further, calling 
#        # init_sims will make the model much more memory-efficient.
#        embedding_model.init_sims(replace=True)

#        # Saving the model for later use. You can load it later using Word2Vec.load()
#        if not exists(model_dir):
#            os.mkdir(model_dir)
#        print('Saving Word2Vec model \'%s\'' % split(model_name)[-1])
#        embedding_model.save(model_name)

#    # add unknown words
#    embedding_weights = [np.array([embedding_model[w] if w in embedding_model \
#                                       else np.random.uniform(-0.25, 0.25, embedding_model.vector_size) \
#                                   for w in vocabulary_inv])]
#    return embedding_weights


#if __name__ == '__main__':
#    import data_helpers

#    print("Loading data...")
#    x, _, _, vocabulary_inv = data_helpers.load_data()
#    w = train_word2vec(x, vocabulary_inv)