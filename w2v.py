from __future__ import print_function
#from gensim.models import word2vec
from os.path import join, exists, split
import os
import numpy as np
import gensim
#https://github.com/alexander-rakhlin/CNN-for-Sentence-Classification-in-Keras/blob/master/data_helpers.py

def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)
def get_random_vectors(word_vectors_model,vocabulary,vector_size=300):
    word_idx_map = dict()
    i=1
    wordInModel=0
    embedding_weights=[]
    embedding_weights = np.zeros(shape=(len(vocabulary)+1, vector_size), dtype='float32')            
    embedding_weights[0] = np.zeros(vector_size, dtype='float32')
    word_idx_map = dict()
    for word in vocabulary: 
        if word not in word_vectors_model:
           rand_vector = np.random.uniform(-0.25, 0.25, vector_size)
           embedding_weights[i]=rand_vector
        else:
            wordInModel +=1
            embedding_weights[i] = word_vectors_model[word]
        word_idx_map[word] = i
        i += 1
    print("Number of words already in word2vec model :" + str(wordInModel))
    return embedding_weights

def build_word2vec(num_sentences,max_sentence_len,data, vocabulary_inv, name="Google",binary_format=True,vector_size=300, min_word_count=1, context=5 ):
    a=5
    model_dir = 'D:\Tema NTNU\Data\Vector_Models'
    model_name="{:s}_word2vec_{:d}".format(name,vector_size)
    model_name = join(model_dir, model_name+ (".bin" if binary_format else ".txt"))
    if exists(model_name):
        print(" Model already exists")
        model =gensim.models.KeyedVectors.load_word2vec_format(model_name, binary=binary_format)   
        print('Loading existing Word2Vec model \'%s\'' % split(model_name)[-1])
    else:
        print(" Building word2vec model from data :" )
        print( "Number of Sentences : "+str(num_sentences))
        print( "Max Sentence Length : "+str(max_sentence_len))
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


