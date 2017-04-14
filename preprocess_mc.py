import numpy as np
import pickle
from collections import defaultdict
import sys, re
import pandas as pd

import gensim
import os

from nltk.corpus import stopwords
import config_file
cachedStopWords = stopwords.words("english")

vec_size=300
import w2v
def build_data_cv(data_folder, cv=10, clean_string=True):
    """
    Loads data and split into 10 folds.
    """
    revs = []
    vocab = defaultdict(float)
    for i in range(len(data_folder)):
        with open(data_folder[i], "r",encoding="utf-8",errors="ignore") as f:
            for line in f:
                rev =[]
                rev.append(line.strip())
                if clean_string:
                    orig_rev = clean_str(" ".join(rev))
                else:
                    orig_rev = " ".join(rev).lower()
                orig_rev=orig_rev[0:800]
                words = set(orig_rev.split())
                for word in words:
                    vocab[word] += 1
                datum  = {"y":i, 
                          "text": ' '.join([word for word in orig_rev.split() if word not in cachedStopWords]), #orig_rev,#                            
                          "num_words": len(orig_rev.split()),
                          "split": np.random.randint(0,cv)}
                revs.append(datum)

    return revs, vocab

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = string.replace("&nbsp"," " )
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string) 

    #Deliu
    string = string.replace("\n","")   
    
    return string.strip() if TREC else string.strip().lower()

def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)   
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip().lower()




def create_ds(data_folder,param,picklFileName,glove_vectors,google_w2v_vectors,trained_on_data_vectors,vectors="Google",binary_format=True):
    print("\n\n\nCreating the dataset ...")
    if (vectors=="Glove"):
        w2v_file=glove_vectors
        binary_format=False
    elif(vectors=="Google"):
        w2v_file=google_w2v_vectors
    elif(vectors=="TOData"):
        w2v_file=trained_on_data_vectors
    print(w2v_file)
    print("loading data...", end=' ')        
    revs, vocab = build_data_cv(data_folder, cv=10, clean_string=True)
    max_l = np.max(pd.DataFrame(revs)["num_words"])
    print("data loaded!")
    print("         number of sentences: " + str(len(revs)))
    print("         vocab size: " + str(len(vocab)))
    print("         max sentence length: " + str(max_l))
    print("         loading "+ vectors + " vectors...", end=' ')

    print(" Loading Vectors") 
    W, word_idx_map =w2v.build_word2vec(len(revs),max_l,revs,vocab,name=vectors,binary_format=binary_format,vector_size=vec_size,context=10)
    ##import tsne 
    ##tsne.visualize_vectors(W,vocab,word_idx_map)
    
    rand_vecs = {}
    print(" Generating random vectors...")
    W2=w2v.get_random_vectors(rand_vecs,vocab,300)

    pickle.dump([revs, W, W2, word_idx_map, vocab,max_l,vectors], open(picklFileName, "wb"))
    print("dataset created!")
    
