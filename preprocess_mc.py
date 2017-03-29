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
                words = set(orig_rev.split())
                for word in words:
                    vocab[word] += 1
                datum  = {"y":i, 
                          "text": orig_rev,#' '.join([word for word in orig_rev.split() if word not in cachedStopWords]),                             
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
    
    return string.strip() if TREC else string.strip().lower()

def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)   
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip().lower()

topics={"0":"accuracy_garmin_nuvi_255W_gps.txt.data",
        "1":"bathroom_bestwestern_hotel_sfo.txt.data",
        "2":"battery-life_amazon_kindle.txt.data",
        "3":"battery-life_ipod_nano_8gb.txt.data",
        "4":"battery-life_netbook_1005ha.txt.data",
        "5":"buttons_amazon_kindle.txt.data",
        "6":"comfort_honda_accord_2008.txt.data",
        "7":"comfort_toyota_camry_2007.txt.data",
        "8":"directions_garmin_nuvi_255W_gps.txt.data",
        "9":"display_garmin_nuvi_255W_gps.txt.data",
        }
path ="D:\\Tema NTNU\\Data\\Opinosis\\topics"
for i in range(len(topics)):
    topics[str(i)] = os.path.join(path,  topics[str(i)])

if __name__=="__main__":    

    parameters= config_file.Parameters()
    param = parameters.param
    vectors="Glove"  #Google  #TOData #Glove
    picklFileName = "mr_train_"+vectors +".p"
    glove_vectors = "D:\Tema NTNU\Data\GloveVectors\glove.6B.300d.txt"
    google_w2v_vectors = "D:\Tema NTNU\Data\Google Word Vectors\GoogleNews-vectors-negative300.bin" 
    trained_on_data_vectors = "w2c_hf_posts.bin"
    binary_format=True
    positive_file = param["positive_data_location"]
    negative_file = param["negative_data_location"]
    #data_folder = [positive_file,negative_file]    
    data_folder = [topics["0"],topics["1"],topics["2"],topics["3"],topics["4"],topics["5"],topics["6"],topics["7"],topics["8"],topics["9"]]  
 
    if (vectors=="Glove"):
        w2v_file=glove_vectors
        binary_format=False
    elif(vectors=="Google"):
        w2v_file=google_w2v_vectors
    elif(vectors=="TOData"):
        w2v_file=trained_on_data_vectors

    print("loading data...", end=' ')        
    revs, vocab = build_data_cv(data_folder, cv=10, clean_string=True)
    max_l = np.max(pd.DataFrame(revs)["num_words"])
    print("data loaded!")
    print("number of sentences: " + str(len(revs)))
    print("vocab size: " + str(len(vocab)))
    print("max sentence length: " + str(max_l))
    print("loading "+ vectors + " vectors...", end=' ')

    #txt =[]
    #for item in revs:
    #    txt.append(item["text"])
    #print(txt)
    #pickle.dump(txt, open("rawtext.txt","wb"))
    W, word_idx_map =w2v.build_word2vec(len(revs),max_l,revs,vocab,name=vectors,binary_format=binary_format,vector_size=vec_size)
    #w2v = gensim.models.KeyedVectors.load_word2vec_format(w2v_file, binary=binary_format)   
    ##w2v = load_bin_vec(w2v_file, vocab)
    #print("word2vec loaded!")
    ##print(w2v.syn0.shape)
    #n=num_words_present_in_model(w2v, vocab)
    #print("num words already in word2vec: " + str(n))
    ##print("num words already in word2vec: " + str(len(w2v)))
    
    #w2v_dict = create_dict_from_word2vec(w2v)
    #add_unknown_words(w2v_dict, vocab)
    #W, word_idx_map = get_W(w2v_dict,k=vec_size)
    
    rand_vecs = {}
    #add_unknown_words(rand_vecs, vocab)
    #W2, _ = get_W(rand_vecs)
    W2=[]
    pickle.dump([revs, W, W2, word_idx_map, vocab,max_l,vectors], open(picklFileName, "wb"))
    print("dataset created!")
    
