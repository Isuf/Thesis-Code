import numpy as np
import pickle
from collections import defaultdict
import sys, re
import pandas as pd

import gensim

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
    pos_file = data_folder[0]
    neg_file = data_folder[1]
    vocab = defaultdict(float)
    with open(pos_file, "r",encoding="utf-8",errors="ignore") as f:
        for line in f:       
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            #orig_rev= ' '.join([word for word in orig_rev.split() if word not in cachedStopWords])
            #orig_rev=orig_rev[0:400]
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum  = {"y":1, 
                      "text": orig_rev,#' '.join([word for word in orig_rev.split() if word not in cachedStopWords]),                             
                      "num_words": len(orig_rev.split()),
                      "split": np.random.randint(0,cv)}
            revs.append(datum)
    with open(neg_file, "r",encoding="utf-8",errors="ignore") as f:
        for line in f:       
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            #orig_rev= ' '.join([word for word in orig_rev.split() if word not in cachedStopWords])
            #orig_rev=orig_rev[0:400]
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum  = {"y":0, 
                      "text":orig_rev, #' '.join([word for word in orig_rev.split() if word not in cachedStopWords]),                             
                      "num_words": len(orig_rev.split()),
                      "split": np.random.randint(0,cv)}
            revs.append(datum)
    return revs, vocab
    
def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size =len(word_vecs)#word_vecs.syn0.shape[0] #len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k), dtype='float32')            
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs: #.vocab:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = list(map(int, header.split()))
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in range(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)   
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
            else:
                f.read(binary_len)
    return word_vecs

def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)  

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

def num_words_present_in_model(keyedvector_model, vocab):
    num =0
    for word in vocab:
        if keyedvector_model.__contains__(word):
            num = num+1
    return num

def create_dict_from_word2vec(w2v_model):
      word_vectors={}
      for word in w2v_model.vocab:
        word_vectors[word] = w2v_model[word]
      return word_vectors

if __name__=="__main__":    

    parameters= config_file.Parameters()
    param = parameters.param
    vectors="TOData"  #Google  #TOData #Glove
    picklFileName = "mr_train_"+vectors +".p"
    glove_vectors = "D:\Tema NTNU\Data\GloveVectors\glove.6B.300d.txt"
    google_w2v_vectors = "D:\Tema NTNU\Data\Google Word Vectors\GoogleNews-vectors-negative300.bin" 
    trained_on_data_vectors = "w2c_hf_posts.bin"
    binary_format=True
    positive_file = param["positive_data_location"]
    #positive_file=param["data"]
    negative_file = param["negative_data_location"]
    data_folder = [positive_file,negative_file]    
 
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
    
