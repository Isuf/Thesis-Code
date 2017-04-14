from ShallowClassifiers import ngrams_and_bow
from ShallowClassifiers import doc2vec 
from TopicModels import LDA
import config_file 
import Utils

from TopicModels import LDA
import pickle
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer

import conv_net_sentence as ConvNN
import preprocess_mc
import os 
#http://stackoverflow.com/questions/40824903/unorderable-types-error-when-importing-sklearn
#### CHANGES MADE TO SKLEARN

import Distributional_Representation
'''
    CNN   for Convelutional Neural Networks 
    'distribuional' for others 

'''
classification_method="LDA"  #distribuional #CNN #LDA

train_or_test="Train" #Train or Test
vectors="Google"  #Google  #TOData #Glove
picklFileName = "mr_train_"+vectors +".p"
glove_vectors = "D:\Tema NTNU\Data\GloveVectors\glove.6B.300d.txt"
google_w2v_vectors = "D:\Tema NTNU\Data\Google Word Vectors\GoogleNews-vectors-negative300.bin" 
trained_on_data_vectors = "w2c_hf_posts.bin"
binary_format=True

parameters= config_file.Parameters()
param = parameters.param



topics={"0":"Credentials_3000.txt",
        "1":"Crypters_500.txt",
        "2":"DDOS_500.txt",
        "3":"Keylogger_500.txt",
        "4":"NoSecurity_4000.txt",
        "5":"Trojan_200.txt",
        "6": "Binary_Class_0_10K.txt",
        "7":"Binary_Class_1_10K.txt"
        }
path ="D:\\Tema NTNU\\Data\\Classification"
for i in range(len(topics)):
    topics[str(i)] = os.path.join(path,  topics[str(i)])   
data_folder = [topics["0"],topics["1"],topics["2"],topics["3"],topics["4"],topics["5"]] 

#data_folder=["D:\\Tema NTNU\\Data\\Vector_Models\\all_distinct_data_from_nulled.txt"]
##data_folder = [topics["5"],topics["6"]] 
''' Create the dataset first '''
#preprocess_mc.create_ds(data_folder,param,picklFileName,glove_vectors,google_w2v_vectors,trained_on_data_vectors,vectors,binary_format=binary_format)


if(classification_method=="CNN"):
    exec(compile(open("conv_net_sentence.py").read(), "conv_net_sentence.py", 'exec'))   

elif(classification_method=="distribuional"):
    '''
        bow : bag-of-words 
        ngrams : n-grams (word or character level; Check config file) 
        lsa : for lsa classification
    '''
    feature_construction_method = "ngrams"  

    print("loading data...", end=' ')
    x = pickle.load(open(picklFileName,"rb"))
    revs, W, W2, word_idx_map, vocab,max_l = x[0], x[1], x[2], x[3], x[4],x[5]
    print("data loaded!")
    #print("         number of sentences: " + str(len(revs)))
    #print("         vocab size: " + str(len(vocab)))
    #print("         max sentence length: " + str(max_l))
    #print("         loading "+ vectors + " vectors...", end=' ')
    labels = [sent["y"] for sent in revs]
    data = [ sent["text"] for sent in revs]
    #ngrams_and_bow.run_classification(data,labels,param)
    Distributional_Representation.run_document_classification(data,labels,feature_construction_method,param,True)

elif(classification_method=="LDA"):
    dataset_locations ={ 
   # "une": ("D:\\Tema NTNU\\Data\\Vector_Models\\all_distinct_data_from_nulled.txt","")
     "une":("D:\\Tema NTNU\\Data\\Classification\\NoSecurity_4000.txt","")
    }

    print("Loading dataset...")
    original_dataset = Utils.loadfile(dataset_locations["une"][0])
    print(len(original_dataset))
    lda,tf,tf_vectorizer=LDA.run_LDA(original_dataset,original_dataset,num_topics=20)

elif(classification_method=="doc2vec"):
    doc2vec.run_classification(param["data"],labels,"Movie Review")


def seperate_relevant_data(all_test_data):
    relevant_data=[]
    irrelevant_data=[]
    with open(all_test_data,encoding="utf8",errors='ignore') as f:
         content = f.readlines()
         for x in content:
             if(int(x[0]) ==1):
                relevant_data.append(x)
             else:
                irrelevant_data.append(x)
    return relevant_data,irrelevant_data





