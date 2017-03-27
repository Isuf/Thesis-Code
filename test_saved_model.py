import pickle
import numpy as np
from collections import defaultdict, OrderedDict
import theano
import theano.tensor as T
import re
import warnings
import sys
import time
warnings.filterwarnings("ignore")  
import Utils 
def test_unseen_data(U,test_set_x,test_set_y,img_h,hidden_units,activations,dropout_rate):
    
    rng = np.random.RandomState(3435)
    x = T.matrix('x')
    y = T.ivector('y')
    Words,conv_layers,params=load("Models_CNN/modeli2.pkl")
    layer1_inputs=[]
    for conv_layer in conv_layers:
        #conv_layer.params.W_conv = params.W_conv
        layer1_input = conv_layer.output.flatten(2)
        layer1_inputs.append(layer1_input)
    layer1_input = T.concatenate(layer1_inputs,1)   
    classifier = MLPDropout(rng, input=layer1_input, layer_sizes=hidden_units, activations=activations, dropout_rates=dropout_rate)
    for i in range(len(classifier.params)):
        classifier.params[i].set_value(params[i].get_value())

    test_pred_layers = []
    test_size = test_set_x.shape[0]
    test_layer0_input = Words[T.cast(x.flatten(),dtype="int32")].reshape((test_size,1,img_h,Words.shape[1]))
    for conv_layer in conv_layers:
        test_layer0_output = conv_layer.predict(test_layer0_input, test_size)
        test_pred_layers.append(test_layer0_output.flatten(2))
    test_layer1_input = T.concatenate(test_pred_layers, 1)
    test_y_pred = classifier.predict(test_layer1_input)
    test_error = T.mean(T.neq(test_y_pred, y))
    test_model_all = theano.function([x,y], [test_error,test_y_pred], allow_input_downcast = True)

    test_loss,test_y_pred = test_model_all(test_set_x,test_set_y)        
    test_perf = 1- test_loss  


    #test_loss,test_y_pred = test_model_all(test_set_x,test_set_y)       
    #test_perf = 1- test_loss[0]
    Words.set_value([[]])
    return test_perf,test_y_pred

# Transforms a sentence to list of indexes 
# "I like Norway " ==> [0,0,0,120,500,275,0,0,...,max_l]
def get_idx_from_sent(sent, word_idx_map, max_l=51, k=300, filter_h=5):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    pad = filter_h - 1
    for i in range(pad):
        x.append(0)
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < max_l+2*pad:
        x.append(0)
    return x

def make_idx_data_testing(revs,cv, word_idx_map,max_l=51, k=300, filter_h=5):
    """
    Transforms sentences into a 2-d matrix.
    """
    print("Transforms sentences into a 2-d matrix.")
    test_matrix = []
    test_data=[]
    for rev in revs:
        sent = get_idx_from_sent(rev["text"], word_idx_map, max_l, k, filter_h)
        sent.append(rev["y"])
        if rev["split"]==cv:          
            test_matrix.append(sent)
            test_data.append(rev["text"])

    test_matrix = np.array(test_matrix,dtype="int")
    return test_matrix,test_data

def make_idx_data_cv(revs, word_idx_map, cv, max_l=51,split_stentence_length=100, k=300, filter_h=5):
    """
    Transforms sentences into a 2-d matrix.
    """
    print("Transforms sentences into a 2-d matrix.")
    train, test = [], []
    for rev in revs:
        sent = get_idx_from_sent(rev["text"], word_idx_map, max_l, k, filter_h)  
        #sent =sent[0:split_stentence_length]
        sent.append(rev["y"])
        if rev["split"]==cv:            
            test.append(sent)        
        else:  
            train.append(sent)   
    train = np.array(train,dtype="int")
    test = np.array(test,dtype="int")
    return [train, test]     
  
def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_test_data(revs,cv):
    test_set = []
    for rev in revs:
        if rev["split"]==cv:
           test_set.append(rev["text"])
    return test_set

def load_test_data(revs,cv):
    test_set = []
    for rev in revs:
        if rev["split"]==cv:
           test_set.append(rev["text"])
    return test_set


if __name__=="__main__":
    print("loading data...", end=' ')
    x = pickle.load(open("mr_test.p","rb"))
    revs, W, W2, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4]
    print("data loaded!")
    
    mode= "-static" #sys.argv[1]
    word_vectors = "-word2vec" #sys.argv[2]
    
    if mode=="-nonstatic":
        print("model architecture: CNN-non-static")
        non_static=True
    elif mode=="-static":
        print("model architecture: CNN-static")
        non_static=False
    exec(compile(open("conv_net_classes.py").read(), "conv_net_classes.py", 'exec'))    
    if word_vectors=="-rand":
        print("using: random vectors")
        U = W2
    elif word_vectors=="-word2vec":
        print("using: word2vec vectors")
        U = W
    results = []

    test_set_x=[]
    test_y_pred=[]  
    results=[]
    data_to_write=[]
    for i in range(10):
        #datasets = make_idx_data_cv(revs, word_idx_map, i, max_l=96,k=300, filter_h=5)
        #img_h = len(datasets[0][0])-1
        #test_set_x = datasets[1][:,:img_h] 
        #test_set_y = np.asarray(datasets[1][:,-1],"int32")
        #print("test sample size: " +str(len(test_set_x)))
        #test_perf,test_y_pred=test_unseen_data(U,test_set_x,test_set_y,img_h,hidden_units=[100,2],activations=[Iden],dropout_rate=[0.5])
        testing_datasets,data_samples = make_idx_data_testing(revs,i,word_idx_map, max_l=96,k=300, filter_h=5)
        img_h = len(testing_datasets[0])-1
        test_set_x = testing_datasets[:,:img_h] 
        test_set_y = np.asarray(testing_datasets[:,-1],"int32")
        print("test sample size: " +str(len(test_set_x)))
        data_to_write.append(data_samples)

        test_perf,y_pred=test_unseen_data(U,test_set_x,test_set_y,img_h,hidden_units=[100,2],activations=[Iden],dropout_rate=[0.5])
        test_y_pred.append(y_pred)
        print("cv: " + str(i) + ", perf: " + str(test_perf))
        results.append(test_perf)  
    
    print(str(np.mean(results)))

    tmp_dataa=[]
    for testbatch in data_to_write:
        for i in range(len(testbatch)):
            tmp_dataa.append(testbatch[i])

    tmp_predy=[]
    for testbatch in test_y_pred:
        for i in range(len(testbatch)):
            tmp_predy.append(testbatch[i])

    res=[]
    for i in range(len(tmp_predy)):
        stri = str(tmp_predy[i]) + "\t" + tmp_dataa[i]#["text"]
        res.append(stri)


        
    Utils.write_list_to_file("rez.txt",res)
