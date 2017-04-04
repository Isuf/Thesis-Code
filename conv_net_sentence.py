"""
Sample code for
Convolutional Neural Networks for Sentence Classification
http://arxiv.org/pdf/1408.5882v2.pdf

Much of the code is modified from
- deeplearning.net (for ConvNet classes)
- https://github.com/mdenil/dropout (for dropout)
- https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
"""
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

#different non-linearities
def ReLU(x):
    y = T.maximum(0.0, x)
    return(y)
def Sigmoid(x):
    y = T.nnet.sigmoid(x)
    return(y)
def Tanh(x):
    y = T.tanh(x)
    return(y)
def Iden(x):
    y = x
    return(y)

vec_size=300
       
def train_conv_net(vectors,i,datasets,
                   U,
                   img_w=300, 
                   filter_hs=[3,4,5],
                   hidden_units=[100,2], 
                   dropout_rate=[0.5],
                   shuffle_batch=True,
                   n_epochs=25, 
                   batch_size=50, 
                   lr_decay = 0.95,
                   conv_non_linear="relu",
                   activations=[Iden],
                   sqr_norm_lim=9,
                   non_static=True,):
    """
    Train a simple conv net
    img_h = sentence length (padded where necessary)
    img_w = word vector length (300 for word2vec)
    filter_hs = filter window sizes    
    hidden_units = [x,y] x is the number of feature maps (per filter window), and y is the penultimate layer
    sqr_norm_lim = s^2 in the paper
    lr_decay = adadelta decay parameter
    """    
    rng = np.random.RandomState(3435)
    img_h = len(datasets[0][0])-1  
    filter_w = img_w    
    feature_maps = hidden_units[0]
    filter_shapes = []
    pool_sizes = []
    for filter_h in filter_hs:
        filter_shapes.append((feature_maps, 1, filter_h, filter_w))
        pool_sizes.append((img_h-filter_h+1, img_w-filter_w+1))
    parameters = [("image shape",img_h,img_w),("filter shape",filter_shapes), ("hidden_units",hidden_units),
                  ("dropout", dropout_rate), ("batch_size",batch_size),("non_static", non_static),
                    ("learn_decay",lr_decay), ("conv_non_linear", conv_non_linear), ("non_static", non_static)
                    ,("sqr_norm_lim",sqr_norm_lim),("shuffle_batch",shuffle_batch)]
    print(parameters)    
    
    #define model architecture
    index = T.lscalar()
    x = T.matrix('x')   
    y = T.ivector('y')
    Words = theano.tensor._shared(value = U, name = "Words")
    zero_vec_tensor = T.vector()
    zero_vec = np.zeros(img_w)
    set_zero = theano.function([zero_vec_tensor], updates=[(Words, T.set_subtensor(Words[0,:], zero_vec_tensor))], allow_input_downcast=True)
   
    layer0_input = Words[T.cast(x.flatten(),dtype="int32")].reshape((x.shape[0],1,x.shape[1],Words.shape[1]))                                  
    conv_layers = []
    layer1_inputs = []
    for i in range(len(filter_hs)):
        filter_shape = filter_shapes[i]
        pool_size = pool_sizes[i]
        conv_layer = LeNetConvPoolLayer(rng, input=layer0_input,image_shape=(batch_size, 1, img_h, img_w),
                                filter_shape=filter_shape, poolsize=pool_size, non_linear=conv_non_linear)
        layer1_input = conv_layer.output.flatten(2)
        conv_layers.append(conv_layer)
        layer1_inputs.append(layer1_input)
    layer1_input = T.concatenate(layer1_inputs,1)
    hidden_units[0] = feature_maps*len(filter_hs)    
    classifier = MLPDropout(rng, input=layer1_input, layer_sizes=hidden_units, activations=activations, dropout_rates=dropout_rate)
    
    #define parameters of the model and update functions using adadelta
    params = classifier.params     
    for conv_layer in conv_layers:
        params += conv_layer.params
    if non_static:
        #if word vectors are allowed to change, add them as model parameters
        params += [Words]
    cost = classifier.negative_log_likelihood(y) 
    dropout_cost = classifier.dropout_negative_log_likelihood(y)           
    grad_updates = sgd_updates_adadelta(params, dropout_cost, lr_decay, 1e-6, sqr_norm_lim)
    
    #shuffle dataset and assign to mini batches. if dataset size is not a multiple of mini batches, replicate 
    #extra data (at random)
    np.random.seed(3435)
    if datasets[0].shape[0] % batch_size > 0:
        extra_data_num = batch_size - datasets[0].shape[0] % batch_size
        train_set = np.random.permutation(datasets[0])   
        extra_data = train_set[:extra_data_num]
        new_data=np.append(datasets[0],extra_data,axis=0)
    else:
        new_data = datasets[0]
    new_data = np.random.permutation(new_data)
    n_batches = new_data.shape[0]/batch_size
    n_train_batches = int(np.round(n_batches*0.9))
    #divide train set into train/val sets 
    test_set_x = datasets[1][:,:img_h] 
    test_set_y = np.asarray(datasets[1][:,-1],"int32")

    #test_set_x, test_set_y = shared_dataset((datasets[1][:,:img_h],datasets[1][:,-1]))
    train_set = new_data[:n_train_batches*batch_size,:]
    val_set = new_data[n_train_batches*batch_size:,:]     
    train_set_x, train_set_y = shared_dataset((train_set[:,:img_h],train_set[:,-1]))
    val_set_x, val_set_y = shared_dataset((val_set[:,:img_h],val_set[:,-1]))
    n_val_batches = int(n_batches - n_train_batches)\

    val_model = theano.function([index], classifier.errors(y),
         givens={
            x: val_set_x[index * batch_size: (index + 1) * batch_size],
             y: val_set_y[index * batch_size: (index + 1) * batch_size]},
                                allow_input_downcast=True)        
    #compile theano functions to get train/val/test errors
    test_model = theano.function([index], classifier.errors(y),
             givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                 y: train_set_y[index * batch_size: (index + 1) * batch_size]},
                                 allow_input_downcast=True)               
    train_model = theano.function([index], cost, updates=grad_updates,
          givens={
            x: train_set_x[index*batch_size:(index+1)*batch_size],
              y: train_set_y[index*batch_size:(index+1)*batch_size]},
                                  allow_input_downcast = True)   
    
    test_pred_layers = []
    test_size = test_set_x.shape[0]
    test_layer0_input = Words[T.cast(x.flatten(),dtype="int32")].reshape((test_size,1,img_h,Words.shape[1]))
    for conv_layer in conv_layers:
        test_layer0_output = conv_layer.predict(test_layer0_input, test_size)
        test_pred_layers.append(test_layer0_output.flatten(2))
    test_layer1_input = T.concatenate(test_pred_layers, 1)
    test_y_pred = classifier.predict(test_layer1_input)
    test_error = T.mean(T.neq(test_y_pred, y))

    test_model_all = theano.function([x,y],
                                     [test_error,test_y_pred],
                                     allow_input_downcast = True)   
    
    #start training over mini-batches
    print('... training')
    epoch = 0
    best_val_perf = 0
    val_perf = 0
    test_perf = 0       
    cost_epoch = 0    
    while (epoch < n_epochs):
        start_time = time.time()
        epoch = epoch + 1
        if shuffle_batch:
            for minibatch_index in np.random.permutation(list(range(n_train_batches))):
                cost_epoch = train_model(minibatch_index)
                set_zero(zero_vec)
        else:
            for minibatch_index in range(n_train_batches):
                cost_epoch = train_model(minibatch_index)  
                set_zero(zero_vec)
        train_losses = [test_model(i) for i in range(n_train_batches)]
        train_perf = 1 - np.mean(train_losses)
        val_losses = [val_model(i) for i in range(n_val_batches)]
        val_perf = 1- np.mean(val_losses)                        
        print(('epoch: %i, training time: %.2f secs, train perf: %.2f %%, val perf: %.2f %%'
             % (epoch, time.time()-start_time, train_perf * 100., val_perf*100.)))
        if val_perf >= best_val_perf:
            best_val_perf = val_perf
            test_loss,test_y_pred = test_model_all(test_set_x,test_set_y)        
            test_perf = 1- test_loss
        val_losses.clear()
        train_losses.clear()

    #if(i==0):
    model_name=vectors+".pkl"
    save([Words,conv_layers,params],"Models_CNN/"+model_name)


    #Free GPU Memory after running in CV (Deliu)
    Words.set_value([[]])     
    layer0_input=[]
    test_layer0_input=[]
    #test_set_x =[]
    test_set_y=[]
    val_set_x=[]
    val_set_y=[]
    train_set_x=[]
    train_set_y=[]
    
    return test_perf, test_set_x, test_y_pred




def test_unseen_data(Words,conv_layers,params,test_set_x,test_set_y,img_h,hidden_units,activations,dropout_rate):
    
    rng = np.random.RandomState(3435)
    x = T.matrix('x')
    y = T.ivector('y')
    #Words = theano.shared(value = U, name = "Words")
  
    #for param in params:
    #    print(param)
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
    test_model_all = theano.function([x,y], [test_error], allow_input_downcast = True)

    test_loss = test_model_all(test_set_x,test_set_y)       
    test_perf = 1- test_loss[0]

    #Words.set_value([[]])
    test_pred_layers=[]
    return test_perf

def save(self, path):
    with open(path, 'wb') as f:
        pickle.dump(self, f)
    #logger.info('save model to path %s' % path)
    return None
def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.tensor._shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.tensor._shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')
        
def sgd_updates_adadelta(params,cost,rho=0.95,epsilon=1e-6,norm_lim=9,word_vec_name='Words'):
    """
    adadelta update rule, mostly from
    https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
    """
    updates = OrderedDict({})
    exp_sqr_grads = OrderedDict({})
    exp_sqr_ups = OrderedDict({})
    gparams = []
    for param in params:
        empty = np.zeros_like(param.get_value())
        exp_sqr_grads[param] = theano.tensor._shared(value=as_floatX(empty),name="exp_grad_%s" % param.name)
        gp = T.grad(cost, param)
        exp_sqr_ups[param] = theano.tensor._shared(value=as_floatX(empty), name="exp_grad_%s" % param.name)
        gparams.append(gp)
    for param, gp in zip(params, gparams):
        exp_sg = exp_sqr_grads[param]
        exp_su = exp_sqr_ups[param]
        up_exp_sg = rho * exp_sg + (1 - rho) * T.sqr(gp)
        updates[exp_sg] = up_exp_sg
        step =  -(T.sqrt(exp_su + epsilon) / T.sqrt(up_exp_sg + epsilon)) * gp
        updates[exp_su] = rho * exp_su + (1 - rho) * T.sqr(step)
        stepped_param = param + step
        if (param.get_value(borrow=True).ndim == 2) and (param.name!='Words'):
            col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
            desired_norms = T.clip(col_norms, 0, T.sqrt(norm_lim))
            scale = desired_norms / (1e-7 + col_norms)
            updates[param] = stepped_param * scale
        else:
            updates[param] = stepped_param      
    return updates 

def as_floatX(variable):
    if isinstance(variable, float):
        return np.cast[theano.config.floatX](variable)

    if isinstance(variable, np.ndarray):
        return np.cast[theano.config.floatX](variable)
    return theano.tensor.cast(variable, theano.config.floatX)
    
def safe_update(dict_to, dict_from):
    """
    re-make update dictionary for safe updating
    """
    for key, val in dict(dict_from).items():
        if key in dict_to:
            raise KeyError(key)
        dict_to[key] = val
    return dict_to
    
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
  
def load_test_data(revs,cv):
    test_set = []
    for rev in revs:
        if rev["split"]==cv:
           test_set.append(rev["text"])
    return test_set

if __name__=="__main__":
    train_or_test="Train" #Train or Test
    vectors="Glove"  #Google  #TOData #Glove
    picklFileName = "mr_train_"+vectors +".p"

    print("loading data...", end=' ')
    x = pickle.load(open(picklFileName,"rb"))
    revs, W, W2, word_idx_map, vocab,max_l = x[0], x[1], x[2], x[3], x[4],x[5]
    print("data loaded!")
    
    num_class=4
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
    Words=[]
    conv_layers=[]
    params=[]
    if(train_or_test=="Test"):
       Words,conv_layers,params=load("Models_CNN/"+vectors+".pkl")

    r = list(range(0,10)) 
    test_set_x=[]
    test_y_pred=[]  
    for i in r:
        print(i)
        datasets = make_idx_data_cv(revs, word_idx_map, i, max_l=max_l,k=vec_size, filter_h=5)
        if(train_or_test=="Test"):
            
            img_h = len(datasets[0][0])-1
            test_set_x = datasets[1][:,:img_h] 
            test_set_y = np.asarray(datasets[1][:,-1],"int32")
            print("Number of test samples:" + str(len(test_set_x)))

            perf=test_unseen_data( Words,conv_layers,params,test_set_x,test_set_y,img_h,hidden_units=[100,2],activations=[Iden],dropout_rate=[0.5])
            test_set_x=[]
            test_set_y=[]
        elif(train_or_test=="Train"):
            perf,test_set_x, test_y_pred = train_conv_net(vectors,i,datasets,
                                  U,
                                  lr_decay=0.95,
                                  filter_hs=[3,4,5],
                                  conv_non_linear="relu",
                                  hidden_units=[100,num_class], 
                                  shuffle_batch=True, 
                                  n_epochs=25, #25,
                                  sqr_norm_lim=9,
                                  non_static=non_static,
                                  batch_size=50,
                                  dropout_rate=[0.5])
            dataset=[]
            test_set_x=[]
        else:
            print("The mode does not exist...")
        print("cv: " + str(i) + ", perf: " + str(perf))
        results.append(perf)  
    
    #test_set = load_test_data(revs, 4)
    #res=[]
    #print(len(test_y_pred))
    #print((test_set_x))
    #for i in range(len(test_y_pred)):
    #    stri = str(test_y_pred[i]) + "\t" + test_set[i]
    #    res.append(stri)


        
    #Utils.write_list_to_file("rez.txt",res)
    print(str(np.mean(results)))
