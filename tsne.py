import sys
import codecs
import numpy as np
import matplotlib.pyplot as plt
 
from sklearn.manifold import TSNE
import random 

import pickle
w= ["sell","buy","thank","awesome","","virus", "keylogger", "crypter", "trojan","crack","hack", "zeus","inject", "exploit", "vulnerability"]
print(random.choice(w))
def visualize_vectors(wv, vocabulary,word_idx_map, words =w):

    #w1= [random.choice(vocabulary) for i in range(100)]
    #print(w1)
    x=[]
    #for word in w1:
    #    if word in word_idx_map:
    #        x.append(word_idx_map[word])

    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])

    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(wv[50:250])
 
    plt.scatter(Y[:, 0], Y[:, 1])
    for label, x, y in zip(vocabulary, Y[:, 0], Y[:, 1]):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.show()
 
train_or_test="Train" #Train or Test
vectors="Google"  #Google  #TOData #Glove
picklFileName = "mr_train_"+vectors +".p"

print("loading data...", end=' ')
x = pickle.load(open(picklFileName,"rb"))
revs, W, W2, word_idx_map, vocab,max_l = x[0], x[1], x[2], x[3], x[4],x[5]

visualize_vectors(W,vocab,word_idx_map)