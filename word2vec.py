
from gensim.models import word2vec
import Utils

def build_word2vec_model(parameters):
    
    print("building w2v model")
  
    loaded_data = Utils.loadfile(parameters["data"])

    sentences=[]
    for sent in loaded_data:
        tokens = sent.split()
        sentences.append(tokens)

    ''' It means that word2vec model does not exist, or a new one should be created(learned) '''
    if (parameters["word2vec"]["Train"]==True):

        model = word2vec.Word2Vec(sentences, size=parameters["word2vec"]["vec_size"],min_count=parameters["word2vec"]["min_count"])
        model.wv.save_word2vec_format(parameters["word2vec"]["model_name"], binary=True)
   

        ''' The model already exists. It can be either Google's model,or one trained on the actual data'''
    else:
        if( parameters["word2vec"]["use_google_w2v"]==True):
           model_name = parameters["word2vec"]["Google_w2v"]
        else:
           model_name = parameters["word2vec"]["model_name"]

        model = word2vec.KeyedVectors.load_word2vec_format(model_name, binary=True)
        model.init_sims(replace=True)

    print("       w2v model built")
    return model,loaded_data


import config_file 

confcls = config_file.Parameters()
param = confcls.param

w2v_model,_ =build_word2vec_model(param)
print(w2v_model.most_similar("film"))
