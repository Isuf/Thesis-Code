
from ShallowClassifiers import ngrams_and_bow
from ShallowClassifiers import doc2vec 
from TopicModels import LDA
import config_file 
import Utils

from TopicModels import LDA

#http://stackoverflow.com/questions/40824903/unorderable-types-error-when-importing-sklearn
#### CHANGES MADE TO SKLEARN 

#parameters= config_file.Parameters()
#param = parameters.param

#labels =[ 1 if i<10000 else 0 for i in range(20000)]
#ngrams_and_bow.run_classification(param["data"],labels,param)
###doc2vec.run_classification(param["data"],labels,"Movie Review")


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

dataset_locations ={ 
    "une": ("D:\\Tema NTNU\\Data\\Nulled\\Deliu\\positive.txt","")
    }

print("Loading dataset...")
#original_dataset = Utils.loadfile("rez.txt")#dataset_locations["une"][0])

relevant_data,irrelevant_data=seperate_relevant_data("rez.txt")
''' Topic Modeling ''' 
lda,tf,tf_vectorizer=LDA.run_LDA(relevant_data,relevant_data,num_topics=5)



