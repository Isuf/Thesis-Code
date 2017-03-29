
from ShallowClassifiers import ngrams_and_bow
from ShallowClassifiers import doc2vec 
from TopicModels import LDA
import config_file 
import Utils

from TopicModels import LDA
import pickle
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer

#http://stackoverflow.com/questions/40824903/unorderable-types-error-when-importing-sklearn
#### CHANGES MADE TO SKLEARN 

parameters= config_file.Parameters()
param = parameters.param

labels =[ 1 if i<10000 else 0 for i in range(20000)]
ngrams_and_bow.run_classification(param["data"],labels,param)
#doc2vec.run_classification(param["data"],labels,"Movie Review")


#vectorizer = TfidfVectorizer( stop_words='english', 
#                                       sublinear_tf=True, max_df=0.95,
#                                       analyzer=param["ngrams_bow"]["ngram_unit"],
#                                       ngram_range=(param["ngrams_bow"]["min_ngrams"],param["ngrams_bow"]["max_ngrams"]))

#clf = Utils.load_pickle("Models_CNN/clf.pkl")
#testData=Utils.loadfile(param["data"])
#test_data=vectorizer.transform(testData)
#print(test_data)
#pred=clf.predict(test_data)
#score = metrics.accuracy_score(labels, pred)
#print("accuracy:   %0.3f" % score)

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

#dataset_locations ={ 
#    "une": ("D:\Tema NTNU\Data\Experiment\Deliu\positive.txt","")
#    }

#print("Loading dataset...")
#original_dataset = Utils.loadfile(dataset_locations["une"][0])

##relevant_data,irrelevant_data=seperate_relevant_data("rez.txt")
#''' Topic Modeling ''' 
##lda,tf,tf_vectorizer=LDA.run_LDA(relevant_data,relevant_data,num_topics=10)
#lda,tf,tf_vectorizer=LDA.run_LDA(original_dataset,original_dataset,num_topics=10)



