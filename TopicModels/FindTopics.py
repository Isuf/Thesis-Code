import sys
sys.path.insert(0, 'TopicModels')

from sklearn.decomposition import NMF, LatentDirichletAllocation
import pyLDAvis
import pyLDAvis.sklearn 
pyLDAvis.enable_notebook()
import pickle 

x = pickle.load(open("lda_model_Nulled.p","rb"))
lda, tf, tf_vectorizer= x[0], x[1], x[2]
lda.fit(tf)

pyLDAvis.sklearn.prepare(lda, tf, tf_vectorizer)