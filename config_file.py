import os

dataset_keywords = "Experiment"
path = "D:\\Tema NTNU\\Data\\"+dataset_keywords+"\\Deliu\\" 

class Parameters: 

      param = {} 

      def __init__(self):
          self.param={
                      "pos_file" : "positive_20K_2603.txt",
                      "neg_file" : "negative_20K_2603.txt",
                      "positive_data_location" : "full_path_positive_data",
                      "negative_data_location" : "full_path_negative_data",
                      "data" : "data.txt",
                      "labels": "labels.txt",

                      "word2vec" : {
                                    "Train" : True,
                                    "model_name":"w2c_hf_posts.bin",
                                    "vec_size" : 300,
                                    "min_count" : 5,
                                    "Google_w2v" : "D:\\Official_Datasets\\Google word2vec trained\\GoogleNews-vectors-negative300.bin",
                                    "use_google_w2v" : False
                       },

                     "ngrams_bow" : {
                                     "min_ngrams": 1,
                                     "max_ngrams": 2,
                                     "use_hashing": False,   # Smth for sparsity
                                     "n_features": 2 ** 16,  #n_features when using the hashing vectorizer. 
                                     "ngram_unit": "char",   #'char' for character;  'word' for word
                                     "method": "ngrams"      # "bow" for Bag-of-Words;  "ngrams" for n(1,2,...) grams
                       }
      }

          self.param["positive_data_location"]=os.path.join(path,  self.param["pos_file"])
          self.param["negative_data_location"]=os.path.join(path,  self.param["neg_file"])
          self.param["data"]=os.path.join(path,  self.param["data"])
          self.param["labels"]=os.path.join(path,  self.param["labels"])

      

#class Configuration :

#      config = {}
#      def __init__(self, use_google_w2v = True):
        
#          self.config ={
#             "pos_file" :       "positive.txt",
#             "neg_file" :       "negative.txt",
#             "positive_data_location" : "",
#             "negative_data_location" : "",
#             "all_data": "Data.txt",
#             "labels": "labels.txt",

#             "CNN_Training" :{
#                              "lr_decay":0.95,
#                              "filter_hs":[3,4,5],
#                              "conv_non_linear":"relu",
#                              "hidden_units":[100,2], 
#                              "shuffle_batch":True, 
#                              "n_epochs":25, 
#                              "sqr_norm_lim":9,
#                              "non_static":True,
#                              "batch_size":50,
#                              "dropout_rate":[0.5]

#                             } ,

#            "ngrams_bow" :{
#                             "min_ngrams": 1,
#                             "max_ngrams": 3,
#                             "use_hashing": False, # Smth for sparsity
#                             "n_features": 2 ** 16,  #n_features when using the hashing vectorizer. 
#                             "ngram_unit": "word", #'char' for character;  'word' for word
#                             "method": "ngrams"
#                            },

#              "doc2vec" :{
#                             "do_train": True,
#                             "show_progress": False
#                            }
#                       }

#          self.config["positive_data_location"]=os.path.join(path,  self.config["pos_file"])
#          self.config["negative_data_location"]=os.path.join(path,  self.config["neg_file"])
#          self.config["all_data"]=os.path.join(path,  self.config["all_data"])
#          self.config["labels"]=os.path.join(path,  self.config["labels"])
#          if (use_google_w2v==True):
#             self.config["word2vec"]["model_name"]=self.config["word2vec"]["Google_w2v"]
#             self.config["word2vec"]["Train"]=False



          
          