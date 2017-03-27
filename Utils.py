import pickle

''' Loads a file and saves its content as a list 
       [ "this is test 1", " this is test 2"]
'''
def loadfile(fileName): 
    with open(fileName,encoding="utf8",errors='ignore') as f:
         content = f.readlines()
         content = [x for x in content] 
    return content


''' get Data and Labels '''
def get_data_and_labels(data_file, labels_file, Dataset_Name=""):
    print("Loading the Data  " +Dataset_Name +" ...\n")
    data = loadfile(data_file)
    #labels = loadfile(labels_file)
    return data, labels_file

''' Write a list to a local file ''' 
def write_list_to_file(fileName, list, mode="w"):
    with open(fileName,mode) as f:
         for item in list:
             f.write(item+'\n')

''' Write the "text" to a local file '''
def write_to_file(fileName, text, mode="w"):
    with open(fileName,mode, encoding='utf-8') as f:
         f.write(text) # python will convert \n to os.linesep 


def save_pickle(self, path):
    with open(path, 'wb') as f:
        pickle.dump(self, f)
    #logger.info('save model to path %s' % path)
    return None
def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)