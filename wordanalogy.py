# Word analogy analysis using WordVector

import warnings

from nltk.tokenize import sent_tokenize, word_tokenize 
import gensim 
from gensim.models import Word2Vec

warnings.filterwarnings('ignore')

sample = open("word_analogy.txt", "r", encoding='cp1252') 
s = sample.read() 

f = s.replace("\n", " ") 
data=[]

# iterate through each sentence in the file 
for i in sent_tokenize(f): 
    temp = [] 
      
    # tokenize the sentence into words 
    for j in word_tokenize(i): 
        temp.append(j.lower()) 
  
    data.append(temp) 

model1 = gensim.models.Word2Vec(data, min_count = 1,  
                              vector_size = 100, window = 5)

print("Cosine similarity between 'alice' " + 
               "and 'wonderland' - CBOW : ", 
    model1.wv.similarity('alice', 'wonderland')) 
      
print("Cosine similarity between 'alice' " +
                 "and 'machines' - CBOW : ", 
      model1.wv.similarity('alice', 'machines')) 

model2 = gensim.models.Word2Vec(data, min_count = 1, vector_size = 100, 
                                             window = 5, sg = 1) 

print("Cosine similarity between 'alice' " +
          "and 'wonderland' - Skip Gram : ", 
    model2.wv.similarity('alice', 'wonderland')) 
      
print("Cosine similarity between 'alice' " +
            "and 'machines' - Skip Gram : ", 
      model2.wv.similarity('alice', 'machines')) 

