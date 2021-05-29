#Similar sounding words using gensim Word2Vec

import gzip
import gensim

data_file="reviews_data.txt.gz"

with gzip.open (data_file, 'rb') as f:
    for i,line in enumerate (f):
        print(line,"\n\n")
        break

def read_input(input_file):
    with gzip.open (input_file, 'rb') as f:
        for i, line in enumerate (f): 
           # do some pre-processing and return a list of words for each review text
            yield gensim.utils.simple_preprocess (line)

# read the tokenized reviews into a list
# each review item becomes a serries of words
# so this becomes a list of lists

documents = list (read_input (data_file))
print(documents[0])

model = gensim.models.Word2Vec (documents, vector_size=150, window=10, min_count=2, workers=10)
model.train(documents,total_examples=len(documents),epochs=10)
print("\n\ndone\n\n")

w1 = "happy"
print(model.wv.most_similar (positive=w1))

w1 = "dirty"
print(model.wv.most_similar(positive=w1))

