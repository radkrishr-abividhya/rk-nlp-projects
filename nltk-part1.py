# Stemming

import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
print(ps.stem("runs"))
print(ps.stem("running"))
print(ps.stem("likes"))
print(ps.stem("wolves"))

# Lemmatization
# https://www.geeksforgeeks.org/python-lemmatization-with-nltk/

# To be executed only once
# nltk.download('wordnet')
 
from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer() 
print(lemmatizer.lemmatize("wolves"))
print(lemmatizer.lemmatize("worst", pos="a"))
print(lemmatizer.lemmatize("better", pos="a"))


# POS Tagging
#https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
# To be executed only once
# nltk.download('averaged_perceptron_tagger')
nltk.pos_tag("Machine Learning is great".split())

# To be executed only once
# nltk.download('maxent_ne_chunker')
# nltk.download('words')

# Name Entity Recognition
#sentence = "Steve jobs was the CEO of Apple Corp"
#tags = nltk.pos_tag(sentence.split())
#nltk.ne_chunk(tags).draw()