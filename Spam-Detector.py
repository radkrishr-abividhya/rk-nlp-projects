import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
# from wordcloud import WordCloud
import wordcloud as wc

# data from:https://www.kaggle.com/uciml/sms-spam-collection-dataset
df = pd.read_csv('spam.csv', encoding='ISO-8859-1')

# drop unnecessary columns
df = df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)

# rename columns to something better
df.columns = ['labels', 'data']

# create binary labels
df['b_labels'] = df['labels'].map({'ham': 0, 'spam': 1})
Y = df['b_labels'].values


#https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
count_vectorizer = CountVectorizer(decode_error='ignore')
X = count_vectorizer.fit_transform(df['data'])
print(count_vectorizer.get_feature_names())


# split up the data
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.33)

# create the model, train it, print scores
model = MultinomialNB()
model.fit(Xtrain, Ytrain)
print("train score:", model.score(Xtrain, Ytrain))


# visualize the data
def visualize(label):
  words = ''
  for msg in df[df['labels'] == label]['data']:
    msg = msg.lower()
    words += msg + ' '
  wordcloud = wc.WordCloud(width=600, height=400).generate(words)
  plt.imshow(wordcloud)
  plt.axis('off')
  plt.show()

#visualize('spam')
#visualize('ham')


# see what we're getting wrong
y_pred = model.predict(Xtest)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Ytest, y_pred)
 
# Test with some new data 
comment = "Free entry in 2 a wkly comp to win FA Cup final"
data = [comment]
vect = count_vectorizer.transform(data).toarray()
my_prediction = model.predict(vect)

if(my_prediction == 0):
     print("Ham")
if(my_prediction == 1):
     print("Spam")
     
