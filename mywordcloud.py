import pandas as pd
import matplotlib.pyplot as plt
# from wordcloud import WordCloud
import wordcloud as wc 
# import wordcloud


# data from:https://www.kaggle.com/uciml/sms-spam-collection-dataset
df = pd.read_csv('spam.csv', encoding='ISO-8859-1')

# drop unnecessary columns
df = df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)

# rename columns to something better
df.columns = ['labels', 'data']

# visualize the data
def visualize(label):
  words = ''
  for msg in df[df['labels'] == label]['data']:
    msg = msg.lower()
    words += msg + ' '
  word_cloud = wc.WordCloud(width=600, height=400).generate(words)
  plt.imshow(word_cloud)
  plt.axis('off')
  plt.show()

visualize('spam')
