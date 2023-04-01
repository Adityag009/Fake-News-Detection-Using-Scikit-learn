#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Your API key is: 1ea8e246889c4d80bf58c2bc46f7a698


# In[ ]:





# In[ ]:





# In[2]:


#import Modules
from newsapi import NewsApiClient
import random


# In[3]:


from datetime import datetime, timedelta
prev_date = datetime.today() - timedelta(days=15)
next_date = datetime.today() - timedelta(days=0)
p_date = str(prev_date.year)+'-'+'0'+str(prev_date.month)+'-'+str(prev_date.day)
c_date = str(next_date.year)+'-'+'0'+str(next_date.month)+'-'+'0'+str(next_date.day)

# Task 2: Create a Get News Method
newsapi = NewsApiClient(api_key='61d7b1d842414eb1a94ff375b20d24ff')
def getNews(sourceId):
    newses = newsapi.get_everything(sources=sourceId,
                                    domains='bbc.co.uk,techcrunch.com',
                                    from_param=p_date,
                                    to=c_date,
                                    language='en',
                                    sort_by='relevancy',
                                    page=3)
    newsData = []
    for news in newses['articles']:
        list = [random.randint(0, 1000), news['title'],news['content'], 'REAL']
        newsData.append(list)
    return newsData


# In[4]:


#Get News Sources
sources = newsapi.get_sources()
sourceList = []
for source in sources['sources']:
    sourceList.append(source['id'])
del sourceList[10:len(sourceList)]
print('New Sources: ',sourceList)


# In[5]:


# Get News using Multiple Sources
dataList = []
for sourceId in sourceList:
    newses = getNews(sourceId)
    dataList = dataList + newses

print('Total News: ', len(dataList))


# In[6]:


import pandas as pd 
df = pd.DataFrame.from_records(dataList)
df.columns = ['','title','text','label']
df.head()


# In[7]:


# Load and concat the DataFrame
trainData = pd.read_csv(r'C:\Users\ADITYA PC\OneDrive\Desktop\Data\nimesh sir\news.csv')
trainData.columns= ['','title','text','label']
data =[trainData,df]
df = pd.concat(data)
df.head()


# In[8]:


#import sckit modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,recall_score,classification_report,confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier


# In[9]:


#split the training 
x_train,x_test,y_train,y_test = train_test_split(df['text'],df.label,test_size=0.3,random_state=100)


# In[10]:


#featur selection
count_vectorize = CountVectorizer(stop_words='english',max_df=0.7)
feature_train = count_vectorize.fit_transform(x_train)
feature_test =count_vectorize.transform(x_test)


# In[11]:


feature_train 


# In[12]:


print(feature_train)


# In[13]:


print(feature_test)


# In[14]:


#Initialise and Apply the classifier 
classifier = PassiveAggressiveClassifier(max_iter=50)
classifier.fit(feature_train,y_train)


# In[15]:


# Test the classifier 
prediction = classifier.predict(feature_test)
score = accuracy_score(y_test,prediction)
print('Accuracy:',score*100)


# In[25]:


test_data = pd.read_csv(r"C:\Users\ADITYA PC\OneDrive\Desktop\Data\nimesh sir\test_data1")
test_labels =test_data.label
test_data.head()


# In[31]:


# Task 13: Select Features and Get Predictions
test_data_feature = count_vectorize.transform(test_data['text'])
prediction = classifier.predict(test_data_feature)


# In[32]:


for i in range(len(test_labels)):
    print(test_labels[i],prediction[i])
score = accuracy_score(test_labels,prediction)
print("Accuracy:",score*100,"%")


# In[ ]:




