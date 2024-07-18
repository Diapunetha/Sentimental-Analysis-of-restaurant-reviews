import pandas as pd
import numpy as np

data=pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3)

data.shape

data.columns

data.head()

data.tail()

data.info

import nltk
import re
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus=[]
for i in range(0,1000):
  review=re.sub(pattern='[^a-zA-Z]',repl=' ',string=data['Review'][i]) #replacing special character with space
  review=review.lower() #converting to lower case
  review_words=review.split()
  ps=PorterStemmer()
  review=[ps.stem(word) for word in review_words]
  review=' '.join(review)
  corpus.append(review)

corpus[:1500]

from sklearn.feature_extraction.text import CountVectorizer   #CountVectorizer convets text to numerical data
cv=CountVectorizer(max_features=1500)
X=cv.fit_transform(corpus).toarray()
y=data.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)

X_train.shape,X_test.shape,y_train.shape,y_test.shape

from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(random_state=0)
classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)
y_pred

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

score1=accuracy_score(y_test,y_pred)
score2=precision_score(y_test,y_pred)

print("Scores")
print("Accuracy Score is {}%".format(round(score1*100,2)))
print("Precision Score is {}%".format(round(score2*100,2)))

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

cm

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

plt.figure(figsize=(10,6))
sns.heatmap(cm, annot=True, cmap='YlGnBu', xticklabels=['Negative','Positive'], yticklabels=['Negative','Positive'])
plt.xlabel("Predicted Values")
plt.ylabel("Actual Label")

best_accuracy=0.0
alpha_val=0.0
for i in np.arange(0.1,1.1,0.1):
    temp_classifier=RandomForestClassifier(random_state=0)
    temp_classifier.fit(X_train,y_train)
    temp_y_pred= temp_classifier.predict(X_test)
    score=accuracy_score(y_test,temp_y_pred)
    print("Accuracy score for alpha={} is: {}%".format(round(i,1),round(score*100,2)))
    if score>best_accuracy:
        best_accuracy=score
        alpha_val=i
print("-------------------------------------------")
print("The best accuracy score is {}% for alpha value as {}".format(round(best_accuracy*100,2),round(alpha_val,1)))

classifier=RandomForestClassifier(random_state=0)
classifier.fit(X_train,y_train)

def predict_sentiment(sample_review):
    sample_review=re.sub(pattern='[^a-zA-Z]',repl=' ',string=sample_review)
    sample_review=sample_review.lower()
    sample_review_words=sample_review.split()
    ps=PorterStemmer()
    final_review=[ps.stem(word) for word in sample_review_words]
    final_review=' '.join(final_review)
    
    temp=cv.transform([final_review]).toarray()
    return classifier.predict(temp)

samplereview=str(input())
if(predict_sentiment(samplereview)):
    print("This is a positive review")
else:
    print("This is a negative review")
