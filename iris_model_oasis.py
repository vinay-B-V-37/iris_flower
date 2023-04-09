#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt


# In[2]:


import seaborn as sns
sns.set(style="white", color_codes=True)


# In[3]:


iris = pd.read_csv("Iris.csv")


# In[4]:


iris.head()


# In[5]:


iris.tail()


# In[6]:


iris


# In[7]:


iris.describe()


# In[8]:


iris.plot(kind="scatter", x="SepalLengthCm", y="SepalWidthCm")


# In[9]:


sns.jointplot(x="SepalLengthCm", y="SepalWidthCm", data=iris, size=8)


# In[10]:


sns.boxplot(x="Species", y="PetalLengthCm", data=iris)


# In[11]:


ax = sns.boxplot(x="Species", y="PetalLengthCm", data=iris)


# In[12]:


ax = sns.stripplot(x="Species", y="PetalLengthCm", data=iris, jitter=True, edgecolor="blue")


# In[13]:


sns.violinplot(x="Species", y="PetalLengthCm", data=iris, size=5)


# In[14]:


sns.pairplot(iris.drop("Id", axis=1), hue="Species", size=6)


# In[15]:


iris.drop("Id", axis=1).boxplot(by="Species", figsize=(12, 6))


# In[16]:


from pandas.plotting import andrews_curves
andrews_curves(iris.drop("Id", axis=1), "Species")


# In[17]:


from pandas.plotting import parallel_coordinates
parallel_coordinates(iris.drop("Id", axis=1), "Species")


# In[18]:


from pandas.plotting import radviz
radviz(iris.drop("Id", axis=1), "Species")


# In[19]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


# In[20]:


X = iris.iloc[:, :-1].values
y = iris.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[21]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)


# In[22]:


y_pred = classifier.predict(X_test)


# In[23]:


print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(y_pred,y_test))


# In[24]:


from sklearn.neighbors import KNeighborsClassifier


# In[25]:


classifier = KNeighborsClassifier(n_neighbors=8)
classifier.fit(X_train, y_train)


# In[26]:


y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(y_pred,y_test))


# In[27]:


from sklearn.svm import SVC


# In[28]:


classifier = SVC()
classifier.fit(X_train, y_train)


# In[29]:


y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(y_pred,y_test))


# In[30]:


from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)


# In[31]:


y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(y_pred,y_test))


# In[32]:


from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train, y_train)


# In[33]:


y_pred = classifier.predict(X_test)


# In[34]:


print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(y_pred,y_test))


# In[35]:


from sklearn.naive_bayes import BernoulliNB
classifier = BernoulliNB()
classifier.fit(X_train, y_train)



# In[36]:


y_pred = classifier.predict(X_test)


# In[37]:


print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


# In[38]:


from sklearn.naive_bayes import ComplementNB
classifier = ComplementNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


# In[39]:


from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(y_pred,y_test))


# In[40]:


from sklearn.metrics import accuracy_score, log_loss
classifiers = [ MultinomialNB(),BernoulliNB(), ComplementNB(),
              ]              
                  


# In[41]:


log_cols=["Classifier", "Accuracy", "Log Loss"]
log = pd.DataFrame(columns=log_cols)
 


# In[42]:


for clf in classifiers:
    clf.fit(X_train, y_train)
    name = clf.__class__.__name__
    
    print("="*30)
    print(name)
    
    print('*Results*')
    train_predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, train_predictions)
    print("Accuracy: {:.4%}".format(acc))
    
    log_entry = pd.DataFrame([[name, acc*100, 11]], columns=log_cols)
    log = log.append(log_entry)
    
    print("="*30)


# In[43]:


sns.set_color_codes("muted")
sns.barplot(x='Classifier', y='Accuracy',  data=log, color="green")


# In[46]:


plt.xlabel('Classifier')
plt.ylabel('Accuracy')
plt.title('Classifier Accuracy')
plt.show()


# In[ ]:




