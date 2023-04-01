# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 11:14:10 2020

@author: Day
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 11:18:26 2019

@author: Day
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
data = pd.read_csv("D:/data/testdatav11.csv",header=0)
data = data.dropna()
print(data.shape)
print(list(data.columns))



#--------------------หากตัวแปรไหนมีกลุ่มมากให้รวมกลุ่ม--------------------------------

#data['AGE'] = pd.to_numeric(data['AGE']) # แปลงเป็น Numeric
#data['SQ'] = pd.to_numeric(data['SQ'])


#data['SQ']=np.where(data['SQ'] >=23, '1', data['SQ'])

#-------------------------------เช็คเปอร์เซนต์ของผลทำนาย -------------------
count_no_sub = len(data[data['Type']==0])
count_sub = len(data[data['Type']==1])
pct_of_no_sub = count_no_sub/(count_no_sub+count_sub)
print("percentage of no subscription is", pct_of_no_sub*100)
pct_of_sub = count_sub/(count_no_sub+count_sub)
print("percentage of subscription", pct_of_sub*100)

data.groupby('Type').mean()



#-------------ข้อมูลที่รวมตัวแปรหุ่นแล้ว และจะใช้ในการประมวลผล------------------------------------
data_final=data
data_final.columns.values

#-----------------------------ทำ SMOTE ทำให้ผลเฉลยมีค่าใกล้เคียงกัน-----ไม่ได้ใช้เพราะเท่ากันอยู่แล้ว----------------------------------
X = data_final.loc[:, data_final.columns != 'Type']
y = data_final.loc[:, data_final.columns == 'Type']



#-------------------------ตัวแปรที่ดี--------------------------------
cols=['Nor_AGE','Nor_SQ','Gender'] 
X=X[cols]
y=y['Type']

#----------------------Implementing the model-------------------------
import statsmodels.api as sm
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())
 
#-------------------------ตัดตัวแปรที่ P>0.05 ออก--------------------------

#---------------------Logistic Regression Model Fitting--------------------------seed
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

#-Predicting the test set results and calculating the accuracy ทำนายผลการทดสอบ-------
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

#---------------Confusion Matrix-----------------------------------------------
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

 

#-----------ROC Curve---------------------------------------

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

#------Reference: Learning Predictive Analytics with Python book------------
