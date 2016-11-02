#!/usr/bin/python

import sys
import pandas as  pd
import csv as csv
import numpy
from algorithms import gaussNB,LogReg,RandForest,LinearS, DTree, SVC
from tester import test_classifier

### 1. Build Feature Lists

features_1 = []
features_2=[]
features_3=[]
features_4=[]
features_5=[]

def fill_feature(filename,label=True):

    features = pd.read_csv(filename,header=0)
    my_list = []
    if label==True:
        my_list.append('label')
    for index,row in features.iterrows():
        colname = 'pixel' + str(row['x'])
        my_list.append(colname)

    return my_list

features_1 = fill_feature('digit_cols1under20.csv')
features_2 = fill_feature('digit_cols2under100.csv')
features_3 = fill_feature('digit_cols3under150.csv')
features_4 = fill_feature('digit_cols4under100extras.csv')
features_5 = fill_feature('digit_cols5under150extras.csv')
features_4_test = fill_feature('digit_cols4under100extras.csv',False)

## the data has been cleaned already (http://www.bmdata.co.uk/titanic_code.html) and converted to numeric
## decision has been made to remove 3 features: Name, Ticket and Cabin

## 2. Load cleaned data
digits = pd.read_csv('digit_train.csv',header=0)
digits_test = pd.read_csv('digit_test.csv',header=0)

## 3. Restrict dataframe to selected features (changed list when testing for best results)

digits = digits[features_4]
digits_test = digits_test[features_4_test]

#5. Convert train and test to numpy arrays
train_data = digits.values
test_data = digits_test.values

#set the features and labels for the training data
features = train_data[0::,1::]
labels = train_data[0::,0]


#6. For my testing - cross validation

#from sklearn import cross_validation
#features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.3, random_state=42)


#7.  SelectKBest (my tests commented out)

from sklearn.feature_selection import SelectKBest

#x = 0
#y = 0
klevel = 331


kbest = SelectKBest(f_classif, k=klevel)
kbest.fit_transform(features, labels)

#for f in features_4:
 #  if x > 0:
#        if kbest.scores_[x-1] > 50:
#            print(f,'KBEST',kbest.scores_[x-1])
#            y = y + 1
 #  x = x + 1

#print(y)

#different features lists with KBEST scores and results with untuned SVC algorithm

#Feature 1 - 51 with KBEST over 2000 (acc 0.844, prec 0.847), 10 with KBEST over 2500 (acc 0.61, prec 0.63)
#Feature 2 - 51 with KBEST over 2000 (acc .844, .847), 10 with KBEST over 2500 (acc .61, prec .63)
#Feature 3 - 48 with KBEST over 2000 (acc .0.843, prec .847), 9 with KBEST over 2500 (acc .589, prec .613)
#Feature 4 - 51 with KBEST over 2000 (acc .844, prec .847), 10 with KBEST over 2500 (acc 60.7, prec .628)
#Feature 5 - 48 with KBEST over 2000 (acc .843, prec .847), 9 with KBEST over 2500 (acc .589, prec .613)

#Feature 4 > 1000 KBEST (197) (accuracy: 0.957, precision: 0.957)
#Feature 4 > 500 KBEST (306) (accuracy: 0.965, precision: 0.965)
#Feature 4 > 250 KBEST (330) (accuracy: 0.965, precision: 0.965)
#Feature 4 > 100 or 50 KBEST (331) (accuracy: 0.964, precision: 0.964)


#  8. Scaler

from sklearn.preprocessing import MinMaxScaler

scaler =MinMaxScaler()


#7. Algorithm test - see algoriths functions, comment in and out.

#clf = gaussNB(scaler,kbest)
clf = SVC(scaler,kbest)
#clf = DTree(scaler,kbest)
#clf = LinearS(scaler,kbest)
#clf = LogReg(scaler,kbest)
#clf = RandForest(scaler,kbest)



def my_scores(clf):

    pred = clf.predict(features_test)
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(pred, labels_test)
    from sklearn.metrics import precision_score
    precision = precision_score(pred,labels_test)
    from sklearn.metrics import recall_score
    rec = recall_score(pred, labels_test)
    print ('accuracy:',acc)
    print('precision:',precision)

#code for testing my algorithms

#clf.fit(features_train,labels_train)
#my_scores(clf)

#final code to dump results to csv file for upload onto kaggle

def kaggle_dump(clf, test_data):
    clf.fit(features, labels)
    predictions = clf.predict(test_data).astype(int)
    predictions_file = open("digit_predictions.csv", "w")
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(["ImageId","Label"])
    open_file_object.writerows(zip(range(1,len(test_data)+1), predictions))
    predictions_file.close()
    print ('Done.')

kaggle_dump(clf,test_data)
