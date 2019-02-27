# -*- coding: utf-8 -*-
"""
Project: Implementing an efficient classification algorithm for car evaluation
@author: Venkata Rahul Jammalamadaka

Programming language: Python, version: 3.6
IDE: Anaconda (Spyder)
"""


import datetime
import scipy
import numpy
import pandas as pd
from sklearn import cross_validation

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
#import numpy as np
from sklearn.model_selection import train_test_split


if __name__ == "__main__":   
    start = datetime.datetime.now()
    
    
    

#Loading the dataset
    dataset = pd.read_csv('car.csv')
    X = dataset.iloc[:,[1,6]].values
    y = dataset.iloc[:,7].values
    tryy=1
    
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size = 0.25, random_state = 0)
    while tryy:
        k1=0
        choice = input("\nChoose classifier below\n1. Random Forest Classifier\n2. SVM_SVC\n3. Decision Tree\n4. Gausian Naive Bayes\n5. KNeighborsClassifier\n: ").strip()
        if choice not in ['1', '2', '3', '4', '5']:
            print ("You have entered wrong input. Please give a valid input")
            
            k1=1
        #Classifier Implementations    
        elif choice == "1":
            print ("\nInput received as "+choice+"...processing inputs now...Evaluating K-fold accuracy on training data and calculating accuracy on test data.\n")
        
            clf = RandomForestClassifier(n_estimators=90).fit(Xtrain, Ytrain)
            accT = clf.score(Xtest, Ytest)
            print ("Test accuracy score by Random Forest Classifier is", accT)
            cv = cross_validation.ShuffleSplit(len(Xtrain), n_iter=1, test_size = 0.1, random_state=0)
            accL = list(cross_validation.cross_val_score(clf, Xtrain, Ytrain, cv=cv))
            acc = float(sum(accL))/len(accL)
            print ("The K-fold accuracy score on training data is", acc)
            print("Total execution time:", (datetime.datetime.now() - start).total_seconds(), "seconds")
        elif choice == "2":
            print ("\nInput received as "+choice+"...processing inputs now...Evaluating K-fold accuracy on training data and calculating accuracy on test data.\n")
        
            clf = SVC(kernel="linear", C=1).fit(Xtrain, Ytrain)
            accT = clf.score(Xtest, Ytest)
            print ("Test accuracy score by SVM_SVC is", accT)
            cv = cross_validation.ShuffleSplit(len(Xtrain), n_iter=1, test_size = 0.1, random_state=0)
            accL = list(cross_validation.cross_val_score(clf, Xtrain, Ytrain, cv=cv))
            acc = float(sum(accL))/len(accL)
            print ("The K-fold accuracy score on training data is", acc)
            print("Total execution time:", (datetime.datetime.now() - start).total_seconds(), "seconds")
        elif choice == "3":
            print ("\nInput received as "+choice+"...processing inputs now...Evaluating K-fold accuracy on training data and calculating accuracy on test data.\n")
        
            clf = tree.DecisionTreeClassifier(criterion='gini').fit(Xtrain, Ytrain)
            accT = clf.score(Xtest, Ytest)
            print ("Test accuracy score by decision tree is", accT)
            cv = cross_validation.ShuffleSplit(len(Xtrain), n_iter=1, test_size = 0.1, random_state=0)
            accL = list(cross_validation.cross_val_score(clf, Xtrain, Ytrain, cv=cv))
            acc = float(sum(accL))/len(accL)
            print ("The K-fold accuracy score on training data is", acc)
            print("Total execution time:", (datetime.datetime.now() - start).total_seconds(), "seconds")
        elif choice == "4":
            print ("\nInput received as "+choice+"...processing inputs now...Evaluating K-fold accuracy on training data and calculating accuracy on test data.\n")
        
            clf = GaussianNB().fit(Xtrain, Ytrain)
            accT = clf.score(Xtest, Ytest)
            print ("Test accuracy score by Gaussian Naive Bayes is", accT)
            cv = cross_validation.ShuffleSplit(len(Xtrain), n_iter=1, test_size = 0.1, random_state=0)
            accL = list(cross_validation.cross_val_score(clf, Xtrain, Ytrain, cv=cv))
            acc = float(sum(accL))/len(accL)
            print ("The K-fold accuracy score on training data is", acc)
            print("Total execution time:", (datetime.datetime.now() - start).total_seconds(), "seconds")
        elif choice == "5":
            print ("\nInput received as "+choice+"...processing inputs now...Evaluating K-fold accuracy on training data and calculating accuracy on test data.\n")
        
            clf = KNeighborsClassifier(n_neighbors=6).fit(Xtrain, Ytrain)
            accT = clf.score(Xtest, Ytest)
            print ("Test accuracy score by KNeighborsClassifier is", accT)
            cv = cross_validation.ShuffleSplit(len(Xtrain), n_iter=1, test_size = 0.1, random_state=0)
            accL = list(cross_validation.cross_val_score(clf, Xtrain, Ytrain, cv=cv))
            acc = float(sum(accL))/len(accL)
            print ("The K-fold accuracy score on training data is", acc)
            print("Total execution time:", (datetime.datetime.now() - start).total_seconds(), "seconds")
        if k1==0:
            ty = input("Do you want to try with another classifier (y/n):") 
            if choice not in ['y', 'n']:
                print ("You have entered wrong input. Process is terminating.")
            if ty != "y":
                tryy=0
            
        
