#----------------imports-----------------

import pandas as pd
import numpy as np
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score, f1_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA

import pickle

#-----------------------------------------

#---------------Reading the csv and preprocessing----------------


def process_csv(features_path, drop_list):
    
    #Read in data from the features csv and drop NA values if they exist
    data = pd.read_csv(features_path)
    data = data.dropna() 

    #Remove chosen columns
    data = data.drop(drop_list, axis=1)
    #Keep track of how many columns we have depending on removed columns
    n = 20 - len(drop_list)

    #Standardizing features
    scaler = StandardScaler()
    #n-2 because we don't standardize the labels "diagnostic" and "is_cancer"
    for i in range(1,n-1):
        data[data.columns[i]] = scaler.fit_transform(data[data.columns[i]].values.reshape(-1, 1))

    data[data.columns[n]]=data[data.columns[n]].astype(int)
    return n, data
    

#-----------------------------------------------------------------

#---------------Create training data-------------------

def create_training_data(data, n):
    
    # Define feature space
    X = data[data.columns[1:(n-1)]] 
    
    
    # Define labels
    y = data[data.columns[n]]
    
    # Convert X and y to numpy arrays
    X,y = X.to_numpy(), y.to_numpy()
    y = np.ravel(y)
    
    # Create test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.20, random_state=6)
    
    #Kfold
    #creating 5 train/validation sets
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2)
    
    return X_train, X_test, y_train, y_test, sss

#-------------------------------------------------------

#---------------Create KNN model and test its performance---------------

def test_KNN(X_train, y_train, sss, neighbours, THRESHOLD, weights= "uniform", p = 2, metric = "minkowski"):
    
    #Some basic code for testing knn
    clf = KNeighborsClassifier(n_neighbors = neighbours,p=p,metric=metric, weights=weights )
    
    accuracies = []
    f1_s = []
    recalls = []
    
    for i, (train_index, val_index) in enumerate(sss.split(X_train, y_train)):

        #fitting the training data to the classifier & making predictions
        clf.fit(X_train[train_index], y_train[train_index])
        predictions = clf.predict(X_train[val_index])

        #Code to use KNN with probabilities
        pr = clf.predict_proba(X_train[val_index])
        proba = np.where(pr[:,0] > THRESHOLD, 0, 1)

        #Display accuracy score aka the percentage of correctly predicted labels
        acc = (accuracy_score(y_train[val_index], proba))
        rec = (recall_score(y_train[val_index], proba))
        f_1 = (f1_score(y_train[val_index], proba))
        accuracies.append(acc)
        recalls.append(rec)
        f1_s.append(f_1)
        print(f"-------------------ACCURACY SCORE: {acc}")
        
        
        
        

        #Display Confusion matrices
        cm = confusion_matrix(y_train[val_index], proba, labels=clf.classes_)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=clf.classes_)
        disp.plot()

        print(classification_report(y_train[val_index], predictions))
        
        
        
    avg_acc = sum(accuracies)/len(accuracies)
    print(f"---------AVERAGE ACCURACY: {avg_acc}")
    
    avg_rec = sum(recalls)/len(recalls)
    print(f"---------AVERAGE RECALL: {avg_rec}")
    
    avg_f_1 = sum(f1_s)/len(f1_s)
    print(f"---------AVERAGE F1 SCORE: {avg_f_1}")
    
#-----------------------------------------------------------------------

#----------------Find the best parameters for KNN-------------------

def find_optimal_KNN(X_train, y_train):

    clf = KNeighborsClassifier()

    param_grid = {'n_neighbors': range(1,22,2), 'weights': ['uniform', 'distance'], 'p': [1,2], 'metric': ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan', 'nan_euclidean']}
    grid_search = GridSearchCV(clf, param_grid, cv=5, scoring="f1_weighted")
    grid_search.fit(X_train, y_train)

    k_val = grid_search.best_params_['n_neighbors']
    weight = grid_search.best_params_['weights']
    p_val = grid_search.best_params_['p']
    metric = grid_search.best_params_['metric']

    return k_val, weight, p_val, metric


#--------------------------------------------------------------------

#------------------PCA on DataFrame-------------------

def PCAdf():
    data_colors = data[['IQR', 'H_STD', 'S_STD', 'V_STD',
        'N_Pred_Colors', 'Black', 'Green', 'Blue',
        'Pink', 'Purple', 'Light-Brown']]
    data_stats = data[['Asymmetry', 'Border_Irregularity']]
    is_cancer = data[['Is_Cancer']]

    # Perform PCA on the features
    pca = PCA(n_components=1)
    pca_component = pca.fit_transform(data_colors)

    # Convert the PCA component to a DataFrame
    pca_df = pd.DataFrame(pca_component, columns=['PCA_Component'])

    # Append the PCA component to the original DataFrame
    df_with_pca = pd.concat([data_stats, pca_df, is_cancer], axis=1)
    return df_with_pca

#-----------------------------------------------------


features_path = "features/features.csv"
drop_list = []

n, data = process_csv(features_path, drop_list)
X = data[data.columns[1:(n-1)]]
y = data[data.columns[n]]
X,y = X.to_numpy(), y.to_numpy()
y = np.ravel(y)

X_train, X_test, y_train, y_test, sss = create_training_data(data, n)

THRESHOLD = 0.5
neigh, weight, p_val, metric  = find_optimal_KNN(X_train, y_train)
#test_KNN(X_train, y_train, sss, neigh, THRESHOLD, p=p_val,metric=metric, weights=weight)

#Apply PCA on the original dataframe
pcaDF = PCAdf()

classifier = KNeighborsClassifier(n_neighbors=neigh, weights=weight, p=p_val, metric=metric)
classifier.fit(X, y)

filename = 'group1_classifier.sav'
pickle.dump(classifier, open(filename, 'wb'))


