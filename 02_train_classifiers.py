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

    #save the labels
    y = data["Is_Cancer"]

    #save the feature space
    X = data.copy()
    X = X.drop(["ID", "Diagnostic", "Is_Cancer"], axis=1)
    
  
    #Standardizing features
    scaler = StandardScaler()
 
    for i in range(0, len(X.columns)):
        X[X.columns[i]] = scaler.fit_transform(X[X.columns[i]].values.reshape(-1, 1))
    
    pca = PCA(n_components=5)
    pca_component = pca.fit_transform(X)
    pickle.dump(pca, open('pca.pkl', 'wb'))

    # Convert the PCA component to a DataFrame
    pca_df = pd.DataFrame(pca_component)

    y= y.astype(int)
    
    return pca_df,y
    

#-----------------------------------------------------------------
#---------------Create training data-------------------

def create_training_data(X, y):
    
    # Convert X and y to numpy arrays
    X,y = X.to_numpy(), y.to_numpy()
    y = np.ravel(y)
    
    # Create test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.20)
    X_train.shape,y_train.shape, X_test.shape,y_test.shape
    
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


features_path = "features/features.csv"
drop_list =  ["Green", "Blue", "Black", "Light-Brown", "Purple", "N_Pred_Colors","White", "H_STD"]

X,y = process_csv(features_path, drop_list)

X_train, X_test, y_train, y_test, sss = create_training_data(X,y)

test_KNN(X_train,y_train,sss,5,0.6)

final_classifier = KNeighborsClassifier(n_neighbors=5)
final_classifier = final_classifier.fit(X,y)


filename = "group1_classifier.sav"
pickle.dump(final_classifier, open(filename, "wb"))

