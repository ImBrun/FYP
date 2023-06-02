import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import os

from extract_features import extract_features

def classify(img, mask):
    x = extract_features(img, mask)
    x = pd.DataFrame([x], columns=['Asymmetry', 'Border_Irregularity', 'IQR', 'S_STD', 'V_STD', 'N_Sus', 'Red', 'Gray-Blue', 'Pink', 'Dark-Brown'])

    pca_reload = pickle.load(open('pca.pkl', 'rb'))
    classifier = pickle.load(open('group1_classifier.sav', 'rb'))

    x = pca_reload.transform(x)
    pred_label = classifier.predict(x)
    pred_prob = classifier.predict_proba(x)
    
    #predicted label at threshold 0.6 for non-cancerous, uncomment and return when you want to see
    #pr_to_label = np.where(pred_prob[:,0] > 0.6, 0, 1)

    return pred_label, pred_prob #, pr_to_label

def test_classify():
    lab = []
    prb = []
    for name in os.listdir('../data/testimg/mask'):
        if name != ".DS_Store":
            print(name)
            mask = plt.imread(os.path.join('../data/testimg/mask', name))
            img = plt.imread(os.path.join('../data/testimg/img', name))
            x, y = classify(img, mask)
            lab.append(x)
            prb.append(y)
    print(lab)
    print(prb)

test_classify()
