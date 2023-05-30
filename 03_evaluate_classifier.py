import pickle
import matplotlib as plt

from extract_features import extract_features

def classify(img, mask):
    x = extract_features(img, mask)

    classifier = pickle.load(open('group1_classifier.sav', 'rb'))

    pred_label = classifier.predict(x)
    pred_prob = classifier.predict_proba(x)

    return pred_label, pred_prob

