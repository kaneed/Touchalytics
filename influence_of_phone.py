import os
## for data
import pandas as pd
import numpy as np
## for plotting
import matplotlib.pyplot as plt
import seaborn as sns
## for statistical tests
import scipy
import statsmodels.formula.api as smf
import statsmodels.api as sm
## for machine learning
from sklearn import model_selection, preprocessing, feature_selection, ensemble, linear_model, metrics, decomposition
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
## for explainer
from lime import lime_tabular
import pickle
import datasplitter
import machine_learning as ml

## GLOBALS ##########################
n = 1 # Number of strokes to analyze
threshold = 0.5 # Placeholder
num_users = 4 # Number of users to analyze (max 41)
#####################################

# Create Folders
folders = ["graphs", "user_models"]
for folder in folders:
    if not os.path.exists(folder):
        os.mkdir(folder)

def test_multiple_strokes(model, n, data_x, data_y):
    global threshold
    cx, cy = datasplitter.groups_of_n(X_test, y_test, n)
    correct = 0
    predicted_prob = []
    predicted = []
    total = len(cx)
    
    for i in range(total):
        pred = ml.predict_n(model, cx[i], threshold)
        pred_proba = ml.predict_n_proba(model, cx[i])
        real = cy[i]
        if pred == real:
            correct += 1
        predicted.append(pred)
        predicted_prob.append(pred_proba)
            
    accuracy = correct / total
    print("Accuracy: " + '{0:.2f}'.format(accuracy) + " (" + str(correct) + "/" + str(total) + ") [THR=" + '{0:.2f}'.format(threshold) + "]")
    return np.asarray(cy), np.asarray(predicted), np.asarray(predicted_prob)


## Read in data
dtfInit = pd.read_csv('featMat.csv')
dtfInit.head()
dtfInit = dtfInit.replace(-np.inf, np.nan)
dtfInit = dtfInit.replace(np.inf, np.nan)
dtfInit["strokeId"] = range(len(dtfInit))
dtfInit = dtfInit.set_index("strokeId") # Unique identifier

# Tracked Information
incorrect_unique_phones = []
incorrect_unique_instructors = []
incorrect_inter_phones = []

for user in range(1, num_users+1):
    print("\nTraining subgroups of user " + str(user) + " (n=" + str(n) + ")")
    
    dtf = dtfInit.copy()
    dtf["Y"] = dtf["user_id"].apply(lambda u: u == user)

    # Split data into test / train.
    experiment_modes = ["unique_phone", "unique_instructor", "inter_phone"]
    for experiment in experiment_modes:
        print("Switching to experiment mode: " + experiment)
        if experiment == "unique_phone":
            dtf_TrainTestGroups = datasplitter.splitByUniquePhone(dtf)
        elif experiment == "unique_instructor":
            dtf_TrainTestGroups = datasplitter.splitByUniqueInstructor(dtf)
        elif experiment == "inter_phone":
            dtf_TrainTestGroups = datasplitter.splitByInterPhone(dtf)
            
        for group in dtf_TrainTestGroups:
            dtf_train = group[0]
            dtf_test = group[1]
            X_train, y_train, X_test, y_test = ml.preprocess(dtf_train, dtf_test)

            # Train and Test
            try:
                model = SVC(kernel='rbf', random_state=1, probability=True)
                model.fit(X_train, y_train)
                
                single_predicted_prob = model.predict_proba(X_test)[:,1]
                fpr, tpr, thresholds = metrics.roc_curve(y_test, single_predicted_prob)
                threshold, _ = ml.optimalThreshold(fpr, tpr, thresholds)
                
                y_test, predicted, predicted_prob = test_multiple_strokes(model, n, X_test, y_test)
                accuracy = metrics.accuracy_score(y_test, predicted)

                if experiment == "unique_phone":
                    incorrect_unique_phones.append(100*(1-accuracy))
                elif experiment == "unique_instructor":
                    incorrect_unique_instructors.append(100*(1-accuracy))
                elif experiment == "inter_phone":
                    incorrect_inter_phones.append(100*(1-accuracy))

            except:
                pass

dataset = [incorrect_inter_phones, incorrect_unique_phones]
labels = ["Inter-Phones", "Unique Phones"]
bp = plt.boxplot(dataset,labels=labels, vert=False)
plt.xlabel("fraction of wrongly classified strokes (%)")
plt.imsave("graphs/figure8_influence-of-phones2.png")
plt.show()









