## for data
import pandas as pd
import numpy as np
## for plotting
import matplotlib.pyplot as plt
import seaborn as sns
## for statistical tests
import scipy
from scipy.optimize import brentq
from scipy.interpolate import interp1d
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

def preprocess(dtf_train, dtf_test):
    # Fill in missing data
    colsMissingData = ["inter_stroke_time", "20p_pairwise_acc", "50p_pairwise_acc", "80p_pairwise_acc", "med_velocity_last3", "ratio_e2e_dist_traj_length", "median_acc_first5"]
    for col in colsMissingData:
        try:
            dtf_train[col] = dtf_train[col].fillna(dtf.loc(col).mean())
            dtf_test[col] = dtf_test[col].fillna(dtf.loc(col).mean())
        except:
            pass
    
    # Remove un-used features
    badCols = ["user_id", "doc_id", "phone_id"]
    dtf_train = dtf_train.drop(badCols, axis=1)
    
    # Drop NaN rows
    dtf_train = dtf_train.replace([-np.inf, np.inf], np.nan)
    dtf_train = dtf_train.dropna()
    dtf_test = dtf_test.replace([-np.inf, np.inf], np.nan)
    dtf_test = dtf_test.dropna()

    # Scale to normalize features:
    scaler = preprocessing.StandardScaler()
    # Scale training data
    X = scaler.fit_transform(dtf_train.drop("Y", axis=1))
    dtf_scaled= pd.DataFrame(X, columns=dtf_train.drop("Y", axis=1).columns, index=dtf_train.index)
    dtf_scaled["Y"] = dtf_train["Y"]
    dtf_scaled.head()
    dtf_train = dtf_scaled
    # Scale testing data
    X = scaler.fit_transform(dtf_test.drop("Y", axis=1))
    dtf_scaled= pd.DataFrame(X, columns=dtf_test.drop("Y", axis=1).columns, index=dtf_test.index)
    dtf_scaled["Y"] = dtf_test["Y"]
    dtf_scaled.head()
    dtf_test = dtf_scaled

    # Choose the desired features
    X_names = ['inter_stroke_time', 'stroke_duration', 'start_x', 'start_y', 'stop_x', 'stop_y', 'direct_e2e_dist', 'mean_resultant_length', 'direction_enum', 'dir_e2e_line', '20p_pairwise_velocity', '50p_pairwise_velocity', '80p_pairwise_velocity', '20p_pairwise_acc', '50p_pairwise_acc', '80p_pairwise_acc', 'med_velocity_last3', 'lgst_dev_e2e_line', '20p_dev_e2e_line', '50p_dev_e2e_line', '80p_dev_e2e_line', 'average_direction', 'trajectory_length', 'ratio_e2e_dist_traj_length', 'average_velocity', 'median_acc_first5', 'midstroke_pressure', 'midstroke_area_covered', 'midstroke_finger_orientation', 'finger_orientation_changed', 'phone_orientation']
    X_train = dtf_train[X_names].values
    y_train = dtf_train["Y"].values
    X_test = dtf_test[X_names].values
    y_test = dtf_test["Y"].values
    return X_train, y_train, X_test, y_test



def optimalThreshold(data_y, predicted_proba):
    precision, recall, thresholds = metrics.precision_recall_curve(data_y, predicted_proba)
    
    # Harmonic Means
    '''
    harmonic_means = (2 * precision * recall) / (precision + recall)
    index = np.nanargmax(harmonic_means)
    '''
    
    # Dist
    '''
    dy_squared = (1-precision)**2
    dx_squared = (1-recall)**2
    dist = np.sqrt(dy_squared + dx_squared)
    index = np.nanargmin(dist)
    '''
    
    # Equal Precision vs Recall
    index = np.nanargmin(np.absolute((precision - recall)))
    
    threshold = thresholds[index]
    eer = 1 - len(data_y[data_y == (predicted_proba > threshold)]) / len(data_y)
    return threshold, eer

def predict_n(model, strokes, threshold):
    avg_prob = predict_n_proba(model, strokes)
    return avg_prob > threshold

def predict_n_proba(model, strokes):
    s = 0
    probs = model.predict_proba(strokes)
    for prob in probs:
        s += prob[1]
    avg_prob = s / len(probs)
    return avg_prob

def plot_roc(y_test, predicted, predicted_prob):
    accuracy = round(metrics.accuracy_score(y_test, predicted), 2)
    auc = round(metrics.roc_auc_score(y_test, predicted_prob), 2)
    fpr, tpr, _ = metrics.roc_curve(y_test, predicted_prob)
    a = np.linspace(0, 1, len(fpr))
    b = a[::-1]
    plt.figure()
    plt.plot(fpr,tpr)
    plt.plot(a,b)
    plt.title("AUC=" + str(auc) + ", ACC=" + str(accuracy))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def plot_prc(data_y, predicted, predicted_prob):
    precision, recall, thresholds = metrics.precision_recall_curve(data_y, predicted_prob)
    fig, ax = plt.subplots()
    ax.plot(recall, precision)
    #add axis labels to plot
    ax.set_title('Precision-Recall Curve')
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')
    ax.set_xlim([-0.1,1.1])
    ax.set_ylim([-0.1,1.1])
    
    #display plot
    plt.show()
