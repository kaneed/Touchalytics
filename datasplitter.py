## for data
import pandas as pd
import numpy as np
import random
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

num_users = 41
num_sessions = 7


def splitTrainAndTestData(dtf, experiment_mode):
    if experiment_mode == "short_term":
        return splitByShortTerm(dtf)
    elif experiment_mode == "long_term":
        return splitByLongTerm(dtf)
    else:
        return splitRandomly(dtf)

def splitRandomly(dtf):
    return [model_selection.train_test_split(dtf, test_size=0.3)]

def splitByShortTerm(dtf):
    trainTestGroups = []
    for i in range(1,num_sessions+1):
        mask = dtf["doc_id"] == i
        data = dtf[mask]
        trainTestGroups.append(splitRandomly(data)[0])
    return trainTestGroups

def splitByInterSession(dtf): 
    return splitRandomly(dtf)

# Train on group 1-5 (April 16-20), test groups 6-7 (April 24-27)
def splitByLongTerm(dtf):
    trainTestGroups = []
    laterMask = dtf["doc_id"] >= 6
    earlyMask = dtf["doc_id"] <= 5
    train = dtf[earlyMask]
    test = dtf[laterMask]
    trainTestGroups.append([test, train])
    return trainTestGroups

def splitByHorizontalVertical(dtf):
    hori = dtf[np.absolute(dtf["stop_x"] - dtf["start_x"]) > np.absolute(dtf["stop_y"] - dtf["start_y"])]
    vert = dtf[np.absolute(dtf["stop_x"] - dtf["start_x"]) < np.absolute(dtf["stop_y"] - dtf["start_y"])]
    return hori, vert

def splitByNUsers(dtf, user_id, other_users):
    idx = other_users
    if user_id <= other_users:
        idx += 1
    dtf_n_users = dtf[dtf["user_id"] <= idx]
    return splitRandomly(dtf_n_users)

def splitByUniquePhone(dtf):
    num_phones = 5
    trainTestGroups = []
    for i in range(1,num_phones+1):
        data = dtf[dtf["phone_id"] == i]
        trainTestGroups.append(splitRandomly(data)[0])
    return trainTestGroups

def splitByUniqueInstructor(dtf):
    experimenter_e = dtf[dtf["phone_id"] == 1]
    experimenter_r = dtf[dtf["phone_id"] == 3]
    instructors = [experimenter_e, experimenter_r]

    trainTestGroups = []
    for instructor in instructors:
        trainTestGroups.append(splitRandomly(instructor)[0])
    return trainTestGroups

def splitByInterPhone(dtf):
    return splitRandomly(dtf)

def groups_of_n(data_x, data_y, n):
    groups_x = []
    groups_y = []
    genuine_strokes = data_x[data_y] 
    ingenuine_strokes = data_x[~data_y]
    
    for i in range(0,len(genuine_strokes),n):
        groups_x.append(genuine_strokes[i:i+n])
        groups_y.append(True)

    for i in range(0,len(ingenuine_strokes),n):
        groups_x.append(ingenuine_strokes[i:i+n])
        groups_y.append(False)
    
    return groups_x, groups_y










        
