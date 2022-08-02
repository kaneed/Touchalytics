# Touchalytics

This repository consists of several files used to analyze touch information of smart devices, based off of the research of Dr. Mario Frank (http://www.mariofrank.net/paper/touchalytics.pdf).

Note that each experiment is performed for all 41 users, and the results are averaged. For small scale testing, the number of users can be reduced at the top of each experiment file in the section titled "GLOBAL", along with some other relevant variables.

There are a few main files that the rest are structured around:
* machine_learning.py
  - This contains several helper functions which use the sklearn library. The most important are as follow:
    - optimalThreshold() - Calculate the optimal threshold based on the equal error rate, as shown (https://stackoverflow.com/questions/28339746/equal-error-rate-in-python) The fpr vs tpr line is interpolated, then intersected with the diagonal EER line from (0,1) to (1,0). The function returns the EER and theshold at that point.
    - predict_n_proba() - Calculate the probability that a group of N strokes is from a genuine user. This is done by averaging the probability that each stroke is genuine.
    - predict_n() - Predict if a group of N strokes is from a genuine user. This returns the thresholded average (True / False) calculated in predict_n_proba()
    
* datasplitter.py
  - This contains functions used to split data into sensible groups for each experiment setting. Each group contains a dataframe for training and a dataframe for testing. Depending on the setting, there may be multiple groups necessary. The main 3 settings are as follows:
    - splitByShortTerm() - Splits the data into several groups by session. This ensures each session's testing and training data is constrained within the same session.
    - splitByInterSession() - Formats the data into a single group, where the training data and testing data span across multiple sessions.
    - splitByLongTerm() - Formats the data into a single group, where the training data consists of early sessions 1-5 and the testing data consists of later sessions 6-7. This ensures that the strokes are tested against the model from long ago (several days prior).
  - Other important functions include:
    - splitRandomly() - Splits the dataframe into a single training and testing group randomly (70% training, 30% testing). Almost all other functions rely on this one.
    - groups_of_n() - Splits the dataframe into homogeneous groups of N strokes, where each group is either all-genuine or all-ingenuine. 
* eer_vs_strokes.py - An example of one main experiment setting, involving combining multiple strokes to reduce the equal error rate. 
  - The dataframe is read in from the features matrix csv file.
  - For each stroke count N ranging from 2-20:
    - For all 41 users:
      - For each session (short term):
        - Split into train/test data, fit the model to the train data.
        - Test groups of N strokes using test_multiple_strokes()
    - Append the mean EER of each user/session to global list 'EERs', to be graphed against N.
  - Graph list 'EERs' against list 'N' (where n ranges from 2:2:20).
