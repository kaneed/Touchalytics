import pickle
import numpy as np
import datasplitter

def predict_n(model, strokes):
    threshold = 0.5
    s = 0
    probs = model.predict_proba(strokes)
    for prob in probs:
        s += prob[1]
    avg_prob = s / len(probs)
    return avg_prob > threshold
    
with open("user_models/user_41.sav", "rb") as f:
    model = pickle.load(f)
with open("user_41_X_test.txt", "rb") as f:
    X_test = pickle.load(f)
with open("user_41_y_test.txt", "rb") as f:
    y_test = pickle.load(f)


arbitrary_stroke = np.array([0.56594178, 0.02241248, 0.39974136, 0.19346856, 0.44017783,
       0.52979574, 0.38802322, 0.9325964 , 0.66666667, 0.73448748,
       0.0943926 , 0.02833163, 0.0030737 , 0.99968263, 0.25207033,
       0.00305802, 0.01588829, 0.59718633, 0.76776212, 0.46778442,
       0.30674941, 0.73548983, 0.18394296, 0.98465287, 0.01062106,
       0.25216912, 0.39169839, 0.05882291, 0.5       , 0.        ,
       0.        ])

#print(arbitrary_stroke)
#pred = model.predict([arbitrary_stroke])
#print("Prediction: " + str(pred))
#score = round(model.score(X_test, y_test), 2)
#print("Overall score: " + str(score))

print("Combining Multiple Strokes")
n = 4 # stroke count
cx, cy = datasplitter.clump_groups_of_n(X_test, y_test, n)

false_positives = 0
true_positives = 0
false_negatives = 0
true_negatives = 0
total = len(cx)

for i in range(total):
    pred = predict_n(model, cx[i])
    real = cy[i]
    if pred and real:
        true_positives += 1
    if not pred and not real:
        true_negatives += 1
    if pred and not real:
        false_positives += 1
    if not pred and real:
        false_negatives += 1

correct = true_positives + true_negatives
accuracy = correct / total
print("Strokes Count (n): " + str(n))
print("Samples: " + str(total) + " groups of " + str(n))
print("False Positive Rate: " + str(false_positives/total) + " [" + str(false_positives) + "]")
print("False Negative Rate: " + str(false_negatives/total) + " [" + str(false_negatives) + "]")
print("Accuracy: " + str(accuracy))
    
    






