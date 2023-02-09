from tensorflow.python.keras import backend as K

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    potential_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall_score = true_positives / (potential_positives + K.epsilon())
    return recall_score

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predictive_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision_score = true_positives / (predictive_positives + K.epsilon())
    return precision_score

def f1_score(y_true, y_pred):
    precision_score = precision(y_true, y_pred)
    recall_score = recall(y_true, y_pred)
    f1_score = 2 * (precision_score * recall_score) \
               / (precision + recall + K.epsilon())
    return f1_score