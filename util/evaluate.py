import torch as th
from sklearn import metrics


def get_metrics(pred_ratings, rating_values):
    prob = th.sigmoid(pred_ratings).view(-1)
    y_pred = (prob >= 0.5).int().cpu().numpy()
    y_score = pred_ratings.view(-1).cpu().tolist()
    y_true = rating_values.cpu().tolist()

    accuracy = metrics.accuracy_score(y_true, y_pred)
    precision1 = metrics.precision_score(y_true, y_pred, zero_division=0)
    recall1 = metrics.recall_score(y_true, y_pred, zero_division=0)
    f1 = metrics.f1_score(y_true, y_pred, zero_division=0)
    mcc = metrics.matthews_corrcoef(y_true, y_pred)

    fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
    auc = metrics.auc(fpr, tpr)
    precision2, recall2, _ = metrics.precision_recall_curve(y_true, y_score)
    aupr = metrics.auc(recall2, precision2)

    return auc, aupr, accuracy, precision1, recall1, f1, mcc
