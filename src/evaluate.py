import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def uauc_score(y_true, y_pred, userid):
    df = pd.DataFrame({"userid": userid, "y_pred": y_pred, "y_true": y_true})
    label_nunique = df.groupby("userid")["y_true"].nunique().reset_index()
    users = label_nunique.loc[label_nunique["y_true"] == 2, "userid"]
    df = df.loc[df["userid"].isin(users)]

    score = df.groupby("userid").apply(lambda x: roc_auc_score(x["y_true"].values, x["y_pred"].values))
    return np.mean(score)
