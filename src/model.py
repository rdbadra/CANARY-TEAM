# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import seaborn as sns

# Transformations
from sklearn.preprocessing import MinMaxScaler

# Sklearn Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection  import GridSearchCV

# imblearn Models
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.ensemble import EasyEnsembleClassifier

# Xgboost
from xgboost import XGBClassifier

# Metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

# Tiempo
import time

# Plots
import matplotlib.pyplot as plt
from xgboost import plot_importance

# Feature Selection
from sklearn.feature_selection import SelectFromModel


class ModelSelection:
    def __init__(self, X_train, X_test, y_train, y_test):
        print("MODEL SELECTION")
        # Delete ID, X, Y

        model = self.train(X_train, y_train)
        y_pred = model.predict(X_test)

        target_names = ["AGRICULTURE",
                        "INDUSTRIAL",
                        "OFFICE",
                        "OTHER",
                        "PUBLIC",
                        "RESIDENTIAL",
                        "RETAIL"]
        self.report = classification_report(y_test,
                                            y_pred,
                                            target_names=target_names)
        self.confusion_matrix = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(model, X_test, y_test,
                              display_labels=target_names,
                              cmap=plt.cm.Blues,
                              normalize=None,
                              xticks_rotation="vertical")
        plt.show()
        self.model = model

    def get_weights(self, y):
        vector = y.copy()
        weights = pd.DataFrame({"Y": vector})
        total = weights["Y"].count()
        counts = weights["Y"].value_counts()
        for place in counts.index:
            weights.loc[weights["Y"] == place,
                        "WEIGHTS"] = 1 - (counts[place] / total)
        return weights["WEIGHTS"]

    def stacking(self, X_train, y_train):
        print("STACKING")
        estimators = [
            ('rf', RandomForestClassifier(max_depth=2, random_state=42,
                                          class_weight='balanced_subsample')),
            ('bag', BalancedBaggingClassifier(random_state=42)),
            ('balanced_rf', BalancedRandomForestClassifier()),
            ('easy', EasyEnsembleClassifier()),
            ('xgb', XGBClassifier(eta=0.1, objective='multi:softmax',
                                  num_class=len(y_train.unique())))
        ]
        xgb = XGBClassifier(eta=0.1, objective='multi:softmax',
                            num_class=len(y_train.unique()))
        stack = StackingClassifier(
            estimators=estimators, final_estimator=xgb
        )
        return stack

    def train(self, X_train, y_train):
        print("TRAINING MODEL")
        start = time.time()
        # clf = RandomForestClassifier(max_depth=2, random_state=42,
        #                              class_weight="balanced")
        # clf = SVC(gamma='auto', random_state=42)
        # clf = BalancedBaggingClassifier(random_state=42)
        # model = RandomForestClassifier(max_depth=2, random_state=42,
        #                               class_weight='balanced_subsample')
        # clf = BalancedRandomForestClassifier()
        # clf = EasyEnsembleClassifier()
        weights = self.get_weights(y_train)
        self.weights = weights
        model = XGBClassifier(eta=0.1, objective='multi:softmax',
                              num_class=len(y_train.unique()))
        # clf = LogisticRegression(solver='lbfgs')
        # model = self.stacking(X_train, y_train)
# =============================================================================
#         params = {
#                 'max_depth': range(3, 10, 2),
#                 'min_child_weight': range(1, 6, 2)
#             }
#         grid = GridSearchCV(model, param_grid=params)
#         grid.fit(X_train, y_train, sample_weight=weights
#                   )
#         self.grid = grid
#         model = grid.best_estimator_
# =============================================================================
        model.fit(X_train, y_train, sample_weight=weights)
        end = time.time()
        print(f"Tiempo de entrenamiento: {end - start}")
        return model
