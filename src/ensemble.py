# -*- coding: utf-8 -*-
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt


class Ensemble:
    def __init__(self, X_train, X_test, y_train, y_test,
                 number_of_models=20):
        print("CREATING ENSEMBLE")
        self.number_of_models = number_of_models
        self.encoder = LabelEncoder()
        self.encoder.fit(y_train)
        train = X_train.copy()
        train["CLASE"] = y_train
        X_train, X_valid, y_train, y_valid = train_test_split(
                X_train, y_train, test_size=0.33, random_state=42,
                stratify=y_train)
        self.fit(X_train, y_train)
        self.fit_ensemble(X_valid, y_valid)
        y_pred = self.predict(X_test)
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
        # plot_confusion_matrix(self, X_test, y_test,
        #                       display_labels=target_names,
        #                       cmap=plt.cm.Blues,
        #                       normalize=None,
        #                       xticks_rotation="vertical")
        
    def get_weights(self, y):
        vector = y.copy()
        weights = pd.DataFrame({"Y": vector})
        total = weights["Y"].count()
        counts = weights["Y"].value_counts()
        for place in counts.index:
            weights.loc[weights["Y"] == place,
                        "WEIGHTS"] = 1 - (counts[place] / total)
        return weights["WEIGHTS"]

    def fit(self, X_train, y_train):
        print("TRAININING")
        total = X_train.copy()
        total["CLASE"] = y_train

        residential = total.loc[total["CLASE"] == "RESIDENTIAL",
                                :]
        not_residential = total.loc[total["CLASE"] != "RESIDENTIAL",
                                    :]
        self.models = []
        start = 0
        step = int(len(residential) / self.number_of_models)
        finish = step
        for i in range(self.number_of_models):
            print(f"START: {start}")
            print(f"FINISH: {finish}")
            dataset = residential[start:finish]
            dataset = pd.concat([dataset, not_residential])
            y = dataset["CLASE"]
            weights = self.get_weights(y)
            X = dataset.drop("CLASE", axis=1)
            print(f"TRAINING SET: {X.shape}")
            model = XGBClassifier(eta=0.1,
                                  objective='multi:softmax',
                                  num_class=len(y.unique()))
            model.fit(X, y, sample_weight=weights)
            self.models.append(model)
            start = finish
            finish += step

    def fit_ensemble(self, X, y):
        print("FITTING ENSEMBLE")
        self.meta_model = XGBClassifier(
                                    eta=0.1,
                                    objective='multi:softmax',
                                    num_class=len(y.unique()))
        train = pd.DataFrame()
        for i in range(len(self.models)):
            predictions = self.models[i].predict(X)
            # predictions = self.encoder.transform(predictions)
            train[str(i)] = predictions
        # weights = self.get_weights(y)
        # self.meta_model.fit(train, y, sample_weight=weights)

    def predict(self, X):
        train = pd.DataFrame()
        for i in range(len(self.models)):
            predictions = self.models[i].predict(X)
            # predictions = self.encoder.transform(predictions)
            train[str(i)] = predictions
        self.train = train
        return train.mode(axis=1)[0]
        # return self.meta_model.predict(train)



