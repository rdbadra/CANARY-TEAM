# -*- coding: utf-8 -*-
from src.model import ModelSelection
from src.ensemble import Ensemble
from src.data_treatment import DataTreatment


import matplotlib.pyplot as plt
from xgboost import plot_importance

import pandas as pd
import numpy as np
import time

path = r"C:\Users\roberto.diaz.badra\Documents\Datathon\CANARY TEAM\Modelar_UH2020\Modelar_UH2020.txt"

data = pd.read_csv(path, sep="|")

data_treatment = DataTreatment()
X_train, X_test, y_train, y_test = data_treatment.split_data(data)


# start = time.time()
# model = Ensemble(X_train, X_test, y_train, y_test,
#                  number_of_models=20)
# end = time.time()
# print(f"TOOK: {end - start}")

# report = model.report
# confusion = model.confusion_matrix
# print(confusion)
# a = model.train

search = ModelSelection(X_train, X_test, y_train, y_test)
report = search.report
model = search.model
confusion = search.confusion_matrix
print(confusion)

plot_importance(model, max_num_features=15)
plt.show()
# estimate = r"C:\Users\roberto.diaz.badra\Documents\Datathon\cajamar\Estimar_UH2020\Estimar_UH2020.txt"
# estimate_data = pd.read_csv(estimate, sep="|")

# X_test = data_treatment.map(estimate_data)

# predictions = model.predict(X_test)

# X_test["CLASE"] = predictions

# output = X_test[["CLASE"]].copy()

# output["ID"] = estimate_data[["ID"]].copy()
# output = output[["ID", "CLASE"]]

# output.to_csv("Minsait_Universitat Oberta de Catalunya_CANARY_TEAM_1.txt",
#               sep="|",
#               index=False)

# from sklearn.cluster import KMeans
# from sklearn.cluster import DBSCAN
# from sklearn.cluster import SpectralClustering

# colors = {"RESIDENTIAL": "blue",
#           "AGRICULTURE": "green",
#           "RETAIL": "red",
#           "OTHER": "yellow",
#           "PUBLIC": "orange",
#           "INDUSTRIAL": "pink",
#           "OFFICE": "purple"}
# X_train["CLASE"] = y_train

# centroid_x = np.sum(X_train["X"]) / len(X_train)
# centroid_y = np.sum(X_train["Y"]) / len(X_train)

# clustering = KMeans(n_clusters=7, random_state=0).fit(X_train[["X", "Y"]])
# clustering = DBSCAN(eps=100, min_samples=10).fit(X_train[["X", "Y"]])
# data["CLUSTER"] = pd.Series(kmeans.labels_)
# labels = clustering.labels_
# for color in colors.keys():
#     plt.scatter(X_train.loc[X_train["CLASE"]==color, "X"],
#                 X_train.loc[X_train["CLASE"]==color, "Y"],
#                 c=colors[color], label=color)

# plt.scatter(X_train["X"], X_train["Y"])
# plt.scatter(centroid_x, centroid_y, c="red")
# plt.legend()

# plt.scatter(X_train["X"],
#             X_train["Y"])
