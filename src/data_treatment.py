# -*- coding: utf-8 -*-
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import math

# Data Sampling
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

# Split
from sklearn.model_selection import train_test_split

# Clustering
from sklearn.cluster import KMeans


class DataTreatment:
    def __init__(self):
        print("DATA TREATMENT")

    def clustering(self, data):
        print("CLUSTERING")
        coordinates = data[["X", "Y"]]
        kmeans = KMeans(n_clusters=5, random_state=0).fit(coordinates)
        self.clustering = kmeans
        data["CLUSTER"] = pd.Series(kmeans.labels_)

    def preprocess(self, data):
        # Impute nulls from MAXBUILDINGFLOOR
        self.impute_nulls(data)
        # Convert catastral data to number
        data = self.transform_quality(data)
        # Feature engineering
        self.feature_engineering(data)
        # Delete high correlated variables
        data = self.delete_high_correlations(data)
        # Remove variables
        # data = self.remove_variables(data)
        # Get only numeric columns
        numeric = self.get_numeric(data)
        self.target = data["CLASE"]

        # numeric = self.feature_engineering(numeric)
        # numeric.loc[:, numeric.columns] = self.min_max(numeric)
        # numeric = self.delete_high_correlations(numeric)
        # self.clustering(numeric)
        self.data = numeric.copy()
        return numeric, self.target

    def map(self, testset):
        print("MAPPING")
        data = testset.copy()
        # data.drop(["ID", "X", "Y"], axis=1, inplace=True)
        self.impute_nulls(data)
        data = self.transform_quality(data)

# =============================================================================
#         data["MEDIAN_FLOOR"] = 0
#         for target in self.median_floor.index:
#             value = self.median_floor.loc[target][0]
#             data.loc[data["CADASTRALQUALITYID"] == target,
#                      'MEDIAN_FLOOR'] = value
#         data["MAX_FLOOR_YEAR"] = 0
#         for target in self.max_floor.index:
#             value = self.max_floor.loc[target][0]
#             data.loc[data["CONTRUCTIONYEAR"] == target,
#                      'MAX_FLOOR_YEAR'] = value
#         data["MIN_FLOOR_YEAR"] = 0
#         for target in self.min_floor.index:
#             value = self.min_floor.loc[target][0]
#             data.loc[data["CONTRUCTIONYEAR"] == target,
#                      'MIN_FLOOR_YEAR'] = value
#         data["MEDIAN_FLOOR_YEAR"] = 0
#         for target in self.median_floor_year.index:
#             value = self.median_floor_year.loc[target][0]
#             data.loc[data["CONTRUCTIONYEAR"] == target,
#                      'MEDIAN_FLOOR_YEAR'] = value
# =============================================================================
        data["AREA_PER_FLOOR"] = data["AREA"] / (data["MAXBUILDINGFLOOR"] + 1)
        # data["r"] = np.sqrt((data["X"]**2) + data["Y"]**2)
        # data["theta"] = np.arctan(data["Y"] / data["X"])
        data["DIST"] = np.sqrt(((self.centroid_x - data["X"]) ** 2) +
                               ((self.centroid_y - data["Y"]) ** 2))
        # for i in data.index:
        #     value = data.loc[i, "DIST"]
        #     proportion = int((self.max_dist / value) / 71)
        #     data.loc[i, "DIST_GROUP"] = proportion
        # data["DIST_GROUP"] = 0
        proportion = ((self.max_dist / data["DIST"]) / 10).astype(int)
        data["DIST_GROUP"] = proportion
        # data["MEDIAN_DIST_FLOOR"] = 0
        # for target in self.median_dist_floor.index:
        #     value = self.median_dist_floor.loc[target][0]
        #     data.loc[data["CONTRUCTIONYEAR"] == target,
        #              'MEDIAN_DIST_FLOOR'] = value
        data["NEW_GROUP"] = 0
        data.loc[data["DIST"] > 760146.0, "NEW_GROUP"] = 1
        data.loc[(data["DIST"] > 340846.0) &
                 (data["DIST"] <= 760146.0), "NEW_GROUP"] = 2
        data.loc[(data["DIST"] > 329866.0) &
                 (data["DIST"] <= 340846.0), "NEW_GROUP"] = 3
        data.loc[data["DIST"] < 329866.0, "NEW_GROUP"] = 4
        data.drop(self.to_drop, axis=1, inplace=True)
        numeric = self.get_numeric(data)
        if "CLASE" not in data.columns:
            return numeric
        target = data["CLASE"]
        return numeric, target

    def feature_engineering(self, data):
        print("FEATURE ENGINEERING")
        # CALCULATE MEDIAN FLOOR FOREACH CADASTRALQUALITY
# =============================================================================
#         median_floor = (data[["MAXBUILDINGFLOOR",
#                              "CADASTRALQUALITYID"]]
#                         .groupby("CADASTRALQUALITYID")
#                         .median())
#         self.median_floor = median_floor
#         data["MEDIAN_FLOOR"] = 0
#         for target in median_floor.index:
#             value = median_floor.loc[target][0]
#             data.loc[data["CADASTRALQUALITYID"] == target,
#                      'MEDIAN_FLOOR'] = value
#         # MAX FLOOR PER YEAR
#         max_floor = data[["MAXBUILDINGFLOOR",
#                          "CONTRUCTIONYEAR"]].groupby("CONTRUCTIONYEAR").max()
#         self.max_floor = max_floor
#         data["MAX_FLOOR_YEAR"] = 0
#         for target in max_floor.index:
#             value = max_floor.loc[target][0]
#             data.loc[data["CONTRUCTIONYEAR"] == target,
#                      'MAX_FLOOR_YEAR'] = value
#         # MIN FLOOR PER YEAR
#         min_floor = data[["MAXBUILDINGFLOOR",
#                           "CONTRUCTIONYEAR"]].groupby("CONTRUCTIONYEAR").min()
#         self.min_floor = min_floor
#         data["MIN_FLOOR_YEAR"] = 0
#         for target in min_floor.index:
#             value = min_floor.loc[target][0]
#             data.loc[data["CONTRUCTIONYEAR"] == target,
#                      'MIN_FLOOR_YEAR'] = value
#         # MEDIAN FLOOR PER YEAR
#         median_floor_year = (data[["MAXBUILDINGFLOOR",
#                                   "CONTRUCTIONYEAR"]]
#                              .groupby("CONTRUCTIONYEAR")
#                              .median())
#         self.median_floor_year = median_floor_year
#         data["MEDIAN_FLOOR_YEAR"] = 0
#         for target in median_floor_year.index:
#             value = median_floor_year.loc[target][0]
#             data.loc[data["CONTRUCTIONYEAR"] == target,
#                      'MEDIAN_FLOOR_YEAR'] = value
# =============================================================================
        data["AREA_PER_FLOOR"] = data["AREA"] / (data["MAXBUILDINGFLOOR"] + 1)
        # data["r"] = np.sqrt((data["X"]**2) + data["Y"]**2)
        # data["theta"] = np.arctan(data["Y"] / data["X"])
        # data["DIST"] = np.sqrt(((0 - data["X"]) ** 2) + ((0 - data["Y"]) ** 2))
        self.centroid_x = np.sum(data["X"]) / len(data)
        self.centroid_y = np.sum(data["Y"]) / len(data)
        data["DIST"] = np.sqrt(((self.centroid_x - data["X"]) ** 2) +
                               ((self.centroid_y - data["Y"]) ** 2))
        self.max_dist = data["DIST"].max()
        # for i in data.index:
        #     value = data.loc[i, "DIST"]
        #     proportion = int((self.max_dist / value) / 71)
        #     data.loc[i, "DIST_GROUP"] = proportion
        proportion = ((self.max_dist / data["DIST"]) / 10).astype(int)
        data["DIST_GROUP"] = proportion
        data["NEW_GROUP"] = 0
        data.loc[data["DIST"] > 760146, "NEW_GROUP"] = 1
        data.loc[(data["DIST"] > 340846) &
                 (data["DIST"] <= 760146), "NEW_GROUP"] = 2
        data.loc[(data["DIST"] > 329866) &
                 (data["DIST"] <= 340846), "NEW_GROUP"] = 3
        data.loc[data["DIST"] < 329866, "NEW_GROUP"] = 4
        # median_dist_floor = (data[["MAXBUILDINGFLOOR",
        #                            "DIST"]]
        #                      .groupby("MAXBUILDINGFLOOR")
        #                      .median())
        # self.median_dist_floor = median_dist_floor
        # for target in median_dist_floor.index:
        #     value = median_dist_floor.loc[target][0]
        #     data.loc[data["CONTRUCTIONYEAR"] == target,
        #              'MEDIAN_DIST_FLOOR'] = value

    def split_data(self, dataset):
        target = dataset["CLASE"]
        data = dataset.drop("CLASE", axis=1)
        X_train, X_test, y_train, y_test = self.split(data,
                                                      target)
        X_train["CLASE"] = y_train
        X_train, y_train = self.preprocess(X_train)
        X_test["CLASE"] = y_test.copy()
        X_test, y_test = self.map(X_test)
        # X_test, y_test = self.oversample_smote(X_test, y_test)
        return X_train, X_test, y_train, y_test

    def min_max(self, data):
        print("MIN MAX SCALER")
        scaler = MinMaxScaler()
        data = scaler.fit_transform(data)
        return data

    def delete_high_correlations(self, data, threshold=0.9):
        print("DELETE HIGH CORRELATIONS")
        corr_matrix = data.corr().abs()

        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),
                                          k=1).astype(np.bool))

        # Find index of feature columns with correlation greater than 0.95
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        to_drop.append("X")
        to_drop.append("Y")
        self.to_drop = to_drop
        print(f"HIGH CORRELATED: {to_drop}")
        return data.drop(to_drop, axis=1)

    def impute_nulls(self, data):
        print("IMPUTING MAXBUILDINGFLOOR")
        value = data['MAXBUILDINGFLOOR'].median()
        data['MAXBUILDINGFLOOR'].fillna(value, inplace=True)

    def remove_variables(self, data):
        print("REMOVE ID")
        return data.drop(["ID", "X", "Y"], axis=1)

    def transform_quality(self, dataset):
        print("TRANSFORM QUALITY TO NUMERIC")
        data = dataset.copy()
        data["CADASTRALQUALITYID"].fillna(data["CADASTRALQUALITYID"].mode()[0],
                                          inplace=True)
        letters = {"A": -2,
                   "B": -1,
                   "C": 0}
        for letter in letters.keys():
            data.loc[data["CADASTRALQUALITYID"] == letter,
                     'CADASTRALQUALITYID'] = letters[letter]
        data["CADASTRALQUALITYID"] = data["CADASTRALQUALITYID"].astype(float)
        return data

    def get_numeric(self, data):
        print("GET NUMERIC COLUMNS")
        numeric_data = data.select_dtypes(include=np.number)
        return numeric_data

    def oversample_smote(self, X_train, y_train):
        print("SMOTE")
        oversample = SMOTE()
        return oversample.fit_resample(X_train, y_train)

    def oversample_smote_undersampling(self, X_train, y_train):
        print("SMOTE WITH UNDERSAMPLING")
        print(f"Shape before smote: {X_train.shape}")
        sampling_strategy = {'RESIDENTIAL': 1000,
                             'INDUSTRIAL': 2000,
                             'PUBLIC': 1000,
                             'OFFICE': 1000,
                             'OTHER': 1500,
                             'RETAIL': 10000,
                             'AGRICULTURE': 1500}
        over = SMOTE()
        under = RandomUnderSampler(sampling_strategy=sampling_strategy)
        steps = [('o', over), ('u', under)]
        pipeline = Pipeline(steps=steps)
        X, y = pipeline.fit_resample(X_train, y_train)
        print(f"Shape after SMOTE and undersampling: {X.shape}")
        return X, y

    def split(self, X, y):
        print("SPLITTING DATA")
        X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.33, random_state=42,
                stratify=y)
        # X_train, y_train = self.oversample_smote(X_train, y_train)
        # X_train, y_train = self.oversample_smote_undersampling(X_train,
        #                                                         y_train)
        return X_train, X_test, y_train, y_test
