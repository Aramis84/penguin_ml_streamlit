import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

penguin_df = pd.read_csv("penguins.csv")

penguin_df.dropna(inplace=True)

target_col = "species"
target = penguin_df[target_col]

features = penguin_df.drop([target_col, "year"], axis=1).copy()

numeric_features = features.select_dtypes(include=np.number).columns.to_list()
cat_feaures = [c for c in features.columns if c not in numeric_features]
cat_feature_unique_values = {k : list(features[k].unique()) for k in cat_feaures}

feature_dict = {"cat":cat_feaures, "numeric":numeric_features}

cat_transformer = OneHotEncoder()
preprocessor = ColumnTransformer(transformers=[("cat", cat_transformer, cat_feaures)],
                                 remainder="passthrough")

target1, uniques = pd.factorize(target)
x_train, x_test, y_train, y_test = train_test_split(features,
                                                    target1, 
                                                    test_size=0.8,
                                                    stratify=target1)

rfc = RandomForestClassifier(random_state=42)
clf_pipeline = Pipeline(steps=[("preprocessor", preprocessor),("classifier", rfc)])

model = clf_pipeline.fit(x_train, y_train)
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
print(f"Train acc : {train_acc} | Test acc : {test_acc}")

feat_names_trans = preprocessor.get_feature_names_out()
feat_names = [v.split("__")[-1] for v in feat_names_trans]


fig, ax = plt.subplots()
ax = pd.Series(clf_pipeline["classifier"].feature_importances_, index=feat_names).plot(kind='barh')
plt.title("Feautures ranked in terms of importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
fig.savefig("feature_importance.png")

with open("random_forest_penguin_model.pickle", "wb") as rf_pickle:
    pickle.dump(model, rf_pickle)

with open("random_forest_penguin_map.pickle", "wb") as uniques_pickle:
    pickle.dump(uniques, uniques_pickle)

with open("feature_dtypes.pickle", "wb") as feature_dtypes:
    pickle.dump(feature_dict, feature_dtypes)

with open("cat_feature_unique_values.pickle", "wb") as cat_feature_unique_values_pickle:
    pickle.dump(cat_feature_unique_values, cat_feature_unique_values_pickle)
    