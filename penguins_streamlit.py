import streamlit as st
import pickle
import pandas as pd

st.title("Penguin Classifier : A machine learning app")
st.write("This app uses 6 inputs about physical attributes and geographical locations"
         " to predict the species of penguion using a model built on the Palmer's Penguin's"
          " dataset. Use the form below to get started...")

password_guess = st.text_input("Enter password")
if password_guess != st.secrets["password"]:
    st.stop()

with open("random_forest_penguin_model.pickle", "rb") as rf_pickle:
    model = pickle.load(rf_pickle)

with open("random_forest_penguin_map.pickle", "rb") as uniques_pickle:
    unique_penguin_mapping = pickle.load(uniques_pickle)

with open("feature_dtypes.pickle", "rb") as feature_dtypes:
    feature_dict = pickle.load(feature_dtypes)

with open("cat_feature_unique_values.pickle", "rb") as cat_feature_unique_values:
    cat_feature_unique_values = pickle.load(cat_feature_unique_values)

user_input_dict = {}
st.header("Enter observation values...")

with st.form("user_inputs"):

    st.subheader("Numeric features")
    for v in feature_dict["numeric"]:
        user_input_dict[v] = st.number_input(label=v, min_value=0.0)

    st.subheader("Categorical features")
    for v in feature_dict["cat"]:
        user_input_dict[v] = st.selectbox(label=v, options=["none"] + 
                                        cat_feature_unique_values[v])
    st.form_submit_button()

st.text("You supplied the following inputs...")
user_input = pd.DataFrame({k: [v] for k, v in user_input_dict.items()})
st.write(user_input)

try:
    prediction = model.predict(user_input)
    species_pred = unique_penguin_mapping[prediction][0]
    st.text(f"Based on the inputs, the penguin is of species '{species_pred}'")
    st.text("The features used in the model ranked by relative importance is shown below")
    st.image("feature_importance.png")
except:
    st.stop()