import streamlit as st
import pandas as pd
import requests
import json

df = pd.read_csv('test_df.csv', sep=';')

sample_df = df.sample(n=100, random_state=42)

id_selected = st.selectbox('Select an ID', sample_df['SK_ID_CURR'].astype(int).unique())

input_data = sample_df.loc[sample_df['SK_ID_CURR'] == id_selected]

# Vérifier si le bouton a été cliqué
if st.button('Obtenir la prédiction'):
    # Préparer les données pour la requête en format 'dataframe_split'
    request_data = {
        "dataframe_split": input_data.to_dict(orient="split")
    }

    # Adresse de votre serveur MLflow
    url = "http://localhost:5001/invocations"

    # Envoyer la requête POST
    response = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        data=json.dumps(request_data)
    )

    if response.status_code == 200:
        st.write(response.json())

    else:
        st.write("Erreur lors de la requête au serveur MLflow.")
