# Importer les modules nécessaires
import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model


def prediction():

    # Charger le modèle sauvegardé
    model = load_model("model/lxgb")

    # Créer une interface utilisateur simple
    st.title("Prédiction du désabonnement des clients")
    st.subheader("Auteur : TAYAWELBA DAWAI HESED -  KIA-22-2A-435")
    st.write("Entrez les informations d'un client et obtenez une prédiction de son statut de désabonnement.")

    # Créer des widgets pour entrer les données du client
    credit_score = st.sidebar.number_input("CreditScore", min_value=0, max_value=1000)
    geography = st.sidebar.selectbox("Geography", ["France", "Spain", "Germany"])
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    age = st.sidebar.number_input("Age", min_value=0)
    tenure = st.sidebar.number_input("Tenure", min_value=0, max_value=10)
    balance = st.sidebar.number_input("Balance", min_value=0.0)
    num_of_products = st.sidebar.number_input("NumOfProducts", min_value=1)
    has_cr_card = st.sidebar.selectbox("HasCrCard", [0, 1])
    is_active_member = st.sidebar.selectbox("IsActiveMember", [0, 1])
    estimated_salary = st.sidebar.number_input("EstimatedSalary", min_value=0.0)

    # Créer un bouton pour lancer la prédiction
    if st.sidebar.button("Prédire"):
        # Créer un dataframe avec les données du client
        data = {"CreditScore": credit_score,
                "Geography": geography,
                "Gender": gender,
                "Age": age,
                "Tenure": tenure,
                "Balance": balance,
                "NumOfProducts": num_of_products,
                "HasCrCard": has_cr_card,
                "IsActiveMember": is_active_member,
                "EstimatedSalary": estimated_salary}
        data_df = pd.DataFrame(data, index=[0])

        # Faire la prédiction avec le modèle chargé
        prediction = predict_model(model, data=data_df)

        # Afficher le résultat de la prédiction

        #st.write(f"La probabilité que le client se désabonne est de {prediction['prediction_label'][0]*100:.2f}%.")
        if prediction['prediction_label'][0] == 0:
            st.success("Le client ne va pas se désabonner.")
        else:
            st.error("Le client va se désabonner.")
        
if __name__ == '__main__':
    prediction()