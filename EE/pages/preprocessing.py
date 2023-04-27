import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import NuSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.svm import SVC
from sklearn.metrics import precision_score
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve

st.set_option('deprecation.showPyplotGlobalUse', False)

def preprocessing():
    st.title("Preprocessing du désabonnement des clients")
    st.subheader("Auteur : TAYAWELBA DAWAI HESED -  KIA-22-2A-435")

    # Fonction pour importer les donnees
    st.cache(persist=True)
    def load_data():
        data = pd.read_csv('data/Churn_Modelling_cc.csv')
        return data
    
    #afficher mes donnees
    df = load_data()
    df_sample = df.sample(100)
    if st.sidebar.checkbox("Afficher les donnees brutes",False):
        st.subheader("Jeu de donnees : Echantillon de 100 observateur")
        st.write(df_sample)

    seed = 123

    # Train / Test split
    def split(df):
        y = df['Exited']
        X = df.drop('Exited', axis = 1)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            stratify=y,
            random_state=seed
        )
        return X_train, X_test, y_train, y_test 

    X_train, X_test, y_train, y_test = split(df)

    classifier = st.sidebar.selectbox(
        "Classificateur",
        ("Random Forest", "SVM", "Logistic Regression")
    )

    # Analyse de la performance des modeles

    def plot_perf(graphes):
        if 'Confusion matrix' in graphes:
            st.subheader('Matrice de confusion')
            plot_confusion_matrix(
                model,
                X_test,
                y_test
            )
            st.pyplot()

        if 'ROC curve' in graphes:
            st.subheader('Courbe Roc Curve')
            plot_roc_curve(
                model,
                X_test,
                y_test
            )
            st.pyplot()

        if 'Precision-Recall curve' in graphes:
            st.subheader('Courbe Precesion-Recall')
            plot_precision_recall_curve(
                model,
                X_test,
                y_test
            )
            st.pyplot()


    # Regrssion logistique

    if classifier == "Logistic Regression":
        st.sidebar.subheader("Hyperparametre du modele")
        hyp_c = st.sidebar.number_input(
            "Choisir la valeur du parametre de regularisation",
            0.01, 10.0
        )
        n_max_iter = st.sidebar.number_input(
            "Choisir le nombre max d'iterations",
            100, 1000, step=10
        )

        graphes_perf = st.sidebar.multiselect(
            "Choisir un graphique de performance du modele ML",
            ("Confusion matrix", "ROC curve", "Precision-Recall curve")
        )
        

        if st.sidebar.button("Execution", key="classify"):
            st.subheader("Logistic Regression Results")

            # Initiation d'un objet LogisticRegression
            model = LogisticRegression(
                C = hyp_c,
                max_iter = n_max_iter,
                random_state=seed
            )

            # Entrainement de l'algorithme
            model.fit(X_train,y_train)

            #Predictions
            y_pred = model.predict(X_test)

            #Metriques de performance
            accuracy = model.score(X_test, y_test)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)

            #afficher les metrique dans l'applis

            st.write("Accuracy: ", accuracy.round(3))
            st.write("Precesion: ", precision.round(3))
            st.write("Recall: ", recall.round(3))

            # Afficher les graphiques de performance
            plot_perf(graphes_perf)



    # SVM

    if classifier == "SVM":
        st.sidebar.subheader("Hyperparametre du modele")
        hyp_c = st.sidebar.number_input(
            "Choisir la valeur du parametre de regularisation",
            0.01, 10.0
        )
        kernel = st.sidebar.radio(
            "Choisir le Kernel",
            ("rbf", "linear")
        )
        gamma = st.sidebar.radio(
            "Gamma",
            ("scale","auto")
        )

        graphes_perf = st.sidebar.multiselect(
            "Choisir un graphique de performance du modele ML",
            ("Confusion matrix", "ROC curve", "Precision-Recall curve")
        )
        

        if st.sidebar.button("Execution", key="classify"):
            st.subheader("Support Vector Machine (SVM) Results")

            # Initiation d'un objet SVC
            model = SVC(
                C = hyp_c,
                kernel = kernel,
                gamma=gamma
            )

            # Entrainement de l'algorithme
            model.fit(X_train,y_train)

            #Predictions
            y_pred = model.predict(X_test)

            #Metriques de performance
            accuracy = model.score(X_test, y_test)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)

            #afficher les metrique dans l'applis

            st.write("Accuracy: ", accuracy.round(3))
            st.write("Precesion: ", precision.round(3))
            st.write("Recall: ", recall.round(3))

            # Afficher les graphiques de performance
            plot_perf(graphes_perf)




    # Random Forest
    if classifier == "Random Forest":
        st.sidebar.subheader("Hyperparametre du modele")
        n_arbres = st.sidebar.number_input(
            "Choisir le Nombre d'abres dans le foret",
            100, 1000, step=10
        )
        profondeur_arbre = st.sidebar.number_input(
            "Choisir la Profondeur Max d'un arbre",
            1, 20, step=1
        )
        bootstrap = st.sidebar.radio(
            "Echantillons boostrap lors de la creation d'un arbre",
            (True,False)
        )

        graphes_perf = st.sidebar.multiselect(
            "Choisir un graphique de performance du modele ML",
            ("Confusion matrix", "ROC curve", "Precision-Recall curve")
        )
        

        if st.sidebar.button("Execution", key="classify"):
            st.subheader("Random Forest Results")

            # Initiation d'un objet RandomForestClassifier
            model = RandomForestClassifier(
                n_estimators=n_arbres,
                max_depth=profondeur_arbre,
                bootstrap=bootstrap,
                random_state=seed
            )

            # Entrainement de l'algorithme
            model.fit(X_train,y_train)

            #Predictions
            y_pred = model.predict(X_test)

            #Metriques de performance
            accuracy = model.score(X_test, y_test)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)

            #afficher les metrique dans l'applis

            st.write("Accuracy: ", accuracy.round(3))
            st.write("Precesion: ", precision.round(3))
            st.write("Recall: ", recall.round(3))

            # Afficher les graphiques de performance
            plot_perf(graphes_perf)






if __name__ == '__main__':
    preprocessing()