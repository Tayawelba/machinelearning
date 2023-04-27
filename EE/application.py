import streamlit as st
import pages.preprocessing
import pages.prediction

st.set_page_config(
    page_title="EE - TAYAWELBA DAWAI HESED",
    page_icon="💻",
)

#st.sidebar.title("Navigation")
selection =  "Accueil"
#selection = st.sidebar.radio("Choisissez une page", ["Accueil", "Preprocessing", "Prediction"])

if selection == "Accueil":
    st.title("Application EE - Mon projet de Machine Learning")
    st.subheader("Auteur : TAYAWELBA DAWAI HESED -  KIA-22-2A-435")
    st.markdown("[Code source](https://github.com/#)")


    # Create body
    
    st.write('Ce projet utilise le dataset Churn_Modelling pour prédire si un client va quitter la banque ou pas.')

    # Créer une liste d'images
    images = ['images/image1.jpeg', 'images/image2.jpeg', 'images/image3.jpeg', 'images/image4.jpeg', 'images/image5.jpeg']

    # Créer un slider pour choisir un indice
    index = st.slider('Choisissez une image', 0, len(images) - 1)
    # Afficher l'image correspondant à l'indice
    st.image(images[index], caption=f'Image {index + 1}', use_column_width=True)
    # Créer un lien vers le code source du projet
    

elif selection == "Preprocessing":
    pages.preprocessing.preprocessing()
elif selection == "Prediction":
    pages.prediction.prediction()