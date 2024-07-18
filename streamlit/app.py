import streamlit as st
import os
import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import h5py
from tensorflow.keras.models import load_model
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from scipy.fft import fft

st.set_page_config(layout='wide', initial_sidebar_state='expanded')
st.markdown(
    "<h1 style='text-align: center; font-size: 32px;'>Bienvenue dans ce tableau de bord de l'aide à la prediction de sexe à partir des données EEG de l'activité cérébrale.</h1>",
    unsafe_allow_html=True
)
# Fonction pour charger des données à partir de différents types de fichiers
def load_data(file, nrows=None):
    file_extension = os.path.splitext(file)[1]
    if file_extension == '.h5':
        # Charger un fichier h5
        with h5py.File(file, 'r') as f:
            ls = list(f.keys())
            print(f"Liste des ensembles de données dans ce fichier {ls}")
            data = f.get('features')
            data = np.array(data)
            return data

    elif file_extension == '.csv':
        # Charger un fichier csv
        data = pd.read_csv(file, nrows=nrows)
        return data.head(nrows)
def apply_fft(data):
    transformed_data = np.zeros_like(data, dtype=complex)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(data.shape[2]):
                transformed_data[i, j, k, :] = fft(data[i, j, k, :])
    return transformed_data

def hjorth_parameters(data):
    """Calculate Hjorth parameters: Activity, Mobility, and Complexity."""
    first_deriv = np.diff(data)
    second_deriv = np.diff(first_deriv)
    
    activity = np.var(data)
    mobility = np.sqrt(np.var(first_deriv) / activity)
    complexity = np.sqrt(np.var(second_deriv) / np.var(first_deriv)) / mobility
    
    return activity, mobility, complexity

def spectral_entropy(data, sampling_rate):
    """Calculate spectral entropy."""
    power_spectrum = np.abs(np.fft.fft(data))**2
    power_spectrum = power_spectrum[:len(power_spectrum) // 2]
    ps_norm = power_spectrum / np.sum(power_spectrum)
    entropy = -np.sum(ps_norm * np.log2(ps_norm + 1e-12))
    return entropy
def extract_features(data, sampling_rate=250):
    features = []
    for sample in data:
        sample_features = []
        for segment in sample:
            for channel in segment:
                # Calcul des features
                hjorth_params = hjorth_parameters(channel)
                entropy = spectral_entropy(channel, sampling_rate)
                sample_features.extend(hjorth_params)
                sample_features.append(entropy)
        features.append(sample_features)
    return np.array(features)

# Charger les fichiers de données
dataX = load_data('C:/Users/seddi/Desktop/Formation Jedha/projet_predict_sex_from_brain_rhythms/X_train_new.h5')
dataY = load_data('C:/Users/seddi/Desktop/Formation Jedha/projet_predict_sex_from_brain_rhythms/y_train_AvCsavx.csv')

# Supprimer la colonne 'id' si elle existe
if 'id' in dataY.columns:
    dataY = dataY.drop('id', axis=1)

# Fonction pour obtenir le résumé du modèle
def get_model_summary(model):
    from io import StringIO
    import sys
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    model.summary()
    summary_string = sys.stdout.getvalue()
    sys.stdout = old_stdout
    return summary_string

# Ajout d'une colonne de labels de genre
if 'label' in dataY.columns:
    dataY['gender_label'] = dataY['label'].map({0: 'Female', 1: 'Male'})
    g_count = dataY['gender_label'].value_counts()
    g_count_df = g_count.reset_index()
    g_count_df.columns = ['gender_label', 'count'] 

    fig_pie = px.pie(g_count_df, names='gender_label', values='count',
                     title='Distribution du Dataset', hole=0.4)

# Charger le modèle CNN
try:
    model_path = 'C:/Users/seddi/Desktop/Formation Jedha/projet_predict_sex_from_brain_rhythms/model_cnn.h5'
    model = load_model(model_path)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Fonction pour prédire le sexe à partir du dataset
def predict_sexe(data):
    data = apply_fft(data)
    data = extract_features(data.real)
    data = data.reshape(data.shape[0], 40, 7, -1)
    predict = model.predict(data)
    return np.argmax(predict, axis=1)

# Fonction pour tracer la matrice de confusion
def plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Homme', 'Femme'], yticklabels=['Homme', 'Femme'], cbar=False, ax=ax)
    ax.set_xlabel('Prédictions')
    ax.set_ylabel('Vraies étiquettes')
    ax.set_title('Matrice de Confusion')
    return fig
# Fonction principale pour l'application Streamlit
def main():
    #st.title('Prédire le sexe à partir de son activité cérébrale')

    # Barre de navigation pour les pages
    page = st.sidebar.selectbox('Page Navigation', ['Introduction', 'Aperçu des Données', 'Prédiction et Matrice de Confusion'])

    if page == 'Introduction':
        st.header("Introduction")
        
 
        st.write("""Le challenge sujet de notre projet est un challenge proposé par une starup de neurotechnologie Dream et consiste 
                 à prédire le sexe d’un individu en fonction de son activité cérébrale.""")
        st.write("""Mesurer l’activité cérébrale 
                 ou autrement dit l’électroencephalographie est un examen médical qui consiste à placer des capteurs sur le crâne d’une personne pour mesurer l’activité électrique des neurones localisés aux alentours de ces capteurs.""")
        st.write(""" Ainsi, on va pouvoir mesurer un champs 
                 magnétique qui sera traduit en time series : le temps selon l’axe des abscisses et l’amplitude de 
                 l’activité mesurée à l’endroit du capteur en ordonnée.""")
    elif page == 'Aperçu des Données':
        st.header("Aperçu des Données")
        
        # Option de vue de données dans la barre latérale
        option = st.sidebar.selectbox('Choisissez la vue des données:', ('X Data', 'Y Data', 'Distribution Femme/Homme'))

        if option == 'X Data':
            st.write(f"The shape of the data is {dataX.shape}")
            st.write("X Data", dataX[:1])

        else:
            col1, col2 = st.columns(2)  # Division en deux colonnes égales
            with col1:
                st.write("Y Data", dataY)
            # Afficher la figure 'fig_pie' dans la deuxième colonne
            with col2:
                st.plotly_chart(fig_pie)
        
        # Bouton pour afficher les graphiques de signaux
        if st.sidebar.button('Aperçu de signaux'):
            t = 2
            fr = 250
            x = [t / fr for t in range(len(dataX[0][0][0]))]
            fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(10, 8))
            titles = ['sig0', 'sig1', 'sig2', 'sig3', 'sig4', 'sig5', 'sig6']

            for i in range(7):
                row, col = divmod(i, 4)
                sig = dataX[0][0][i]
                axes[row, col].plot(x, sig)
                axes[row, col].set_title(titles[i])

            plt.tight_layout(pad=3.0)
            fig.suptitle('Les signaux', fontsize=15, y=0.95)

            if axes.shape[1] == 4:
                axes[1, 3].axis('off')

            st.pyplot(fig)
        
    elif page == 'Prédiction et Matrice de Confusion':
        #st.header("Prédiction et Matrice de Confusion")

        # Option de modèle CNN dans la barre latérale
        cnn_model_option = st.sidebar.selectbox(
            'Actions du Modèle CNN:',
            ('Sommaire du Modèle', 'Prédiction du Modèle CNN','Classification Report', 'Matrice de Confusion')
        )

        if cnn_model_option == 'Sommaire du Modèle':
            summary = get_model_summary(model)
            st.text(summary)

        elif cnn_model_option == 'Prédiction du Modèle CNN':
            num_predictions = st.number_input('Combien de prédictions?', min_value=1, max_value=len(dataX), value=5)
            selected_data = dataX[:num_predictions]
            selected_data = selected_data.reshape(selected_data.shape[0], 40, 7, -1)

            predictions = predict_sexe(selected_data)
            results = ['Homme' if predict == 0 else 'Femme' for predict in predictions]

            results_counter = Counter(results)

            for i, result in enumerate(results):
                st.write(f"Prediction {i + 1}: {result}")

            st.write("Nombre de prédictions 'Homme':", results_counter['Homme'])
            st.write("Nombre de prédictions 'Femme':", results_counter['Femme'])
        elif cnn_model_option == 'Classification Report':  
            y_true = dataY['label'].values  
            predictions = predict_sexe(dataX)
            st.text(classification_report(y_true, predictions, target_names=['Homme', 'femme']))
            
        elif cnn_model_option == 'Matrice de Confusion':  
            y_true = dataY['label'].values  
            predictions = predict_sexe(dataX)

            cm = confusion_matrix(y_true, predictions)
            fig = plot_confusion_matrix(cm)
            st.pyplot(fig)

            plt.close(fig)    

if __name__ == '__main__':
    main()
