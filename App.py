import streamlit as st
import pandas as pd
import tensorflow as tf
import shared_functions as sf
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

#### Defining some general properties of the app
#########################################################
st.set_page_config(
    page_title = 'Solarproduktion Vorhersage',
    page_icon="☀️",
    layout="wide"
)


#### Import Model & Scaler
#########################################################
scaler = pickle.load(open('scaler.sav', 'rb'))
model = tf.keras.models.load_model('new_model')


#### Define Header of app
#########################################################
st.title("Solarenergieproduktion der St.Galler Stadwerke")
st.write('Dieses Dashboard für die Solarproduktion ist ein Werkzeug, mit dem Benutzer die geschätzte Strommenge überwachen können, die von Solaranlagen erzeugt wird. Diese Informationen können dazu beitragen, die Kosten für den Energieeinkauf zu minimieren.')
st.header("Vorhersage der Tagesproduktion")


#### Define Upload Field
#########################################################
st.sidebar.subheader("Datenimport")
uploaded_data = st.sidebar.file_uploader("Wählen Sie eine Datei mit Wetterdaten für die Vorhersage aus", type="csv")


#### Data Transformation & Prediction
#########################################################
global df
if uploaded_data is not None:
    print(uploaded_data)
    try:
        new_data = pd.read_csv(
            uploaded_data,
            header=0,
            index_col='date',
            parse_dates=True)
    except Exception as e:
        print(e)
        # Add User Feedback
        st.error('Die Daten konnten nicht korrekt geladen werden', icon="🚨")
    
    only_x_data = new_data.drop('production_adj', axis=1)
    
    # Data is transformed into 3D-structure
    n_timesteps = 288
    y_new_data = new_data.production_adj[n_timesteps - 1:]
    X_new_data = sf.to_sequences(new_data.drop(columns='production_adj'), length=n_timesteps)

    # Data is scaled based on training data
    new_data_scaled = scaler.transform(X_new_data.reshape(-1, X_new_data.shape[-1])).reshape(X_new_data.shape)
    
    # Predictions are made
    predictions = model.predict(new_data_scaled)
    
    # Predictions are saved in DataFrame
    df = y_new_data.to_frame(name='Actuals')
    df['progn. Solarproduktion'] = predictions
    df.drop('Actuals', axis=1, inplace=True)


    row1_col1, row1_col2  = st.columns([2,1])

    # Add User Feedback
    st.success('Die Vorhersage konnte erstellt werden', icon="✅")
    

#### Creating Plot
#########################################################

# defining two columns for layouting plots 
row2_col1, row2_col2  = st.columns([99,1])

try:
    # Create a standard seaborn line chart 
    fig, ax = plt.subplots(figsize=(8,3.7))
    ax = sns.lineplot(data=df, x=df.index, y='progn. Solarproduktion')
    # Put seaborn figure in col 1 
    row2_col1.pyplot(fig)

    # Add Download Button
    st.download_button(label = "Download der prognositzierten Solarproduktion",
                       data = df.to_csv().encode("utf-8"),
                       file_name = "prognostizierte_solarproduktion.csv")
    
    # Add checkbox allowing us to display raw data
    if st.checkbox("Zeige Wetterdaten an", False):
        st.subheader("Wetterdaten")
        st.write(only_x_data)

except Exception as e:
    print(e)
    # Add User Feedback
    st.info('Wählen Sie eine Datei mit Wetterdaten für die Vorhersage aus', icon="ℹ️")
