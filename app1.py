from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import logging

app = Flask(__name__)

# Configurar el registro
logging.basicConfig(level=logging.DEBUG)

# Cargar el modelo, el scaler y el encoder
model = joblib.load("Animacion_RF_model.pkl")  # Modelo de Random Forest
scaler = joblib.load("x_scaler.pkl")  # Scaler usado para escalar las variables seleccionadas
encoder = joblib.load("encoder.pkl")  # Encoder para transformar valores categóricos

app.logger.debug("Modelo, scaler y encoder cargados correctamente. Verificando compatibilidad...")
app.logger.debug(f"Encoder categories: {encoder.categories_}")
app.logger.debug(f"Scaler feature names: {scaler.get_feature_names_out()}")

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos enviados en el request
        source = request.form['source'].title()  # Normalizar texto
        genre = request.form['genre'].title()  # Normalizar texto
        airing = request.form['airing'].lower()  # Normalizar booleano
        rank = float(request.form['rank'])
        members = float(request.form['members'])
        reviewers = float(request.form['reviewers'])

        app.logger.debug(f"Datos recibidos: source={source}, genre={genre}, airing={airing}, rank={rank}, members={members}, reviewers={reviewers}")

        # Convertir datos categóricos a numéricos usando el encoder
        cat_data = [[source, genre, airing]]  # Formato 2D para el encoder
        encoded_cat_data = encoder.fit_transform(cat_data)
        app.logger.debug(f"Valores codificados: {encoded_cat_data}")

        # Crear un DataFrame con los datos transformados
        data_df = pd.DataFrame([[encoded_cat_data[0][0], encoded_cat_data[0][1], encoded_cat_data[0][2], rank, members, reviewers]],
                              columns=['source', 'genre', 'airing', 'rank', 'members', 'reviewers'])
        app.logger.debug(f"DataFrame antes de escalar: {data_df}")

        # Escalar los datos usando el scaler cargado
        data_df_scaled = pd.DataFrame(scaler.transform(data_df), columns=data_df.columns)
        app.logger.debug(f"DataFrame escalado: {data_df_scaled}")

        # Realizar predicciones con el modelo
        prediction = model.predict(data_df_scaled)
        app.logger.debug(f"Predicción: {prediction[0]}")

        return jsonify({'Animation_score': prediction[0]})
    except Exception as e:
        app.logger.error(f"Error en la predicción: {str(e)}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)