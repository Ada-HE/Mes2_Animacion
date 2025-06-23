from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import logging

app = Flask(__name__)

# Configurar el registro
logging.basicConfig(level=logging.DEBUG)

# Cargar el modelo entrenado
model = joblib.load("Animacion_random_forest_model.pkl")  # Asegúrate de que el nombre del archivo coincida
app.logger.debug("Modelo cargado correctamente.")

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos enviados en el request
        rank = float(request.form['rank'])
        popularity = float(request.form['popularity'])
        studio = float(request.form['studio'])  # Valor numérico codificado
        source = float(request.form['source'])  # Valor numérico codificado
        members = float(request.form['members'])
        reviewers = float(request.form['reviewers'])
        genre = float(request.form['genre'])    # Valor numérico codificado

        # Crear un DataFrame con los datos usando las características correctas
        data_df = pd.DataFrame([[rank, popularity, studio, source, members, reviewers, genre]],
                              columns=['rank', 'popularity', 'studio', 'source', 'members', 'reviewers', 'genre'])
        app.logger.debug(f"DataFrame enviado: {data_df}")

        # Realizar predicciones
        prediction = model.predict(data_df)
        app.logger.debug(f"Predicción: {prediction[0]}")

        # Devolver las predicciones como respuesta JSON
        return jsonify({'Animation_score': prediction[0]})
    except Exception as e:
        app.logger.error(f"Error en la predicción: {str(e)}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)