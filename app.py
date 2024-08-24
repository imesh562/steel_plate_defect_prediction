from flask import Flask, request, render_template
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import joblib

app = Flask(__name__)

model = tf.keras.models.load_model('model.h5')
scaler = joblib.load('scaler.pkl')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    features = [float(request.form[col]) for col in [
        'X_Minimum', 'X_Maximum', 'Y_Minimum', 'Y_Maximum', 'Pixels_Areas', 'X_Perimeter',
        'Y_Perimeter', 'Sum_of_Luminosity', 'Minimum_of_Luminosity', 'Maximum_of_Luminosity',
        'Length_of_Conveyer', 'TypeOfSteel_A300', 'TypeOfSteel_A400', 'Steel_Plate_Thickness',
        'Edges_Index', 'Empty_Index', 'Square_Index', 'Outside_X_Index', 'Edges_X_Index',
        'Edges_Y_Index', 'Outside_Global_Index', 'LogOfAreas', 'Log_X_Index', 'Log_Y_Index',
        'Orientation_Index', 'Luminosity_Index', 'SigmoidOfAreas'
    ]]

    input_data = np.array(features).reshape(1, -1)

    input_data_scaled = scaler.transform(input_data)

    prediction = model.predict(input_data_scaled)

    defect_index = np.argmax(prediction, axis=1)[0]

    defects = ['Bumps', 'Dirtiness', 'K_Scatch', 'Other_Faults', 'Pastry', 'Stains', 'Z_Scratch']
    result = defects[defect_index]

    prediction_table = dict(zip(defects, prediction[0]))

    return render_template('result.html', prediction_text=f'Defect: {result}', prediction_table=prediction_table)


if __name__ == "__main__":
    app.run(debug=True)
