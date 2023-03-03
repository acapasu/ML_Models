from flask import Flask,request,jsonify
import joblib
import pandas as pd
import numpy as np



# CREATE FLASK APP
app = Flask(__name__)


# CONNECT POST API CALL ---> predict() Function  #http://localhost:5000/predict
@app.route('/predict',methods=['POST'])
def predict():
    feat_data = request.json
    # Convert JSON request to Pandas DataFrame
    df = pd.DataFrame(feat_data)
    fila_cruda = df.iloc[0]
    fila_procesada = formato_filas(fila_cruda)
    # print(fila_procesada)
    prediction = model.predict(fila_procesada)

    # print(prediction)
    return jsonify({'prediction':str(prediction)})
    
    # # GET JSON REQUEST
    # feat_data = request.json
    # # CONVERT JSON to PANDAS DF (col names)
    # df = pd.DataFrame(feat_data)

    # # df = df.reindex(columns=col_names)
    # # # PREDICT
    # # prediction = list(model.predict(df))
    
    
    # # return jsonify({'prediction':str(prediction)})

model = joblib.load("final_model.pkl") 
col_names = joblib.load("column_names.pkl") 
escalador = joblib.load("escalador.pkl") 

def formato_filas (entrada, column_names = col_names, scaler = escalador):
    respuesta = pd.Series(data = np.zeros(len(column_names)), index = column_names)
    if entrada['island'] == 'Dream':
        respuesta['island_Dream'] = 1
    if entrada['island'] == 'Torgersen':
        respuesta['island_Torgersen'] = 1
    if entrada['sex'] == 'Male':
        respuesta['sex_Male'] = 1
    respuesta['bill_length_mm'] = entrada['bill_length_mm']
    respuesta['bill_depth_mm'] = entrada['bill_depth_mm']
    respuesta['flipper_length_mm'] = entrada['flipper_length_mm']
    respuesta['body_mass_g'] = entrada['body_mass_g']
    
    return escalador.transform([respuesta.values])


# LOAD MY MODEL and LOAD COLUMN NAMES
if __name__ == '__main__':

    model = joblib.load("final_model.pkl") 
    col_names = joblib.load("column_names.pkl") 
    escalador = joblib.load("escalador.pkl") 
    # model = joblib.load("final_model.pkl")
    # col_names = joblib.load('column_names.pkl')
    app.run(debug=True)