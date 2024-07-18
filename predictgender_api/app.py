import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import h5py
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from scipy.fft import fft
import os

description = """
EEG Gender Prediction API helps you to predict the gender based on EEG data using a CNN model.

To get the prediction, you need to upload EEG data in a compatible format.

API Endpoints:

## Predictions 
* '/predict_gender': Predict gender based on EEG data
"""

tags_metadata = [
    {
        "name": "Predictions",
        "description": "Endpoints that use our CNN model to predict gender based on EEG data"
    },
    {
        "name": "Default",
        "description": "Default endpoint"
    }
]

app = FastAPI(
    title="EEG Gender Prediction API",
    description=description,
    version="1.0",
    contact={
        "name": "Seddik AMROUN",
    },
    openapi_tags=tags_metadata
)

class EEGData(BaseModel):
    file_path: str

@app.get("/", tags=["Default"])
async def read_root():
    return {"message": "Welcome to the EEG Gender Prediction API. Use /docs to see the API documentation."}

@app.post("/predict_gender", tags=["Predictions"])
async def predict_gender(data: EEGData):
    """
    Predict gender based on EEG data. Endpoint will return a dictionary like this:
    '''
    {'prediction': gender}
    '''
    You need to provide the file path to the EEG data.
    """
    
    dataX = load_data('C:/Users/seddi/Desktop/Formation Jedha/projet_predict_sex_from_brain_rhythms/X_train_new.h5')
    predictions = predict_sexe(dataX)
    results = ['Male' if predict == 0 else 'Female' for predict in predictions]
    
    response = {"prediction": results}
    return response

def load_data(file):
    file_extension = os.path.splitext(file)[1]
    if file_extension == '.h5':
        with h5py.File(file, 'r') as f:
            data = f.get('features')
            data = np.array(data)
            return data
    elif file_extension == '.csv':
        data = pd.read_csv(file)
        return data.values

def apply_fft(data):
    transformed_data = np.zeros_like(data, dtype=complex)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(data.shape[2]):
                transformed_data[i, j, k, :] = fft(data[i, j, k, :])
    return transformed_data

def hjorth_parameters(data):
    first_deriv = np.diff(data)
    second_deriv = np.diff(first_deriv)
    
    activity = np.var(data)
    mobility = np.sqrt(np.var(first_deriv) / activity)
    complexity = np.sqrt(np.var(second_deriv) / np.var(first_deriv)) / mobility
    
    return activity, mobility, complexity

def spectral_entropy(data, sampling_rate):
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
                hjorth_params = hjorth_parameters(channel)
                entropy = spectral_entropy(channel, sampling_rate)
                sample_features.extend(hjorth_params)
                sample_features.append(entropy)
        features.append(sample_features)
    return np.array(features)

def predict_sexe(data):
    model_path = 'C:/Users/seddi/Desktop/Formation Jedha/projet_predict_sex_from_brain_rhythms/model_cnn.h5'
    model = load_model(model_path)
    data = apply_fft(data)
    data = extract_features(data.real)
    data = data.reshape(data.shape[0], 40, 7, -1)
    predict = model.predict(data)
    return np.argmax(predict, axis=1)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8502)
