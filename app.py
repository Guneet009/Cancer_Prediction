from src.Pipeline.training_pipeline import TrainingPipeline, TrainingPipelineConfig
from src.Pipeline.prediction_pipeline import CancerModel
import sys
from src.utils.main_utils.utils import load_object
from src.Custom_Exception.CustomException import CustomException
from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import threading

app = Flask(__name__)

training_complete = False  # Track training status

def train_model():
    global training_complete
    training_complete = False  # Reset before training starts
    training_pipeline_config = TrainingPipelineConfig()
    train_pipeline = TrainingPipeline(training_pipeline_config)
    train_pipeline.run_pipeline()  # Run your actual training
    training_complete = True  # Mark as completed

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['GET','POST'])
def train():
    if request.method=='GET':
        return render_template('index.html')
    global training_complete
    if not training_complete:
        thread = threading.Thread(target=train_model)
        thread.start()
        return jsonify({"message": "Training started!"})
    else:
        return jsonify({"message": "Training already completed!"})

@app.route('/status', methods=['GET'])
def status():
    return jsonify({"message": "Training Completed" if training_complete else "Training in progress..."})

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method=='GET':
        return render_template('predict.html')
    try:
        preprocesor=load_object("final_model/preprocessor.pkl")
        final_model=load_object("final_model/model.pkl")
        cancer_model = CancerModel(preprocessor=preprocesor,model=final_model)
        features = [
            'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
            'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean',
            'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se',
            'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se',
            'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst',
            'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst',
            'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
        ]

        input_data = [float(request.form[feature]) for feature in features]
        input_array = np.array(input_data).reshape(1, -1)  # Reshape for model

        # Predict using the loaded model
        
        prediction = cancer_model.predict(input_array)
        prediction_text = "Malignant (Cancer Detected)" if prediction[0] == 1 else "Benign (No Cancer Detected)"

        return render_template('predict.html', prediction=prediction_text)
    except Exception as e:
        raise CustomException(e,sys)
    
    
if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000,debug=True)
