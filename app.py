from src.Pipeline.training_pipeline import TrainingPipeline, TrainingPipelineConfig
from flask import Flask, jsonify
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

@app.route('/train', methods=['POST'])
def train():
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

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000,debug=True)
