# from flask import Flask, request, jsonify
# import tensorflow as tf
# import numpy as np

# app = Flask(__name__)

# # Load the model
# model = tf.keras.models.load_model('walking_classifier_tf.h5')

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.get_json(force=True)
#     predictions = model.predict(np.array(data['features']))
#     return jsonify({'predictions': predictions.tolist()})

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)


import os
from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib

app = Flask(__name__)

# Load the saved scaler and model
scaler = StandardScaler()
scaler.mean_ = np.loadtxt('scaler_mean.csv', delimiter=',')
scaler.scale_ = np.loadtxt('scaler_std.csv', delimiter=',')

# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="walking_classifier.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the request
        input_data = request.json['data']

        # Convert to numpy array and reshape
        input_data = np.array(input_data, dtype=np.float32)

        # Normalize the data
        input_data = scaler.transform(input_data)

        # Check the input shape
        if input_data.shape[1] != input_details[0]['shape'][1]:
            return jsonify({"error": "Input shape does not match model expected shape."})

        # Prepare the prediction list
        predictions = []

        # Run the model for each input data
        for i in range(input_data.shape[0]):
            single_input_data = input_data[i].reshape(1, -1)
            interpreter.set_tensor(input_details[0]['index'], single_input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])[0]
            predictions.append(float(output_data))

        # Convert to binary labels
        threshold = 0.5
        predicted_labels = (np.array(predictions) > threshold).astype(int).tolist()

        # Return the predictions as JSON
        return jsonify({"predictions": predicted_labels})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
