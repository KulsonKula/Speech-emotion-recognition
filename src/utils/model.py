import tensorflow as tf
import os
import numpy as np

def load_model(model_filename):
    model_path = os.path.join("model", model_filename)
    model = tf.keras.models.load_model(model_path)
    return model

def predict_emotion(model, features):
    emotion = {
        1:"neutral",
        2:"calm",
        3:"happy",
        4:"sad",
        5:"angry",
        6:"fearful",
        7:"disgust",
        8:"surprised"
    }
    predict = model.predict(features)
    idx = np.argmax(predict)

    return emotion[idx]
  