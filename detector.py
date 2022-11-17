#import required libraries 
import numpy as np
from tensorflow import keras


model_1 = keras.models.load_model('emotion_detection_model.h5')
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
def get_result(img):

    pred = model_1.predict(img)
    fin = np.argmax(pred)
    label = emotions[fin]
    return label
