import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import numpy as np
from keras.applications import imagenet_utils

loaded_model = keras.models.load_model("mobilenettools.keras")

def prepare_image(file): 
    img_path = 'deneme/'
    img = image.load_img(img_path + file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

preprocessed_image = prepare_image('12.jpg')
predictions = loaded_model.predict(preprocessed_image)
#results = imagenet_utils.decode_predictions(predictions)
index = np.argmax(predictions)

labels = ["1- Akıllı Anahtar", "2- Çift Taraflı İngiliz Anahtarı", "3- Çift Taraflı Yıldız Anahtar", "4- Cırcır Anahtarı", "5- Fort Pense", "6- Pense", "7- Tornavida", "8- Yankeski"]

print(labels[index])