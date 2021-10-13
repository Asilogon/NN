from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
import tensorflow as tf
import glob
import numpy as np
import matplotlib.pyplot as plt
import os

from tensorflow.python.ops.gen_array_ops import size



def load_image(filename):
  size = 224
  raw = tf.io.read_file(filename,)
  image = tf.image.decode_png(raw, channels=3)
  img = tf.image.resize(image, (size, size))
  img = img / 255.0
  return img



model = load_model("Neural_network.h5")
os.system('cls' if os.name == 'nt' else 'clear')
way_phote = input("Введите путь к фотографиям: ")
fail = glob.glob(f"{way_phote}\*.jpg")
for i in fail:
    img = load_image(i)
    image = load_img(i)
    img_expended = np.expand_dims(img, axis=0)
    prediction = model.predict(img_expended)[0][0]
    pred_label = 'ЭТО КОТ' if prediction < 0.5 else 'ЭТО СОБАКА'
    plt.figure()
    plt.imshow(image)
    plt.title(f'{pred_label} {prediction}')
    plt.show()
