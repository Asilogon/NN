import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout


SIZE = 224

def load_image(filename):
  raw = tf.io.read_file(filename,)
  image = tf.image.decode_png(raw, channels=3)
  img = tf.image.resize(image, (SIZE, SIZE))
  img = img / 255.0
  return img

def resize_image(img, label):   # Подготовка фотографий для нейросети
  img = tf.cast(img, tf.float32)
  img = tf.image.resize(img, (SIZE, SIZE))
  img = img / 255.0
  return img, label



train, _ = tfds.load('cats_vs_dogs', split=['train[:100%]'], with_info=True, as_supervised=True) # Загрузка dataset фотографий кошек и собак

train_resized = train[0].map(resize_image)            # Подготовка фотографий в dataset к обучению нейросетью 
train_batches = train_resized.shuffle(1000).batch(16)

base_layers = tf.keras.applications.MobileNetV2(input_shape=(SIZE, SIZE, 3), include_top=False) # Скачивание подготовленной нейросети
base_layers.trainable = False # Не обучаем подготовленную нейросеть


model = tf.keras.Sequential([                               # Параметри нейросети
                             base_layers,
                             GlobalAveragePooling2D(),
                             Dropout(0.2),
                             Dense(1)
])
model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy']) #компеляция нейронки
model.fit(train_batches, epochs=1) # обучение нейросети
#model.save("Neural_network.h5") #Сохраняет нейросеть
