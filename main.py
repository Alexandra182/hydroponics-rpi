import subprocess
import serial, time, random

# InfluxDB
from influxdb import InfluxDBClient
client = InfluxDBClient(host='<host_name>', port=<port>)
client.switch_database('<database_name>')

json_body1 = [
  {
    "measurement": "hydroponics-strawberries",
    "fields": {
      "value": 1
    }
  }
]

json_body2 = [
  {
    "measurement": "hydroponics-strawberries",
    "fields": {
      "value": 0
    }
  }
]

# Serial communication with Arduino
ser = serial.Serial('/dev/serial0', 115200, timeout=1)
ser.flush()

# Tensorflow + Keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import json
from os.path import join
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np

# Delete the previous photo and capture a new one 
subprocess.run(["rm", "./photos/strawberry.jpg"])
subprocess.run(["raspistill", "-o", "./photos/strawberry.jpg"])

ser.write(b"S\n")
line = ser.readline().decode('utf-8').rstrip()
dataset = line.split("-")
print(dataset)
if len(dataset) == 5:
  ph = float(dataset[0])
  tds = int(dataset[1])
  temp = int(dataset[2])
  humidity = int(dataset[3])
  waterlvl = int(dataset[4])

  print("ph: ",ph," tds: ",tds," temp: ",temp," humidity: ",humidity," waterlvl: ", waterlvl)

  json_body = [
    {
      "measurement": "hydroponics-ph",
      "fields": {
        "value": ph
      }
    },
    {
      "measurement": "hydroponics-tds",
      "fields": {
        "value": tds
      }
    },
    {
      "measurement": "hydroponics-temp",
      "fields": {
        "value": temp
      }
    },
    {
      "measurement": "hydroponics-humidity",
      "fields": {
        "value": humidity
      }
    },
    {
      "measurement": "hydroponics-waterlvl",
      "fields": {
        "value": waterlvl
      }
    }
  ]
  client.write_points(json_body)

image_dir = './photos/'
filename = 'strawberry.jpg'
img_path = join(image_dir, filename)
image_size = 224

def read_and_prep_image(img_path, img_height = image_size, img_width = image_size):
    img = load_img(img_path, target_size = (img_height, img_width))
    img_array = np.array([img_to_array(img)])
    output = preprocess_input(img_array)
    return(output)

# Predict strawberry fruit

# Source: https://github.com/Kaggle/learntools/blob/master/learntools/deep_learning/decode_predictions.py
def decode_predictions(preds, top, class_list_path):
  if len(preds.shape) != 2 or preds.shape[1] != 1000:
    raise ValueError('`decode_predictions` expects '
                     'a batch of predictions '
                     '(i.e. a 2D array of shape (samples, 1000)). '
                     'Found array with shape: ' + str(preds.shape))
  CLASS_INDEX = json.load(open(class_list_path))
  results = []
  for pred in preds:
    top_indices = pred.argsort()[-top:][::-1]
    result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
    result.sort(key = lambda x: x[2], reverse = True)
    results.append(result)
  return results

my_model = ResNet50(weights = './resnet50/resnet50_weights_tf_dim_ordering_tf_kernels.h5')
test_data = read_and_prep_image(img_path)
preds = my_model.predict(test_data)

most_likely_labels = decode_predictions(preds, top = 3, class_list_path = './resnet50/imagenet_class_index.json')
label = most_likely_labels[0][0][1]

# Predict health
classifier = load_model('./model/model80.h5')

def predict_health(image_src):
    test_image = image.load_img(image_src, target_size = (200, 200))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = classifier.predict(test_image)
    print('Probabilities:', result[0])

    index = np.argmax(result[0])
    return index

label2 = predict_health(img_path)

# Send data to InfluxDB
if label == 'strawberry':
  print(most_likely_labels[0][0])
  print('strawberry')
  client.write_points(json_body1)

if label2 == 1:
  print('Not Healthy')
  client.write_points(json_body2)

client.write_points(json_body1)
client.write_points(json_body2)

