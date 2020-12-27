from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import argparse
import io
import time
import numpy as np
import picamera
import pymongo
import datetime


from pymongo import MongoClient
from datetime import datetime



from PIL import Image
from tflite_runtime.interpreter import Interpreter


def load_labels(path):
  with open(path, 'r') as f:
    return {i: line.strip() for i, line in enumerate(f.readlines())}


def set_input_tensor(interpreter, image):
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image


def classify_image(interpreter, image, top_k=1):
  """Returns a sorted array of classification results."""
  set_input_tensor(interpreter, image)
  interpreter.invoke()
  output_details = interpreter.get_output_details()[0]
  output = np.squeeze(interpreter.get_tensor(output_details['index']))

  # If the model is quantized (uint8 data), then dequantize the results
  if output_details['dtype'] == np.uint8:
    scale, zero_point = output_details['quantization']
    output = scale * (output - zero_point)

  ordered = np.argpartition(-output, top_k)
  return [(i, output[i]) for i in ordered[:top_k]]


def main():
 # parser = argparse.ArgumentParser(
     # formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  #parser.add_argument(
     #'--model', help='/home/pi/Desktop/mobilenet_v1_1.0_224_quant.tflite', required=True)
  #parser.add_argument(
     # '--labels', help='/home/pi/Desktop/labels_mobilenet_quant_v1_224.txt', required=True)
 # args = parser.parse_args()

  labels = load_labels('/home/pi/Desktop/classes.txt')

  interpreter = Interpreter('/home/pi/Desktop/converted_model_khawla2.tflite')
  interpreter.allocate_tensors()
  _, height, width, _ = interpreter.get_input_details()[0]['shape']

  with picamera.PiCamera(resolution=(640, 480), framerate=30) as camera:
    camera.start_preview()
    try:
      stream = io.BytesIO()
      for _ in camera.capture_continuous(
          stream, format='jpeg', use_video_port=False):
        stream.seek(0)
        image = Image.open(stream).convert('RGB').resize((224,224))
        
        
        start_time = time.time()
        results = classify_image(interpreter, image)
        elapsed_ms = (time.time() - start_time) * 1000
        
        
        x=datetime.now()
        x_str=x.strftime("%d/%m/%Y %H:%M:%S")
        print(x_str)
   
        result=results[0][0]
        
        data=[]
      
        if result == 0 :
            data.append({"food":'Bread', "Date": x_str})
        elif result== 1 :
            data.append({"food":'Dairy product',"Date": x_str})
                            
        elif result== 2 :
            data.append({"food":'Dessert',"Date": x_str})
            
        elif result== 3 :
            data.append({"food":'Egg',"Date": x_str})
        elif result== 4 :
            data.append({"food":'Fried_Food',"Date": x_str})
        elif result== 5 :
            data.append({"food":'Meat',"Date": x_str})
        elif result== 6 :
            data.append({"food":'Noodles-Pasta',"Date": x_str})
        elif result== 7 :
            data.append({"food":'Rice',"Date": x_str})
        elif result== 8 :
            data.append({"food":"Sea-food","Date": x_str})
        elif result== 9 :
            data.append({"food":"Soup","Date": x_str})
        
        else:
                        
            data.append({"food":"Vegetable-Fruit","Date": x_str})
            
            
        
                         
                               
        print(data)          
            


       
        try:
            client = MongoClient()
            client = MongoClient('localhost', 27017) #établir la connection avec le serveur mongodb 
            print("successfully connection with mongo")
        except:
            print("no connection with mongo ")
            
        db = client.pymongo_test
        collection=db.food #specifier le nom de la base de données que je vais utiliser
        data= collection.insert_many(data) # inserer les données
        
        
        
        
        stream.seek(0)
        stream.truncate()
        camera.annotate_text = '%s' % (result)
        time.sleep(10)
        camera.stop_preview()

        
    finally:
      
      camera.close()
      
      


if __name__ == '__main__':
  main()
  
