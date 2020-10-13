import utils.visualization_utils as viz_utils
import matplotlib.pyplot as plt
import utils.label_map_util
import tensorflow as tf
import numpy as np
import requests
import tempfile
import time
import json
import os

from flask import Flask, render_template, request, redirect, jsonify, flash
from werkzeug.utils import secure_filename
from hashlib import sha256
from io import BytesIO
from PIL import Image

app = Flask(__name__)
app.secret_key = b'_5$GFS#y2L"**&^*&FR%&#^F4Q8z\n\xec]/'

category_index = {  1: {'id': 1, 'name': 'rider'},
                    2: {'id': 2, 'name': 'bicycle'},
                    3: {'id': 3, 'name': 'other_person'},
                    4: {'id': 4, 'name': 'trailer'},
                    5: {'id': 5, 'name': 'traffic_light_r'},
                    6: {'id': 6, 'name': 'traffic_light_g'},
                    7: {'id': 7, 'name': 'bus'},
                    8: {'id': 8, 'name': 'motorcycle'},
                    9: {'id': 9, 'name': 'traffic_sign'},
                    10: {'id': 10, 'name': 'traffic_light'},
                    11: {'id': 11, 'name': 'truck'},
                    12: {'id': 12, 'name': 'other_vehicle'},
                    13: {'id': 13, 'name': 'train'},
                    14: {'id': 14, 'name': 'pedestrian'},
                    15: {'id': 15, 'name': 'traffic_light_y'},
                    16: {'id': 16, 'name': 'car'}
                }


ALLOWED_EXTENSIONS = ['jpeg', 'png', 'jpg']
tf.keras.backend.clear_session()
detect_fn = tf.saved_model.load('./faster_rcnn_trained_model/saved_model/')

@app.route('/')
def home():
    return redirect('/upload')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    image_found = False
    if request.method == 'POST' and request.files['photo']:
        image = Image.open(request.files['photo'])
        extension = get_image_extension(request.files['photo'])
        print("IMAGE_STUFF: ",request.files['photo'])

        if extension not in ALLOWED_EXTENSIONS:
            flash(f"This image extension (.{extension}) is not supported. Upload {' '.join(ALLOWED_EXTENSIONS)} only.", "error")
            print(f"ERROR in upload(): The image extension is not supported.")
            return redirect('/')
        image_found = True

    elif request.method == 'POST' and request.form['text']:
        link = request.form['text']
        extension = link.split('.')[-1]

        if(extension not in ALLOWED_EXTENSIONS):
            flash(f"Image URL must be of .png, .jpg or .jpeg extension.", "error")
            print(f"ERROR in upload(): The image URL extension is not supported.")
            return redirect('/')
    
        response = requests.get(link)
        image = Image.open(BytesIO(response.content))
        image_found = True
    
    user_image = None
    if image_found:
        user_image = detect_boxes(image)

    return render_template('index.html', user_image=user_image)

def get_image_extension(image):
    return secure_filename(image.filename).rsplit('.')[-1]

def detect_boxes(image):
    (im_width, im_height) = image.size
    image_np = np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

    input_tensor = np.expand_dims(image_np, 0)
    start_time = time.time()
    detections = detect_fn(input_tensor)
    end_time = time.time()
    print("Prediction Time: ", end_time - start_time)
    plt.rcParams['figure.figsize'] = [42, 21]
    image_np_with_detections = image_np.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'][0].numpy(),
        detections['detection_classes'][0].numpy().astype(np.int32),
        detections['detection_scores'][0].numpy(),
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.40,
        agnostic_mode=False)

    plt.figure(figsize=(24,30))
    img = Image.fromarray(image_np_with_detections , 'RGB')
    image_name = f"{tempfile.NamedTemporaryFile().name}.jpg".split('/')[-1]
    image_path = f'./static/uploads/{image_name}'
    print("IMAGE_NAME: ", image_name)
    print("IMAGE_PATH: ", image_path)
    img.save(image_path)
    return image_path


if __name__ == '__main__':
    app.run()

