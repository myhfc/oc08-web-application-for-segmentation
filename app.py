import requests
import os
import io
import base64
from PIL import Image
import numpy as np
from flask import (Flask, render_template ,request, send_from_directory)

app = Flask(__name__)

STATIC_FOLDER = 'static/'
MODEL_FOLDER = os.path.join(STATIC_FOLDER, 'models/')
UPLOAD_FOLDER = os.path.join(STATIC_FOLDER,'uploads/')
RESULT_FOLDER = os.path.join(STATIC_FOLDER,'results/')

model_name_dict = {"deeplab_v3plus_augment":"deeplab_v3plus_mobilenetv2_512_augment",
                 "deeplab_v3plus":"deeplab_v3plus_mobilenetv2_512",
                 "unet_xception_augment":"unet_xception_xception_512_augment",
                 "unet_xception":"unet_xception_xception_512",
                }
# Home Page
@app.route('/', methods=['POST','GET'])
def index():
    if request.method=="POST":
        ###
        modelName = request.form["model_name"]
        model_name = model_name_dict[modelName]
        if (model_name.split("_")[-1]=="augment"):
            resize = int(model_name.replace("_augment","").split("_")[-1])
        else :
            resize = int(model_name.split("_")[-1])
        
        ###
        img_file = request.files['image']
        image_data = img_file.read()
        image_resized = Image.open(io.BytesIO(image_data)).resize((resize, resize))
        fullname = os.path.join(UPLOAD_FOLDER, img_file.filename)
        image_resized.save(fullname)
        print(fullname)

        # Define the Flask API endpoint URL
        #url = 'http://localhost:5000/api/predict/'
        url = "https://oc8-segmentation-app.herokuapp.com/api/predict/"

        # Send the POST request with the image array as the request body
        with open(fullname, "rb") as image_file:
            files = {"image": image_file}
            response = requests.post(url, files=files)

        received_file = response.json()
        print( response.status_code)

        # Decode the base64-encoded data to a byte string
        img_bytes = base64.b64decode(received_file["image"])
        processed_image = Image.open(io.BytesIO(img_bytes)).resize((resize,resize))
        result_fname = "result_"+img_file.filename
        fullname_pr = os.path.join(UPLOAD_FOLDER, result_fname)
        processed_image.save(fullname_pr)

        
        return render_template('index.html', predict=True, image_file_name=img_file.filename,
                               result_fname=result_fname, fullname =fullname_pr ) 
    else:
        return render_template('index.html', predict=False)


@app.route('/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/<result_filename>')
def result_file(result_filename):
    return send_from_directory(UPLOAD_FOLDER, result_filename)

"""
@app.route('/<filename>')
def ground_truth_seg_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)
"""

if __name__ == '__main__':
    port = int(os.environ.get('PORT',5000))
    app.run(host="0.0.0.0", port=port, debug=True)
    
