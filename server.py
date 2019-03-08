import os
import numpy as np
from PIL import Image
from feature_extractor import FeatureExtractor
import glob
import pickle
from datetime import datetime
from flask import Flask, request, render_template, jsonify, send_file
import codecs, json 

app = Flask(__name__)

# Read image features
fe = FeatureExtractor()
features = []
img_paths = []
for feature_path in glob.glob("static/feature/*"):
    features.append(pickle.load(open(feature_path, 'rb')))
    img_paths.append('static/img/' + os.path.splitext(os.path.basename(feature_path))[0] + '.jpg')


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['query_img']

        img = Image.open(file.stream)  # PIL image
        uploaded_img_path = "static/uploaded/" + datetime.now().isoformat() + "_" + file.filename
        img.save(uploaded_img_path)

        query = fe.extract(img)
        dists = np.linalg.norm(features - query, axis=1)  # Do search
        ids = np.argsort(dists)[:8] # Top 8 results

        for id in ids:
            if (0 <= dists[id] <= 1):
                scores = [(dists[id], img_paths[id]) for id in ids]

        return render_template('index.html', query_path=uploaded_img_path, scores=scores)
        #a =np.asarray(scores)
        #b=a.tolist()
        #return json.dumps(b, separators=(',', ':'), sort_keys=True, indent=4)
    else:
        return render_template('index.html')


@app.route('/search', methods=['POST'])
def meth():
    if 'file' not in request.files:
        return "No file found"
    file = request.files['file']
    #file.save("static/test.jpg")
    #return "file successfully saved"
    img = Image.open(file.stream)  # PIL image
    uploaded_img_path = "static/uploaded/" + datetime.now().isoformat() + "_" + file.filename
    img.save(uploaded_img_path)

    query = fe.extract(img)
    dists = np.linalg.norm(features - query, axis=1)  # Do search
    ids = np.argsort(dists)[:8] # Top 8 results

    for id in ids:
        if (0 <= dists[id] <= 1):
            scores = [(dists[id], img_paths[id]) for id in ids]
    
    a =np.asarray(scores)
    b=a.tolist()
    return json.dumps(b, separators=(',', ':'), sort_keys=True, indent=4)


@app.route('/img/<string:ip>')
def img(ip):
    image = "static/img/"+ip
    #try:
    reqimage = Image.open(image)
    #except IOError:
     #   pass
    reqimage.show()
    return "success"
    

 
if __name__=="__main__":
#     app.run("0.0.0.0", 5000)
    app.run()
