from crypt import methods
from flask import Flask, request, Response
import detect
import base64
from PIL import Image
from io import BytesIO
import cv2
import json
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/detect", methods=['GET', 'POST'])
def yolo_detect():
    if request.method == 'POST':
        print(request)
        img = request.get_json()["img"]
        opt = detect.parse_opt()
        #weights 설정 (default = 'yolov3.pt)
        opt.weights = 'best_0225.pt'
        detect.check_requirements(exclude=('tensorboard', 'thop'))
        #detect
        output_dict = detect.run(**vars(opt), source=img)
        #encode
        output_dict['image'] = base64.b64encode(output_dict['image'])
        output_dict = json.dumps(output_dict)
        res = Response(output_dict)
        res.headers['Content-Type'] = "application/json"
        res.headers['charset'] = 'utf-8'
        return res
    else:
        return 'this api is for detect'

if __name__ == "__main__":
    app.run(debug=True, host="localhost", port=80)