from flask import Flask, request, Response
import detect_y5 as detect
import base64
from PIL import Image
from flask_cors import CORS
import cv2
import json

app = Flask(__name__)
CORS(app)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/detect", methods=['GET', 'POST'])
def yolo_detect():
    #데이터 받기
    if request.method == 'POST':
        new_img = request.get_json()["img"]
        source_img = new_img.encode('utf-8')
    else:
        path = "./미역국.jpeg"
        with open(path, 'rb') as img:
            source_img = base64.b64encode(img.read())
    
    #모델에 통과시키기
    opt = detect.parse_opt()
    opt.device = 'cpu'
    opt.weights = '/home/team07/yolov5/runs/train/batch_16_0.004_epoch_50_v5x6/weights/best.pt'
    detect.check_requirements(exclude=('tensorboard', 'thop'))
    output_dict = detect.run(**vars(opt), source=source_img)

    #전송 전 이미지 전처리
    temp_img = cv2.cvtColor(output_dict["image"], cv2.COLOR_BGR2RGB)
    temp_img = Image.fromarray(temp_img)
    temp_img.save('flask_result.jpg')
    with open('flask_result.jpg', 'rb') as img:
        new_img = base64.b64encode(img.read()).decode('utf-8')

    #json형식으로 변환
    output_dict["image"] = new_img
    img_dict = json.dumps(output_dict)

    #response 설정
    res = Response(img_dict)
    res.headers['Content-Type'] = "application/json"
    res.headers['charset'] = 'utf-8'

    return res

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
