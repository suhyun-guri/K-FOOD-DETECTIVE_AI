import detect_y5 as detect
import base64
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import json
from pathlib import Path
from utils.general import print_args

FILE = Path(__file__).resolve()

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#테스트할 이미지의 파일 경로 설정
filepath = '/project/guri/Elice/yolov5/yolov5/data/images/2337D43D50FA4B4C1D.jpeg'
with open(filepath, 'rb') as img:
    base64_string = base64.b64encode(img.read())
    temp = Image.open(img)
    ori_image = cv2.cvtColor(np.array(temp), cv2.COLOR_BGR2RGB)

#Model Detect.py RUN
opt = detect.parse_opt()
opt.hide_conf = True
#cpu설정
# opt.device = 'cpu'
#weights 설정 (default = 'yolov3.pt)
opt.weights = '/project/guri/Elice/yolov5/yolov5/runs/train/batch_16_0.004_epoch_50_v5x6/weights/best.pt'
# opt.weights = '/project/guri/Elice/yolov5/yolov5/runs/train/batch_8_0.002_epoch_100_v5x6/weights/best.pt'
# opt.weights = './custom_weights/best copy.pt'
detect.check_requirements(exclude=('tensorboard', 'thop'))
print_args(FILE.stem, opt)
output_dict = detect.run(**vars(opt), source=base64_string)
#image, bbox, class 확인 (image는 일부만 확인)
print(f"image : {output_dict['image'][0]} bbox : {output_dict['bbox']}, class : {output_dict['class']}")

#Save output image (bbox 그려져있는 image)
output_image = output_dict['image']
print(ori_image)
# print(output_image.shape)
output_img_name = filepath.split('/')[-1][:-4]
cv2.imwrite(f'./test_result/{output_img_name}.jpg', output_image)

with open("class.json", "r", encoding="UTF-8") as json_file:
    class_dict = json.load(json_file)
    reverse_dict= dict(map(reversed,class_dict.items()))

output_dict1 = {}
output_dict1['class'] = output_dict['class']
from utils.plots import Annotator, colors
for i, bbox in enumerate(output_dict['bbox']):
    temp_img = ori_image.copy()
    label = output_dict['class'][i]
    print(bbox, label)
    if ori_image.shape[0] >= 1000:
        line_thickness = 10
    else:
        line_thickness = 3
    c = reverse_dict[label]
    annotator = Annotator(temp_img, line_width=line_thickness, example=str(1))
    annotator.box_label(bbox, label, color=colors(c, True))
    im0 = annotator.result()
    output_dict1[f'image{i+1}'] = im0
    cv2.imwrite(f'./test_result/{output_img_name}-{label}.jpg', im0)
print(output_dict1)
print('-------------end-------------')