import detect as detect
import base64
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import json


#테스트할 이미지의 파일 경로 설정
filepath = './data/images/foods.jpg'
with open(filepath, 'rb') as img:
    base64_string = base64.b64encode(img.read())

#Model Detect.py RUN
opt = detect.parse_opt()
#weights 설정 (default = 'yolov3.pt)
opt.weights = './custom_weights/best_lr_0.002.pt'
detect.check_requirements(exclude=('tensorboard', 'thop'))
output_dict = detect.run(**vars(opt), source=base64_string)

#image, bbox, class 확인 (image는 일부만 확인)
print(f"image : {output_dict['image'][0]} bbox : {output_dict['bbox']}, class : {output_dict['class']}")

#Save output image (bbox 그려져있는 image)
output_image = output_dict['image']
output_img_name = filepath.split('/')[-1][:-4]
cv2.imwrite(f'./test_result/test_result_{output_img_name}.jpg', output_image)

print('-------------end-------------')