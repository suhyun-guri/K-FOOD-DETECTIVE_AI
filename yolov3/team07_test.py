import detect as detect
import base64
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import json


#테스트할 이미지의 파일 경로
filepath = './data/images/삼계탕.jpg'
with open(filepath, 'rb') as img:
    base64_string = base64.b64encode(img.read())

#run
opt = detect.parse_opt()
#weights 설정 (default = 'yolov3.pt)
opt.weights = 'best_0225.pt'
opt.save_txt = True
detect.check_requirements(exclude=('tensorboard', 'thop'))
#detect
output_dict = detect.run(**vars(opt), source=base64_string)

print(output_dict['bbox'], output_dict['class'])

output_image = output_dict['image']

img = np.array(output_image)
cv2.imwrite('./team07_test_result.jpg', img)

print('-------------end-------------')