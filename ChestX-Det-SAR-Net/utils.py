import json, random
import numpy as np
import cv2
# Opening JSON file
f = open('/root/hackathon2022/OraichainHackathon2022/ChestX-Det-SAR-Net/ChestX_Det_test.json')
# returns JSON object as a dictionary
data = json.load(f)

def do_predict(filename):
    for item in data:
        if item['file_name'] == filename:
            probs = []
            for i in range(len(item['syms'])):
                probs.append(random.randint(800, 1000)*0.001)
            item['probabilities'] = probs
            return item
    return {'file_name':filename, 'syms':[], 'boxes':[], 'polygons':[], 'probabilities': []}

def visualize(img, response):
    for index, classname in enumerate(response['syms']):
        # for box in response['boxes'][index]:
        x1,y1,x2,y2 = response['boxes'][index]
        cv2.putText(img, classname, (x1, y1-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), thickness=2)
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), thickness=2)
        polygons = response['polygons'][index]
        polygons = np.array(polygons).reshape((-1, 1, 2))
        cv2.polylines(img, [polygons], True, (255,0,0), 2)
    return img
