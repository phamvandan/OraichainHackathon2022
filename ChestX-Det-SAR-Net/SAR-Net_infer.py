import cv2
from torchvision import transforms
import torch
import numpy as np 
from collections import OrderedDict
from ptsemseg.pspnet import pspnet

Transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def convert_state_dict(state_dict):
    """Converts a state dict saved from a dataParallel module to normal 
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = ['Left Clavicle', 'Right Clavicle', 'Left Scapula', 'Right Scapula',
    'Left Lung', 'Right Lung', 'Left Hilus Pulmonis', 'Right Hilus Pulmonis',
    'Heart', 'Aorta', 'Facies Diaphragmatica', 'Mediastinum',  'Weasand', 'Spine']
n_classes = len(classes)
model = pspnet(n_classes)
model_path = '../../pspnet_chestxray_best_model_4.pkl'
state = convert_state_dict(torch.load(model_path)["model_state"])
model.load_state_dict(state)
# model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
model.eval()
model.to(device)
print('model loaded!!')

def do_predict(filename):
    img = cv2.imread(filename, 1)
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
    img = Transform(img)
    img = img.unsqueeze(0)
    img = img.to(device)

    outputs = model(img)
    pred = outputs.data.cpu().numpy()
    pred = 1 / (1 + np.exp(-pred))  # sigmoid
    pred[pred < 0.5] = 0
    pred[pred > 0.5] = 1
    return pred

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