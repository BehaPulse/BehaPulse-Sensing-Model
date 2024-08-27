"""
    Inference.py
"""
from modules.utils import load_yaml
from models.utils import get_model

from datetime import datetime, timezone, timedelta
from tqdm import tqdm
import numpy as np
import pandas as pd
import random, os, torch
import requests
import torch.nn.functional as F

import time

# Config
PROJECT_DIR = os.path.dirname(__file__)
predict_config = load_yaml(os.path.join(PROJECT_DIR, 'config', 'inference_config.yaml'))

# Serial
train_serial = predict_config['TRAIN']['train_serial']

# Recorder Directory
RECORDER_DIR = os.path.join(PROJECT_DIR, 'results', 'train', train_serial)

# Train config
train_config = load_yaml(os.path.join(RECORDER_DIR, 'train_config.yml'))

# SEED
torch.manual_seed(predict_config['PREDICT']['seed'])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(predict_config['PREDICT']['seed'])
random.seed(predict_config['PREDICT']['seed'])

model_name = train_config['TRAINER']['model']

label_dict = train_config['label_map']
label_dict_ko = predict_config['label_map_ko']
# Gpu
os.environ['CUDA_VISIBLE_DEVICES'] = str(predict_config['PREDICT']['gpu'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__' :
    
    print('Load Model...')
    model_name = train_config['TRAINER']['model']
    model_args = train_config['MODEL'][model_name]
    
    model = get_model(model_name=model_name, model_args=model_args).to(device)
    
    checkpoint = torch.load(os.path.join(RECORDER_DIR, 'model.pt'))
    model.load_state_dict(checkpoint['model'])
    model.eval()

    print('Finish Load Model...')
    while (True) : 
        try :
            mac_address = train_config['DIRECTORY']['mac_address']
            
            result_dict = {}
            
            for mac in mac_address:
                # print(f'Predicting {mac}...')
                api_url = f"{predict_config['BehaPulse']['Base_url']}:{predict_config['BehaPulse']['Port']}/device/CSI/{mac}"
                response = requests.get(api_url)
                if response.status_code == 200:
                    
                    # 2차원 데이터가 return 됨
                    csi_data = response.json()
                    csi_data = csi_data[0:train_config['DATASET']['window_size']]
                    
                    csi_data = torch.tensor(csi_data).unsqueeze(0).unsqueeze(0).to(device)
                    pred = model(csi_data)
                    
                    outputs = F.log_softmax(pred, dim=1)
                    
                    # MAC 주소별로 결과 합산
                    if mac not in result_dict:
                        result_dict[mac] = outputs
                    else:
                        result_dict[mac] += outputs
                
            # 여러 MAC 주소의 결과 합산
            total_output = sum(result_dict.values())

            # 최종 예측
            _, predicted_label = torch.max(total_output, dim=1)
            
            # 라벨을 문자열로 변환
            predicted_label_str = list(label_dict.keys())[list(label_dict.values()).index(predicted_label.item())]
            
            # print(f"Final predicted label: {predicted_label_str}")
            
            # mac address를 기반으로 personId 가져오기
            api_url = f"{predict_config['BehaPulse']['Base_url']}:{predict_config['BehaPulse']['Port']}/device/{mac_address[0]}"
            response = requests.get(api_url)
            deviceId = response.json()['device']['deviceId']
            
            api_url = f"{predict_config['BehaPulse']['Base_url']}:{predict_config['BehaPulse']['Port']}/user_device/{deviceId}"
            response = requests.get(api_url)
            user_email = response.json()['user_device'][1]
            
            api_url = f"{predict_config['BehaPulse']['Base_url']}:{predict_config['BehaPulse']['Port']}/user_dashboard_device/user_dashboard_devices/person/{user_email}/{deviceId}"
            response = requests.get(api_url)
            personId = response.json()['user_dashboard_device'][2]
            
            predicted_label_ko = label_dict_ko.get(predicted_label_str, "")

                
            print(f"상태 : {predicted_label_ko}")
            
            api_url = f"{predict_config['BehaPulse']['Base_url']}:{predict_config['BehaPulse']['Port']}/dashboard/update/state/{personId}"
            response = requests.put(api_url, json={"status":predicted_label_ko})
            
        except Exception as e:
            pass
        
        finally :
            # 200ms 대기
            time.sleep(0.2)