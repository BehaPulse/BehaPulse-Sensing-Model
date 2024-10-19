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
import warnings
warnings.filterwarnings('ignore')

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
device_cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__' :
    
    print('Load Model...')
    model_name = train_config['TRAINER']['model']
    model_args = train_config['MODEL'][model_name]
    
    model = get_model(model_name=model_name, model_args=model_args).to(device_cuda)
    
    checkpoint = torch.load(os.path.join(RECORDER_DIR, 'model.pt'))
    model.load_state_dict(checkpoint['model'])
    model.eval()

    print('Finish Load Model...')
    while (True) : 
        try :
            mac_address = train_config['DIRECTORY']['mac_address']
            
            result_dict = {}
            
            for mac in mac_address:
                print(f'Predicting {mac}...')
                api_url = f"{predict_config['BehaPulse']['protocol']}://{predict_config['BehaPulse']['host']}:{predict_config['BehaPulse']['port']}/device/CSI/{mac}"
                response = requests.get(api_url, verify=False)
                if response.status_code == 200:
                    
                    # 2차원 데이터가 return 됨
                    csi_data = response.json()
                    csi_data = csi_data[0:train_config['DATASET']['window_size']]
                    
                    print(f"Device Cuda: {device_cuda}")
                    csi_data = torch.tensor(csi_data).unsqueeze(0).unsqueeze(0).to(device_cuda)
                    pred = model(csi_data)
                    
                    outputs = F.log_softmax(pred, dim=1)
                    print(f'Output : {outputs}')
                    
                    # MAC 주소별로 결과 합산
                    if mac not in result_dict:
                        result_dict[mac] = outputs
                    else:
                        result_dict[mac] += outputs            
            
            # mac address를 기반으로 personId 가져오기
            api_url = f"{predict_config['BehaPulse']['protocol']}://{predict_config['BehaPulse']['host']}:{predict_config['BehaPulse']['port']}/device/{mac_address[0]}"
            response = requests.get(api_url, verify=False)
            deviceId = response.json()['device']['deviceId']
            
            api_url = f"{predict_config['BehaPulse']['protocol']}://{predict_config['BehaPulse']['host']}:{predict_config['BehaPulse']['port']}/user_device/{deviceId}"
            response = requests.get(api_url, verify=False)
            user_email = response.json()['user_device'][1]
            
            api_url = f"{predict_config['BehaPulse']['protocol']}://{predict_config['BehaPulse']['host']}:{predict_config['BehaPulse']['port']}/user_dashboard_device/user_dashboard_devices/person/{user_email}/{deviceId}"
            response = requests.get(api_url, verify=False)
            personId = response.json()['user_dashboard_device'][2]
            
            # 사용자의 활동별 weight 가져오기
            api_url = f"{predict_config['BehaPulse']['protocol']}://{predict_config['BehaPulse']['host']}:{predict_config['BehaPulse']['port']}/sensitivity/{user_email}"
            response = requests.get(api_url, verify=False)
            sensitivity = response.json().get("sensitivityList", [])
            
            # Create a weight dictionary from sensitivity list
            weight_dict = {item['targetStatus']: item['weight'] for item in sensitivity}
            print(f'Weight Dict : {weight_dict}')
            
            # 여러 MAC 주소의 결과 합산
            total_output = sum(result_dict.values())
            print(f'Total Output : {total_output}')
            
            # # Aggregate results from multiple MAC addresses
            # total_output = sum(result_dict.values())
            
            # Apply the weights to the total_output
            weighted_output = total_output.clone()
            for label, idx in label_dict.items():
                target_status_ko = label_dict_ko.get(label, "")
                weight = weight_dict.get(target_status_ko, 1)  # Default weight is 1 if not specified
                weighted_output[0, idx] += np.log(weight)
            
            print(f'Weighted Output : {weighted_output}')
            # 최종 예측
            
            # Final prediction using the weighted output
            _, predicted_label = torch.max(weighted_output, dim=1)         
            print(f"Final predicted label: {predicted_label}")
            
            # 라벨을 문자열로 변환
            predicted_label_str = list(label_dict.keys())[list(label_dict.values()).index(predicted_label.item())]
            
            print(f"Final predicted label: {predicted_label_str}")
            
            
            
            predicted_label_ko = label_dict_ko.get(predicted_label_str, "")

                
            print(f"상태 : {predicted_label_ko}")
            
            api_url = f"{predict_config['BehaPulse']['protocol']}://{predict_config['BehaPulse']['host']}:{predict_config['BehaPulse']['port']}/dashboard/update/state/{personId}"
            response = requests.put(api_url, json={"status":predicted_label_ko}, verify=False)
            
            print("상태 업데이트 결과:", response.status_code)
            
            data = {
                "personId": personId,
                "inferenceTime": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "inferencedStatus": predicted_label_ko
            }
            api_url = f"{predict_config['BehaPulse']['protocol']}://{predict_config['BehaPulse']['host']}:{predict_config['BehaPulse']['port']}/state_inference/register"
            response = requests.post(api_url, json=data, verify=False)
            print("LOG 등록 결과:", response.status_code)
            
            ### Smart Bulb start

            import colorsys

            def hex_to_hue_saturation(hex_color):
                # Remove the hash symbol if present
                hex_color = hex_color.lstrip('#')

                # Convert hex to RGB
                r, g, b = tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))

                # Convert RGB to HSV
                hsv = colorsys.rgb_to_hsv(r, g, b)

                hue = hsv[0] * 100  # Scale hue to 0-100 range
                saturation = hsv[1] * 100  # Scale saturation to 0-100 range

                return int(hue), int(saturation)

            # Get SmartThings Token from Flask
            api_url = f"{predict_config['BehaPulse']['protocol']}://{predict_config['BehaPulse']['host']}:{predict_config['BehaPulse']['port']}/user/st_token/{user_email}"

            response = requests.get(api_url, verify=False)
            print(response.json())

            if response.status_code == 200:
                res = response.json()['user']
                access_token = res['stAccessToken']
                refresh_token = res['stRefreshToken']

                # Get DeviceId of bulb from access token
                url = 'https://api.smartthings.com/v1/devices'
                headers = {
                    'Authorization': f'Bearer {access_token}',
                    'Content-Type': 'application/json'
                }
                
                response = requests.get(url, headers=headers)
                print(response.status_code)
                # print(response.content)
                bulb_device_id = None
                if response.status_code == 200:
                    devices = response.json()
                    for device in devices['items']:
                        for i in device['components']:
                            for j in i['categories']:
                                if 'Light' in j['name']:
                                    print(device['deviceId'], device['name'], j['name'])
                                    bulb_device_id = device['deviceId']
                        # print(device['components'][0]['categories'][0]['name'])
                    
                if bulb_device_id:

                    # 상태별 선호 색상 및 밝기 가져오기
                    
                    color_brightness_api_url = f"{predict_config['BehaPulse']['protocol']}://{predict_config['BehaPulse']['host']}:{predict_config['BehaPulse']['port']}/color_brightness/{personId}"

                    color_brightness_api_res = requests.get(color_brightness_api_url, verify=False)
                    print(color_brightness_api_res.json())
                    print(color_brightness_api_res.status_code)
                    color_brightness_obj = color_brightness_api_res.json()

                    preferred_brightness = None
                    preferred_color = None

                    for item in color_brightness_obj['colorBrightness']:
                        if predicted_label_ko == item['status']:
                            preferred_brightness = item['brightness']
                            preferred_color = item['color']


                if preferred_brightness and preferred_color:
                    preferred_hue, preferred_saturation = hex_to_hue_saturation(preferred_color)
                    print(f"Hue: {preferred_hue}, Saturation: {preferred_saturation}")

                    url = f'https://api.smartthings.com/v1/devices/{bulb_device_id}/commands'
                    headers = {
                        'Authorization': f'Bearer {access_token}',
                        'Content-Type': 'application/json'
                    }

                    payload = {
                        "commands": [
                            {
                                "component": "main", 
                                "capability": "switchLevel",
                                "command": "setLevel",
                                "arguments": [int(preferred_brightness)]  # Brightness Value (0-100)
                            },
                            {
                                "component": "main",
                                "capability": "colorControl",
                                "command": "setColor",
                                "arguments": [
                                    {
                                        "hue": int(preferred_hue),       # Hue value (0-100)
                                        "saturation": int(preferred_saturation)  # Saturation value (0-100)
                                    }
                                ]
                            }
                        ]
                    }
                    print("Trying to call API", url, payload, headers)
                    response = requests.post(url, headers=headers, json=payload)
                    print("Response:", response.status_code, response.text)
                    if response.status_code == 200:
                        print(f"전구의 밝기를 {int(preferred_brightness)}%로, 전구의 Hue를 {int(preferred_hue)}로, 전구의 Saturation을 {int(preferred_saturation)}로 설정했습니다.")
                    else:
                        print(f"오류 발생: {response.status_code}, {response.text}")
            
            ## Smart Bulb End
            
        except Exception as e:
            print('Exception:' + str(e))
            pass
        
        finally :
            # 200ms 대기
            time.sleep(20)