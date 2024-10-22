"""
    Raspberry Pi에서 Flask Server로 CSI 데이터 전송
"""

import bluetooth
import requests
import time

# HTTP 프로토콜로 CSI 데이터 전송
def send_file_lines_as_json(url, file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # 중복 제거
    lines = list(dict.fromkeys(lines))
    
    json_data = {
        "file_lines": lines
    }

    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, json=json_data, headers=headers)
    
    print(response.status_code)
    print(response.text)

# UART로 받은 CSI 데이터 중 오류가 있는 데이터를 제거
def process_text_file(input_file, output_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    processed_lines = []
    for line in lines:
        if line.startswith('CSI_DATA') and line.count('[') == 1 and line.count(']') == 1:
            processed_lines.append(line)

    with open(output_file, 'w') as file:
        # 중복 제거
        processed_lines = list(dict.fromkeys(processed_lines))
        file.writelines(processed_lines)

def main():
    sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
    target_name = "ESP_SPP_ACCEPTOR"
    now = time.localtime()
    file_name = "CSI_BT_" + time.strftime("%Y%m%d_%H%M%S", now) + ".txt"
    f = open(file_name, 'w')  
    buf = ''

    nearby_devices = bluetooth.discover_devices(duration=8, lookup_names=True)
    target_address = None

    for addr, name in nearby_devices:
        if target_name == name:
            target_address = addr
            break

    if target_address is None:
        print("Could not find target Bluetooth device.")
        return

    port = 1
    sock.connect((target_address, port))
    print(f"Connected to {target_name} at {target_address}")

    try:
        while True:
            data = sock.recv(1024)
            if len(data) > 0:
                decoded_data = data.decode('utf-8', errors='ignore')
                buf += decoded_data
                try:
                    f.write(buf)
                    f.flush()
                    
                    # 버퍼 초기화
                    buf = ''  
                    
                    process_text_file(file_name, "(modified)" + file_name)
                    
                    # Flask Server로 CSI 데이터 전송
                    send_file_lines_as_json('https://192.9.200.141/device/CSI', "(modified)" + file_name)

                except UnicodeDecodeError as e:
                    print(f"Failed to decode data: {e}")
            else:
                print("No data received")
    except bluetooth.btcommon.BluetoothError as e:
        print(f"Bluetooth error: {e}")
    finally:
        f.close()
        sock.close()
        print("Disconnected")

if __name__ == "__main__":
    main()
