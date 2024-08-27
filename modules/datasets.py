from torch.utils.data import Dataset
import os
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

def apply_fixed_window(df, window_size):
    """Applies a fixed-size window to the DataFrame."""
    num_rows = len(df)
    windows = []
    
    # Split the dataframe into chunks of window_size
    for start in range(0, num_rows, window_size):
        end = start + window_size
        if end <= num_rows:  
            window = df.iloc[start:end].copy()
            windows.append(window)
    
    return windows

def load_dataset(root, mac_adress, window_size): 
    print('Loading dataset...')
    
    data = {}
    
    mac_adress = [mac.replace(':', '_') for mac in mac_adress]
    # print(mac_adress)
    
    for mac in mac_adress:
        dir_path = os.path.join(root, mac)
        if not os.path.exists(dir_path):
            print(f"Directory for MAC address {mac} not found.")
            continue
        
        if mac not in data:
            data[mac] = {} 
            
        files = os.listdir(dir_path)
        # print(f'Files in {mac}: {files}')
        
        for file in files:
            if file.endswith('.csv'):  
                file_path = os.path.join(root, mac, file)
                
                df = pd.read_csv(file_path)
                
                # 불필요한 컬럼 삭제 진행
                df = df.drop(columns=['mac','time'], errors='ignore')
                            
                label = file.split('_')[0]
                # label += '_' + file.split('_')[1]
                windows = apply_fixed_window(df, window_size)
                
                if label not in data[mac]:
                    data[mac][label] = []  # Initialize a list for each label if it doesn't exist
                    
                data[mac][label].extend(windows)
                    
    return data
    

def SplitDataset(dir:str, val_size:float=0.1, seed=40, window_size:int=10, mac_adress:list=[]):
    data = load_dataset(dir, mac_adress, window_size)
  
    print('Splitting dataset...')
    train_data = []
    val_data = []
    
    for mac in data:
        for label in data[mac]:
            windows = data[mac][label]
            train_windows, val_windows = train_test_split(windows, test_size=val_size, random_state=seed)
            
            train_data.extend([(window, label) for window in train_windows])
            val_data.extend([(window, label) for window in val_windows])
    
    return train_data, val_data

class CSIDataset(Dataset):
    def __init__(self, data, label_map):
        self.data = data
        self.label_map = label_map

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        window, label = self.data[idx]
        
        # Convert DataFrame to numpy array (or PyTorch tensor if required)
        window = window.values.astype(np.float32)
        
        # Convert label to integer using label_map
        label = self.label_map.get(label, -1)  # Get label as integer, -1 if not found
        
        return window, label
    
if __name__ == '__main__':
    # SplitDataset(dir = '../data', mac_adress = ['00:00:00:00:00:00', '11:11:11:11:11:11'])
    pass
    

        