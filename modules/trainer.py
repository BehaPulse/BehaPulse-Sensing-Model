import torch
from time import time
from tqdm import tqdm
import torch.nn.functional as F

class Trainer():
    
    def __init__(self, model, optimizer, scheduler, loss, metrics, device, logger, amp, interval=100):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.loss = loss
        self.metrics = metrics
        self.device = device
        self.logger = logger
        self.amp = amp
        self.interval = interval
        
        # History
        self.loss_sum = 0 # Epoch loss sum
        self.loss_mean = 0 # epoch loss mean
        self.filenames = list()
        self.y = list()
        self.y_preds = list()
        self.score_dict = dict()
        self.elapsed_time = 0
        
    def train(self, mode, dataloader, epoch_index=0):
        start_timestamp = time()
        self.model.train() if mode == 'train' else self.model.eval()
        
        for batch_index, (x,y) in enumerate(tqdm(dataloader)):
            x, y = x.to(self.device, dtype=torch.float), y.to(self.device, dtype=torch.long)
            print(x.shape)
            x = x.unsqueeze(1).float()
            print(x.shape)
            
            # Inference
            y_pred = self.model(x)
            y_pred = F.log_softmax(y_pred, dim=1)
            _, predicted = torch.max(y_pred.data, 1)
            
            # Loss
            loss = self.loss(y_pred, y)
            
            # Update
            if mode == 'train':
                self.optimizer.zero_grad()
                
                if self.amp is None:
                    loss.backward()
                
                else:
                    with self.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                    
                self.optimizer.step()
            
            elif mode in ['val','test']:
                pass
            else:
                raise ValueError('Mode should be either train, val, or test')
            
            # History
            #self.filenames += filename
            self.loss_sum += loss.item()
            self.y_preds.append(predicted)
            self.y.append(y)
            
            # Logging
            if batch_index % self.interval == 0:
                msg = f"batch: {batch_index}/{len(dataloader)} loss: {loss.item()}"
                self.logger.info(msg)
                
        self.scheduler.step()

        # Epoch history
        self.loss_mean = self.loss_sum / len(dataloader)
        
        # Metric
        self.y_preds = torch.cat(self.y_preds, dim=0).cpu().tolist()
        self.y = torch.cat(self.y, dim=0).cpu().tolist()
        
        for metric_name, metric_func in self.metrics.items():
            score = metric_func(self.y, self.y_preds)
            self.score_dict[metric_name] = score
        
        # Elapsed time
        end_timestamp = time()
        self.elapsed_time = end_timestamp - start_timestamp
        
    def clear_history(self):
        self.loss_sum = 0
        self.loss_mean = 0
        self.y_preds = list()
        self.y = list()
        self.score_dict = dict()
        self.elapsed_time = 0
                
                