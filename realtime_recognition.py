import threading
import os.path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import torch
import torch.nn.functional as F
from AvianRDKWrapper.ifxRadarSDK import *
from debouncer import Debouncer
from algorithms.dbf import DBF
from common import do_preprocessing, do_inference_processing, configure_device
from algorithms.range_doppler import DopplerAlgo
from model import FeatureExtractor, GestureNet, FinetuneGestureNet
# I undestood too late that recording whole model requires it to be under the same module
# That's a quick but terrible fix
FeatureExtractor = FeatureExtractor
GestureNet = GestureNet
FinetuneGestureNet = FinetuneGestureNet



class PredictionPipeline:
    def __init__(self, num_receivers=3):
        self.model_path = 'model/trained_model_finetune_7cl_25ep_custom_split.pt'
        self.encoder_path = "model/encoder_7.npy"

        self.model = self.load_model(self.model_path)

        self.encoder: LabelEncoder = LabelEncoder()
        self.encoder.classes_ = np.load(self.encoder_path)
        self.debouncer = Debouncer(memory_length=12)

        self.num_receivers = num_receivers
        self.visualizer = Visualizer(self.encoder, self.num_receivers)
             
    def start_gui(self):
        self.visualizer.start_gui()   

    def run(self):
        with Device() as device:
            num_rx_antennas = self.num_receivers
            configure_device(device, num_rx_antennas)

            algo = DopplerAlgo(device.get_config(), num_rx_antennas)
            # In the end, we don't use angle info
            # dbf = DBF(num_rx_antennas)

            while True:
                frame_data = device.get_next_frame()

                data_all_antennas = []

                for i_ant in range(0, num_rx_antennas):  # For each antenna
                    mat = frame_data[i_ant, :, :]
                    dfft_dbfs = algo.compute_doppler_map(mat, i_ant)
                    data_all_antennas.append(dfft_dbfs)

                range_doppler = do_inference_processing(data_all_antennas)
                
                self.debouncer.add_scan(range_doppler)
                probs = self.predict_probabilities(self.debouncer.get_scans())
                     
                probs = torch.squeeze(probs)
                probs = F.softmax(probs, dim=-1)
                probs = probs.numpy()
                
                self.visualizer.add_prediction_step(range_doppler.numpy(), probs)
                label_index = self.debouncer.debounce(probs)

                if label_index is not None:
                    print(self.encoder.inverse_transform([label_index])[0])
     

    def load_model(self, path):
        return torch.load(path, map_location=torch.device('cpu'))

    def predict_probabilities(self, data):
        data = torch.cat(data)
        # Adding dummy dimension for batch
        data = data[:, None, :]

        with torch.no_grad():
            predictions = self.model(data)

        return predictions

   
class Visualizer():
    def __init__(self, encoder, num_receivers):
        self.encoder = encoder
        self.num_receivers = num_receivers
        
        self.last_range_doppler = None
        self.fig = plt.figure(figsize=(18, 10))
        self.plots = list(self.prepare_range_doppler_subplots())
        self.anim = FuncAnimation(self.fig, self.visualize_data, interval=100)
        
        self.num_classes = len(self.encoder.classes_)
        self.prob_history_length = 30
        self.probs_history = pd.DataFrame(columns=np.arange(0, self.num_classes, 1))
        self.fig2 = plt.figure(figsize=(18, 10))
        self.plots2 = list(self.prepare_probs_subplots())
        self.anim2 = FuncAnimation(self.fig2, self.visualize_probs, interval=100)
        
    def start_gui(self):
        plt.show()
        
    def add_prediction_step(self, range_doppler, probs):
        if len(self.probs_history) >= self.prob_history_length:
            self.probs_history = self.probs_history.iloc[1:]
            
        self.last_range_doppler = range_doppler
        self.probs_history = pd.concat([self.probs_history, 
                                        pd.DataFrame.from_records([{index: value for index, value in enumerate(probs)}])], ignore_index=True)
        
    def prepare_range_doppler_subplots(self):
        gs = GridSpec(nrows=1, ncols=self.num_receivers, hspace=0.7)
        
        for index in range(self.num_receivers):
            ax = self.fig.add_subplot(gs[index])
            ax.set_xlabel('Velocity (m/s)')
            ax.set_ylabel('Range (m)')
            ax.set_xticks([0, 8, 16, 24, 31])
            ax.set_yticks([0, 8, 16, 24, 31])
            ax.set_xticklabels([-3, -1.5, 0, 1.5, 3])
            ax.set_yticklabels([0, 0.25, 0.5, 0.75, 1])
            yield ax.imshow(np.zeros((32, 32)))
        
    def prepare_probs_subplots(self, nrows=4, ncols=2):
        gs = GridSpec(nrows=nrows, ncols=ncols, hspace=0.7)
        # HADCODED, CHANGE when number of classes changes
        # ['finger_circle' 'finger_rub' 'no-action' 'palm_hold' 'pull' 'push' 'swipe']
        encoder_to_better_labels = [0, 3, 6, 1, 4, 5, 2]
        # encoder_to_better_labels = [0, 3, 1, 4, 5, 2]

        for index in range(self.num_classes):
                new_index = encoder_to_better_labels[index]
                
                i = new_index// ncols
                j = new_index % ncols

                ax = self.fig2.add_subplot(gs[i, j])
                ax.set_title(self.encoder.inverse_transform([index])[0])
                ax.set_xlim(0, self.prob_history_length)
                ax.set_ylim(0, 1)
                ax.set_xlabel('Number of frames')
                ax.set_ylabel('Probability')
                
                yield ax.plot([], [])[0]
                
    def visualize_data(self, _):
        if self.last_range_doppler is None:
            return
        
        for index, channel in enumerate(self.last_range_doppler[0, :, :, :]):
            self.plots[index].set_data(channel)
            self.plots[index].set_clim(channel.min(), channel.max())

    def visualize_probs(self, _):
        for index, (_, column_data) in enumerate(self.probs_history.items()):
            self.plots2[index].set_data(column_data.index, column_data)
    
if __name__ == '__main__':
    obj = PredictionPipeline()
    
    threading.Thread(target=obj.run).start()
    obj.start_gui()
