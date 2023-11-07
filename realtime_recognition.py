from AvianRDKWrapper.ifxRadarSDK import *
from dbf import DBF
from common import do_preprocessing, do_postprocessing, configure_device
from range_doppler import DopplerAlgo, linear_to_dB
from cfar import ca_cfar
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import asyncio


class PredictionPipeline:
    def __init__(self, debouncer_length=7, num_receivers=3):
        # self.model_path = "/home/ilya/Downloads/trained_model_25ep.pt"
        self.model_path = "/home/ilya/Downloads/trained_model_finetune_4cl_20ep.pt"
        self.encoder_path = "/home/ilya/Downloads/encoder_4(1).npy"

        self.model = self.load_model(self.model_path)
        self.frame_memory = []

        self.encoder: LabelEncoder = LabelEncoder()
        self.encoder.classes_ = np.load(self.encoder_path)

        self.debounce_length = debouncer_length
        self.num_receivers = num_receivers

        _, self.ax = plt.subplots(ncols=num_receivers)
        self.plots = np.empty_like(self.ax)

    def run(self):
        with Device() as device:
            num_rx_antennas = self.num_receivers
            configure_device(device, num_rx_antennas)

            algo = DopplerAlgo(device.get_config(), num_rx_antennas)
            dbf = DBF(num_rx_antennas)

            while True:
                frame_data = device.get_next_frame()

                data_all_antennas = []

                for i_ant in range(0, num_rx_antennas):  # For each antenna
                    mat = frame_data[i_ant, :, :]
                    dfft_dbfs = algo.compute_doppler_map(mat, i_ant)
                    data_all_antennas.append(dfft_dbfs)

                if len(self.frame_memory) + 1 == self.debounce_length:
                    self.frame_memory.pop(0)


                processed = do_postprocessing(data_all_antennas)
                # processed = torch.unsqueeze(processed, 0)

                self.frame_memory.append(processed)

                # labels = ['Pinch index', 'Palm tilt', 'Finger slider', 'Pinch pinky',
                #           'Slow swipe', 'Fast swipe', 'Push', 'Pull', 'Finger rub', 'Circle', 'Palm hold', 'No action']
                # print(labels[np.argmax(self.predict_probabilities(self.frame_memory))])
                print(self.encoder.inverse_transform([np.argmax(self.predict_probabilities(self.frame_memory))])[0])
                self.visualize(processed)

    def load_model(self, path):
        return torch.load(path, map_location=torch.device('cpu'))

    def predict_probabilities(self, data):
        data = torch.cat(data)
        # Adding dummy dimension for batch
        data = data[:, None, :]

        with torch.no_grad():
            predictions = self.model(data)

        return predictions

    def visualize(self, data):
        for index, channel in enumerate(data[0, :, :, :]):
            if self.plots[index] is None:
                self.plots[index] = self.ax[index].imshow(channel)
            else:
                self.plots[index].set_data(channel)

        plt.draw()
        plt.pause(1e-3)


class FeatureExtractor(torch.nn.Module):
    def __init__(self, in_features, out_features) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_features, 32, 3)
        self.norm1 = torch.nn.BatchNorm2d(32)
        self.pool1 = torch.nn.MaxPool2d(2)

        self.conv2 = torch.nn.Conv2d(32, 64, 3)
        self.norm2 = torch.nn.BatchNorm2d(64)
        self.pool2 = torch.nn.MaxPool2d(2)

        self.conv3 = torch.nn.Conv2d(64, 128, 3)
        self.norm3 = torch.nn.BatchNorm2d(128)
        self.pool3 = torch.nn.MaxPool2d(2)

        self.conv4 = torch.nn.Conv2d(128, 256, 3)
        self.norm4 = torch.nn.BatchNorm2d(256)
        self.pool4 = torch.nn.MaxPool2d(2)

        self.relu = torch.nn.ReLU()
        self.flatten = torch.nn.Flatten()
        self.fc = torch.nn.Linear(512, out_features)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x):

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu(x)
        x = self.pool3(x)

        # x = self.conv4(x)
        # x = self.norm4(x)
        # x = self.relu(x)
        # x = self.pool4(x)

        x = self.flatten(x)
        x = self.fc(x)
        x = self.dropout(x)

        return x


class GestureNet(torch.nn.Module):
    def __init__(self, num_input_channels = 4, num_cnn_features=256, num_rnn_hidden_size=256, num_classes=7) -> None:
        super().__init__()

        self.num_rnn_hidden_size = num_rnn_hidden_size

        self.frame_model = FeatureExtractor(num_input_channels, num_cnn_features)
        # self.frame_model = vgg11()
        # first_conv_layer = [torch.nn.Conv2d(4, 3, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)]
        # first_conv_layer.extend(list(self.frame_model.features))

        # self.frame_model.features = torch.nn.Sequential(*first_conv_layer)
        # self.frame_model.classifier[6] = torch.nn.Linear(4096, num_cnn_features)

        self.temporal_model = torch.nn.LSTM(input_size=num_cnn_features, hidden_size=num_rnn_hidden_size)

        self.fc1 = torch.nn.Linear(num_rnn_hidden_size, num_rnn_hidden_size // 2)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(num_rnn_hidden_size // 2, num_classes)

    def forward(self, x):
        hidden = None

        for frame in x:
            features = self.frame_model(frame)
            features = torch.unsqueeze(features, 0)
            out, hidden = self.temporal_model(features, hidden)

        # out = torch.squeeze(out)
        # print(out.shape)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        # out = torch.nn.Softmax(dim=1)(out)


        return out

class FinetuneGestureNet(GestureNet):
  def __init__(self, num_classes, weights_path="/content/drive/MyDrive/research_project_models/trained_model_25ep.pt"):
    super().__init__(num_input_channels=3, num_classes=12)

    self.load_state_dict(torch.load(weights_path).state_dict())
    self.fc2 = torch.nn.Linear(self.num_rnn_hidden_size // 2, num_classes)

if __name__ == '__main__':
    PredictionPipeline().run()

