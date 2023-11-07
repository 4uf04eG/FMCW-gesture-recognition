import numpy as np
import torch
from torchvision import transforms
from AvianRDKWrapper.ifxRadarSDK import *
from sklearn.preprocessing import MinMaxScaler, normalize
from cv2 import resize, INTER_AREA

def do_preprocessing(range_doppler):
    # range_doppler = 20 * np.log(np.abs(range_doppler))
    # Normalizing to [0, 1]

    range_doppler = np.abs(range_doppler)
    for index, channel in enumerate(range_doppler):
        min = np.min(channel)
        max = np.max(channel)
        range_doppler[index] = (channel - min) / (max - min)

    # min = np.min(range_doppler)
    # max = np.max(range_doppler)
    # normalized = (range_doppler - min) / (max - min)
    # range_doppler = normalized

    range_doppler = np.transpose(range_doppler, (2, 1, 0))
    range_doppler = resize(range_doppler, dsize=(128, 128), interpolation=INTER_AREA)
    range_doppler = np.transpose(range_doppler, (2, 1, 0))


    # # Scaling down and normalizing to 0 mean and 1 std
    # range_doppler = np.array(range_doppler)
    # range_doppler = np.expand_dims(range_doppler, 0)
    # tensor = torch.from_numpy(range_doppler).float()
    # range_doppler = VideoTransform((32, 32))(tensor)

    # Filtering
    # range_doppler = np.array(range_doppler)
    # range_doppler = np.where(range_doppler > 0.7, range_doppler, 0.0)
    # range_doppler = torch.from_numpy(range_doppler).float()

    #
    # range_doppler = np.array(range_doppler)[0, :, :, :]
    # range_doppler = [ca_cfar(channel, (4, 4), (8,8 ), 1.5) for channel in range_doppler]
    # range_doppler = np.array(range_doppler)
    # range_doppler = np.expand_dims(range_doppler, 0)
    # tensor = torch.from_numpy(range_doppler).float()
    # range_doppler = tensor

    return range_doppler

def do_postprocessing(range_doppler: np.array):
    # range_doppler = normalize_complex_arr(range_doppler)
    range_doppler = np.abs(range_doppler)
    min = np.min(range_doppler)
    max = np.max(range_doppler)
    normalized = (range_doppler - min) / (max - min)
    range_doppler = normalized
    # print(normalized)
    # range_doppler = MinMaxScaler().fit_transform(range_doppler)

    range_doppler = np.transpose(range_doppler, (2, 1, 0))
    range_doppler = resize(range_doppler, dsize=(32, 32), interpolation=INTER_AREA)
    range_doppler = np.transpose(range_doppler, (2, 1, 0))

    range_doppler = torch.from_numpy(range_doppler).float()
    range_doppler = torch.unsqueeze(range_doppler, 0)
    #
    #
    # range_doppler = np.array(range_doppler)
    # tensor = torch.from_numpy(range_doppler).float()
    # tensor = torch.unsqueeze(tensor, 0)
    # range_doppler = VideoTransform((32, 32))(tensor)

    return range_doppler

class VideoTransform(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, video):
        h, w = self.size
        L, C, H, W = video.size()
        rescaled_video = torch.FloatTensor(L, C, h, w)

        vid_mean = video.mean()
        vid_std = video.std()

        transform = transforms.Compose([
            transforms.Resize(self.size, antialias=True),
            # transforms.Normalize(0, 1),
        ])

        for l in range(L):
            frame = video[l, :, :, :]
            frame = transform(frame)
            # plt.imshow(frame.permute(1, 2, 0))
            # plt.show()
            rescaled_video[l, :, :, :] = frame

        return rescaled_video


def configure_device(device: Device, num_receivers: int):
    rx_mask = (1 << num_receivers) - 1

    metric = {
        'sample_rate_Hz': 2500000,
        'range_resolution_m': 0.025,
        'max_range_m': 1,
        'max_speed_m_s': 3,
        'speed_resolution_m_s': 0.024,
        'frame_repetition_time_s': 1 / 9.5,
        'center_frequency_Hz': 60_750_000_000,
        'rx_mask': rx_mask,
        'tx_mask': 1,
        'tx_power_level': 31,
        'if_gain_dB': 25,
        # "rx_antennas": [1, 2, 3],
        # "tx_antennas": [1],
        # "mimo_mode": "tdm",
        # "tx_power_level": 22,
        # "if_gain_dB": 30,
        # "range_resolution_m": 0.15,
        # "max_range_m": 3,
        # "max_speed_m_s": 3,
        # "speed_resolution_m_s": 0.04,
        # "frame_repetition_time_s": 1 / 8
    }
    # cfg = {
    #     'rx_mask': rx_mask,
    #     'tx_mask': 1,
    #     "tx_power_level": 31,
    #     "if_gain_dB": 60,
    #     "start_frequency_Hz": 61020098000,
    #     "end_frequency_Hz": 61479902000,
    #     "num_chirps_per_frame": 16,
    #     "num_samples_per_chirp": 128,
    #     "chirp_repetition_time_s": 7e-05,
    #     "frame_repetition_time_s": 5e-3,
    #     "sample_rate_Hz": 2330000
    # }

    cfg = device.metrics_to_config(**metric)
    # cfg = device.metrics_to_config(
    # sample_rate_Hz = 1_000_000,
    #
    # # range_resolution_m= 0.025,
    # # max_range_m= 1,
    # # max_speed_m_s= 5,
    # # speed_resolution_m_s= 0.025,
    # # frame_repetition_time_s= 1 / 6,
    # # center_frequency_Hz= 60_750_000_000,
    # rx_mask= rx_mask,
    # tx_mask= 1,
    # tx_power_level= 30,
    # if_gain_dB= 30,
    # )
    device.set_config(**cfg)