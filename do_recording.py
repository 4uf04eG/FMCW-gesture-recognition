import numpy

from AvianRDKWrapper.ifxRadarSDK import *
from range_doppler import DopplerAlgo, linear_to_dB
from common import do_preprocessing, configure_device
import os
import numpy as np
import glob
import re


def main():
    action_name = input("Action name: ")
    num_samples = int(input("Number of samples per action: "))
    num_frames = int(input("Number of frames per sample: "))

    recordings = []
    sample_i = 0

    while sample_i < num_samples:
        print(f'Recording {sample_i + 1} sample')
        recording = list(record(num_frames))

        answer = input("Save it? (Y or N) ").lower()

        if answer == "y":
            recordings.append(recording)
            sample_i += 1

    processed = [[np.array(do_preprocessing(frame)) for frame in sample] for sample in recordings]
    save_recordings('new_dataset/raw_dataset', 'new_dataset/processed_dataset', action_name, recordings, processed)


def record(num_frames):
    with Device() as device:
        num_rx_antennas = device.get_sensor_information()["num_rx_antennas"]
        configure_device(device, num_rx_antennas)
        algo = DopplerAlgo(device.get_config(), num_rx_antennas)

        for frame_number in range(num_frames):  # For each frame
            frame_data = device.get_next_frame()
            data_all_antennas = []

            for i_ant in range(0, num_rx_antennas):  # For each antenna
                mat = frame_data[i_ant, :, :]
                dfft_dbfs = algo.compute_doppler_map(mat, i_ant)
                data_all_antennas.append(dfft_dbfs)

            yield data_all_antennas


def save_recordings(raw_path, processed_path, action_name, raw, processed):
    raw_path = f'{raw_path}/{action_name}'
    os.makedirs(raw_path, exist_ok=True)

    processed_path = f'{processed_path}/{action_name}'
    os.makedirs(processed_path, exist_ok=True)

    file_paths = glob.glob(raw_path + '/*.npy')
    last_index = len(file_paths)

    for index, sample in enumerate(raw):
        numpy.save(f"{raw_path}/{last_index + index:02d}", sample)

    for index, sample in enumerate(processed):
        numpy.save(f"{processed_path}/{last_index + index:02d}", sample)


if __name__ == "__main__":
    main()
