import numpy as np
import glob
import re
import os
from common import do_preprocessing, do_inference_processing


class Preprocessor:
    def __init__(self):
        self.samples_path = "/home/ilya/RDK-TOOLS-SW/new_dataset/raw_dataset"
        self.output_path = "/home/ilya/RDK-TOOLS-SW/new_dataset/preprocessed_dataset"

    def run(self):
        folder_paths = glob.glob(self.samples_path + "/*")

        for folder_path in folder_paths:
            for video_path in glob.glob(folder_path + "/*.npy"):
                data = np.load(video_path)

                if len(data.shape) == 5:
                    data = np.squeeze(data, axis=1)

                data = [do_preprocessing(frame) for frame in data]

                class_label, index = re.findall(r'/([\w_-]*)/([\d_]*).npy', video_path)[0]
                self.save_recording(class_label, index, data)

    def save_recording(self, action_name, index, data):
        path = f"{self.output_path}/{action_name}"
        os.makedirs(path, exist_ok=True)
        np.save(f"{path}/{index}", data)


if __name__ == "__main__":
    Preprocessor().run()
