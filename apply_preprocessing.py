import matplotlib.pyplot as plt
import numpy as np
import glob
import re
import os
from common import do_postprocessing


class Postprocessor:
    def __init__(self, num_receivers=3):
        # self.model_path = "/home/ilya/Downloads/trained_model_25ep.pt"
        self.samples_path = "/home/ilya/RDK-TOOLS-SW/dataset/raw_dataset"
        self.output_path = "/home/ilya/RDK-TOOLS-SW/dataset/preprocessed_dataset"


    def run(self):
        folder_paths = glob.glob(self.samples_path + "/*")

        for folder_path in folder_paths:
            for video_path in glob.glob(folder_path + f"/*.npy"):
                data = np.load(video_path)

                if len(data.shape) == 5:
                    data = np.squeeze(data, axis=1)

                data = do_postprocessing(data)

                class_label, index = re.findall(r'/([\w_-]*)/([\d_]*).npy', video_path)[0]
                self.save_recording(class_label, index, data)

    def save_recording(self, action_name, index, data):
        path = f"{self.output_path}/{action_name}"
        os.makedirs(path, exist_ok=True)
        np.save(f"{path}/{index}", data)



if __name__ == "__main__":
    Postprocessor().run()