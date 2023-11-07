import matplotlib.pyplot as plt
import numpy as np
import glob
import re

from common import do_postprocessing, do_preprocessing


class SampleVisualizer:
    def __init__(self, num_receivers=3):
        # self.model_path = "/home/ilya/Downloads/trained_model_25ep.pt"
        self.samples_path = "/home/ilya/RDK-TOOLS-SW/new_dataset/raw_dataset"

        _, self.ax = plt.subplots(ncols=num_receivers)
        self.plots = np.empty_like(self.ax)

    def run(self):
        folder_paths = glob.glob(self.samples_path + "/*")

        allowed_types = ['test2']
        folder_paths = [path for path in folder_paths if any(t in path for t in allowed_types)]

        for folder_path in folder_paths:
            for video_path in glob.glob(folder_path + f"/*.npy"):
                data = np.load(video_path)


                if len(data.shape) == 5:
                    data = np.squeeze(data, axis=1)

                class_label = re.findall(r'/([\w_-]*)/[\d_]*.npy', video_path)[0]

                for frame in data:
                    frame = do_preprocessing(frame)
                    self.visualize(frame, class_label)

    def visualize(self, data, label):
        fig = plt.gcf()
        fig.canvas.set_window_title(label)

        for index, channel in enumerate(data):
            if self.plots[index] is None:
                self.plots[index] = self.ax[index].imshow(channel)
            else:
                self.plots[index].set_data(channel)

        plt.draw()
        plt.pause(1e-1)

    def clear(self):
        self.plots = np.empty_like(self.ax)


if __name__ == "__main__":
    SampleVisualizer().run()