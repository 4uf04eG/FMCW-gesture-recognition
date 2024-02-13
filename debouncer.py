class Debouncer:
    def __init__(self, detect_threshold=0.6, noise_threshold=0.3, memory_length=12, min_num_detections=3):
        # Based on https://dl.acm.org/doi/pdf/10.1145/3411764.3445367

        self.detect_threshold = detect_threshold
        self.noise_threshold = noise_threshold,
        self.memory_length = memory_length
        self.min_num_detection = min_num_detections

        self.frame_memory = []
        self.detection_memory = []

    def add_scan(self, frame):
        if len(self.frame_memory) > 0 and len(self.frame_memory) + 1 == self.memory_length:
            self.frame_memory.pop(0)

        self.frame_memory.append(frame)

    def get_scans(self):
        return self.frame_memory

    def debounce(self, probs):
        detected_actions = (probs > self.detect_threshold).nonzero()[0]

        if len(detected_actions) != 1:
            if len(detected_actions) == 0:
                print('Not sure about the action')
            else:
                print(f'Not sure about the action. There are {len(detected_actions)} possibilities')
            return None

        noise_actions = probs[probs < self.noise_threshold]
        if len(noise_actions) != len(probs) - 1:
            return None

        main_action = detected_actions[0]

        if len(self.detection_memory) > 0 and len(self.detection_memory) + 1 >= self.min_num_detection:
            self.detection_memory.pop(0)

        self.detection_memory.append(main_action)

        if all(self.detection_memory == main_action):
            return main_action
        else:
            return None
