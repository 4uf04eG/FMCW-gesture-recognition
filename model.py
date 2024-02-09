import torch

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
    def __init__(self, num_input_channels=4, num_cnn_features=256, num_rnn_hidden_size=256, num_classes=7) -> None:
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
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        # out = torch.nn.Softmax(dim=1)(out)

        return out


class FinetuneGestureNet(GestureNet):
    def __init__(self, num_classes,
                 weights_path="/content/drive/MyDrive/research_project_models/trained_model_25ep.pt"):
        super().__init__(num_input_channels=3, num_classes=12)

        self.load_state_dict(torch.load(weights_path).state_dict())
        # self.fc2 = torch.nn.Linear(self.num_rnn_hidden_size // 2, num_classes)
        
        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(self.num_rnn_hidden_size // 2, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, num_classes)
        )
