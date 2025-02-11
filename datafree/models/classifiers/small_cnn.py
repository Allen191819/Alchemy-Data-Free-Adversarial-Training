import torch.nn as nn


class SmallCNN(nn.Module):
    def __init__(self, drop=0.5, num_classes=10, num_channels=1):
        super(SmallCNN, self).__init__()

        self.num_channels = num_channels
        self.num_classes = num_classes

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(self.num_channels, 32, 3),
            nn.ReLU(True),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 32, 3),
            nn.ReLU(True),
            nn.BatchNorm2d(32),

            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(True),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, 3),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 5 * 5, 200),
            nn.ReLU(True),

            nn.Dropout(drop),
            nn.Linear(200, 200),
            nn.ReLU(True),

            nn.Linear(200, self.num_classes),
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input):
        features = self.feature_extractor(input)
        logits = self.classifier(features.view(-1, 64 * 5 * 5))
        return logits

class SmallCNN_2(nn.Module):
    def __init__(self, drop=0.5, num_classes=10, num_channels=1):
        super(SmallCNN_2, self).__init__()

        self.num_channels = num_channels
        self.num_classes = num_classes

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(self.num_channels, 32, 3),
            nn.ReLU(True),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 32, 3),
            nn.ReLU(True),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 32, 3),
            nn.ReLU(True),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, 3),
            nn.ReLU(True),
            nn.BatchNorm2d(64),

            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(True),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, 3),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 4 * 4, 200),
            nn.ReLU(True),

            nn.Dropout(drop),
            nn.Linear(200, 200),
            nn.ReLU(True),

            nn.Linear(200, self.num_classes),
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input):
        features = self.feature_extractor(input)
        logits = self.classifier(features.view(-1, 64 * 4 * 4))
        return logits


class SmallFCN_1(nn.Module):
    def __init__(self, drop=0.5, num_classes=10, num_channels=1):
        super(SmallFCN_1, self).__init__()

        self.num_channels = num_channels
        self.num_classes = num_classes

        self.fc1 = nn.Sequential(
            nn.Linear(32*32,512),
            nn.ReLU(True),
            nn.Dropout(drop)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512,512),
            nn.ReLU(True),
            nn.Dropout(drop)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(512,200),
            nn.ReLU(True),
            nn.Dropout(drop)
        )
        self.classifier = nn.Sequential(
            nn.Linear(200, self.num_classes),
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input):
        x = self.fc1(input.view(-1,32*32))
        x = self.fc2(x)
        x = self.fc3(x)
        logits = self.classifier(x)
        return logits


class SmallFCN_2(nn.Module):
    def __init__(self, drop=0.5, num_classes=10, num_channels=1):
        super(SmallFCN_2, self).__init__()

        self.num_channels = num_channels
        self.num_classes = num_classes

        self.fc1 = nn.Sequential(
            nn.Linear(32*32,512),
            nn.ReLU(True),
            nn.Dropout(drop)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512,512),
            nn.ReLU(True),
            nn.Dropout(drop)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(512,512),
            nn.ReLU(True),
            nn.Dropout(drop)
        )
        self.fc4 = nn.Sequential(
            nn.Linear(512,256),
            nn.ReLU(True),
            nn.Dropout(drop)
        )
        self.fc5 = nn.Sequential(
            nn.Linear(256,200),
            nn.ReLU(True),
            nn.Dropout(drop)
        )
        self.classifier = nn.Sequential(
            nn.Linear(200, self.num_classes),
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input):
        x = self.fc1(input.view(-1,32*32))
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        logits = self.classifier(x)
        return logits

