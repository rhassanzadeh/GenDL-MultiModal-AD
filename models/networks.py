import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseModule(nn.Module):
    def initialize_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm3d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            init.constant_(m.bias, 0)
            

class GeneratorsMRItoFNC(BaseModule):
    def __init__(self):
        super().__init__()
        
        ch1, ch2, ch3 = 8, 16, 32
        
        self.downsampling = nn.Sequential(
            nn.Conv3d(1, ch1, kernel_size=5, stride=1, padding=0), # padding=1
            nn.BatchNorm3d(ch1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=3),

            nn.Conv3d(ch1, ch2, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm3d(ch2),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=3),

            nn.Conv3d(ch2, ch3, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm3d(ch3),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=3),
        )

        self.flatten = nn.Flatten()

        self.fc = nn.Sequential(
#             nn.Linear(2560, 1378),
            nn.Linear(1152, 1378),
            nn.Tanh(),
        )
        
        # Initialize layers
        self.initialize_weights(self.downsampling)
        self.initialize_weights(self.fc)
        
    def forward(self, x):
        x = self.downsampling(x)
        x = self.flatten(x)
        x = self.fc(x)
        
        return x

    
class GeneratorFNCtosMRI(BaseModule):
    def __init__(self):
        super().__init__()
        
        ch1, ch2, ch3, ch4 = 8, 8, 16, 16
        
        self.fc = nn.Sequential(
#             nn.Linear(1378, 2400),
            nn.Linear(1378, 1280),
            nn.ReLU(inplace=True),
        )

        self.upsampling = nn.Sequential(
            nn.ConvTranspose3d(ch4, ch3, kernel_size=3, stride=2, output_padding=1),
            nn.BatchNorm3d(ch4),
            nn.LeakyReLU(inplace=True),
            nn.Upsample(size=(12, 15, 12)),
            
            nn.ConvTranspose3d(ch3, ch2, kernel_size=3, stride=2, output_padding=1),
            nn.BatchNorm3d(ch2),
            nn.LeakyReLU(inplace=True),
            nn.Upsample(size=(27, 33, 27)),
            
            nn.ConvTranspose3d(ch2, ch1, kernel_size=4, stride=2, output_padding=1),
            nn.BatchNorm3d(ch1),
            nn.LeakyReLU(inplace=True),
                        
            nn.ConvTranspose3d(ch1, 1, kernel_size=7, stride=2, output_padding=1),
            nn.BatchNorm3d(1),
            nn.LeakyReLU(inplace=True),
#             nn.Upsample(size=(121, 145, 121))
            
            # Final convolution layer to refine features and adjust dimensions
            nn.Conv3d(1, 1, kernel_size=3, padding=1),
            nn.Upsample(size=(121, 145, 121)), # Final size adjustment
            nn.Tanh(),
        )
        
        # Initialize layers
        self.initialize_weights(self.upsampling)
        self.initialize_weights(self.fc)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 16, 4, 5, 4) # Reshape to 3D feature map # 5, 6, 5
        x = self.upsampling(x) # shape: 121, 145, 121
        
        return x

    
class DiscriminatorsMRI(BaseModule):
    def __init__(self):
        super().__init__()

        ch1 = 8
        ch2 = 16
        ch3 = 32
        ch4 = 32
        
        self.cnn = nn.Sequential(
            nn.Conv3d(1, ch1, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm3d(ch1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=3),

            nn.Conv3d(ch1, ch2, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm3d(ch2),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=3),

            nn.Conv3d(ch2, ch3, kernel_size=3, padding=1),
            nn.BatchNorm3d(ch3),
            nn.LeakyReLU(inplace=True),

            nn.Conv3d(ch3, ch4, kernel_size=3, padding=1),
            nn.BatchNorm3d(ch4),
            nn.LeakyReLU(inplace=True),

            nn.Conv3d(ch4, ch4, kernel_size=3, padding=1),
            nn.BatchNorm3d(ch4),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=3),
        )
        
        self.flatten = nn.Flatten()
            
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(2*ch4, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=True),
#             nn.Dropout(0.2),
            nn.Linear(64, 1),
        )
        
        # Initialize layers
        self.initialize_weights(self.cnn)
        self.initialize_weights(self.fc)
            
    def forward(self, x):
        x = self.cnn(x)
        x = self.flatten(x)
        x = self.fc(x)
        
        return x

    
class DiscriminatorFNC(BaseModule):
    def __init__(self):
        super().__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(1378, 8),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(),
            nn.Linear(8, 1),
        )

        # Initialize layers
        self.initialize_weights(self.fc)
            
    def forward(self, x):
        x = self.fc(x)
        
        return x
    
    