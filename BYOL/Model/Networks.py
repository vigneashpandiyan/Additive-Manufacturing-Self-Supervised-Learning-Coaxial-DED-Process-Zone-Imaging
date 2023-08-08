import torch.nn as nn
import torch.nn.functional as F

class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
    
    def forward(self, x):
        # Do your print / debug stuff here
        print(x.shape)
        return x

class EmbeddingNet(nn.Module):    
    def __init__(self,dropout,embedding):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(
                                     # PrintLayer(),
                                     nn.Conv2d(in_channels=3, out_channels=4, kernel_size=16),
                                     nn.BatchNorm2d(4),
                                     nn.ReLU(),
                                     nn.MaxPool2d(2,2),
                                     nn.Dropout(dropout),
                                     # PrintLayer(),
                                     nn.Conv2d(in_channels=4, out_channels=8, kernel_size=16),
                                     nn.BatchNorm2d(8),
                                     nn.ReLU(),
                                     nn.MaxPool2d(2,2),
                                     nn.Dropout(dropout),
                                     # PrintLayer(),
                                     nn.Conv2d(in_channels=8, out_channels=16, kernel_size=16),
                                     nn.BatchNorm2d(16),
                                     nn.ReLU(),
                                     nn.MaxPool2d(2,2),
                                     nn.Dropout(dropout),
                                     # PrintLayer(),
                                     nn.Conv2d(in_channels=16, out_channels=32, kernel_size=16),
                                     nn.BatchNorm2d(32),
                                     nn.ReLU(),
                                     nn.MaxPool2d(2,2),
                                     nn.Dropout(dropout),
                                     # PrintLayer(),#torch.Size([256, 32, 15, 5])
                                     
                                     )

        self.fc = nn.Sequential(nn.Linear(32 *5 * 15, embedding),
                                nn.BatchNorm1d(embedding),
                                # nn.ReLU(),
                                # # PrintLayer(),##torch.Size([100, 256])
                                # nn.Linear(128, embedding),
                                # nn.BatchNorm1d(128),
                                )

    def forward(self, x):
        output = self.convnet(x)
        output = output.contiguous().view(-1, 32 *5 * 15)
        output = self.fc(output)
        return output
    
class ConvNet(nn.Module):
    def __init__(self, embedding_net,embedding, projection_size,n_classes):
        super(ConvNet, self).__init__()
    
        self.online_common = embedding_net
        self.predictor_online =  nn.Linear(embedding, projection_size)
        self.online = nn.Sequential(self.online_common, self.predictor_online)
        self.classifier =  nn.Linear(projection_size, n_classes)
        self.network = nn.Sequential(self.online,
                                           self.classifier)
        
    def forward(self, x):
        output = self.network(x)
        
        return output

    def get_embedding(self, x):
        return self.online(x)



