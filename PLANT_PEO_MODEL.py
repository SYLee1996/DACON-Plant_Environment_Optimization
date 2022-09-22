import timm 
import torch
from torch import nn
from torchvision import models


class CNNRegressor(torch.nn.Module):
    def __init__(self, config):
        super(CNNRegressor, self).__init__()
        self.ouput_dim = config['embedding_dim']
        self.in_channels = config['in_channels']
        
        self.layer1 = torch.nn.Sequential(
            nn.Conv2d(self.in_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.layer2 = torch.nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.layer3 = torch.nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.layer4 = torch.nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.regressor = torch.nn.Sequential(
            nn.Linear(64*32*32, self.ouput_dim),
            # nn.ReLU(),
            )

    def forward(self, x):
        # Simple CNN Model (Batch, 3, 128, 128 -> Batch, 64, 7, 7)
        x = self.layer1(x) # (Batch, 3, 128, 128)
        x = self.layer2(x) # (Batch, 16, 32, 32)
        x = self.layer3(x) # (Batch, 32, 16, 16)
        x = self.layer4(x) # (Batch, 64, 7, 7) -> Flatten (Batch, 64*7*7(=3136))
        x = torch.flatten(x, start_dim=1) # Regressor (Batch, 3136) -> (Batch, 1)
        
        out = self.regressor(x)
        return out
    
    
class CNN_Encoder(nn.Module):
    def __init__(self, config):
        super(CNN_Encoder, self).__init__()
        self.model = config['model']
        self.ouput_dim = config['embedding_dim']
        self.in_channels = config['in_channels']
        self.pretrain = config['pretrain']
        
        if self.model == 'efficientnet_b4':
            self.model = timm.create_model('efficientnet_b4', pretrained=self.pretrain, num_classes=self.ouput_dim) 
            
        elif self.model == 'efficientnet_el_pruned':
            self.model = timm.create_model('efficientnet_el_pruned', pretrained=self.pretrain, 
                                                        in_channels=self.in_channels,
                                                        num_classes=1) 
            
        elif self.model == 'efficientnet_b5':
            self.model = models.efficientnet_b5(pretrained=self.pretrain)
            self.model.features[0][0]=nn.Conv2d(self.in_channels, 48, 
                                                kernel_size=(3, 3), 
                                                stride=(2, 2), 
                                                padding=(1, 1), 
                                                bias=False)
            self.model.classifier[1] =nn.Linear(in_features=self.model.classifier[1].in_features, 
                                                out_features=self.ouput_dim, 
                                                bias=True)
            
        elif self.model == 'efficientnet_b6':
            self.model = models.efficientnet_b6(pretrained=self.pretrain)
            self.model.features[0][0]=nn.Conv2d(self.in_channels, 56, 
                                                kernel_size=(3, 3), 
                                                stride=(2, 2), 
                                                padding=(1, 1), 
                                                bias=False)
            self.model.classifier[1] =nn.Linear(in_features=self.model.classifier[1].in_features, 
                                                out_features=self.ouput_dim, 
                                                bias=True)
            
    def forward(self, inputs):
        output = self.model(inputs)
        return output
    
    
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class RNN_Decoder(nn.Module):
    def __init__(self, config):
        
        super(RNN_Decoder, self).__init__()
        self.num_classes = config['num_classes']
        self.max_seq_len = config['max_len']
        self.num_lstm_out = config['embedding_dim']
        self.num_lstm_layers = config['num_lstm_layers']

        self.conv1_nf = config['conv1_nf']
        self.conv2_nf = config['conv2_nf']
        self.conv3_nf = config['conv3_nf']

        self.lstm_drop_p = config['lstm_drop_p']
        self.conv_drop_p = config['conv_drop_p']
        self.fc_drop_p = config['fc_drop_p']
        
        self.convDrop = nn.Dropout(self.conv_drop_p)
        self.dropout = nn.Dropout(self.fc_drop_p)
        self.gelu = nn.GELU()
        self.relu = nn.ReLU()
        
        self.lstm=torch.nn.LSTM(input_size=self.max_seq_len,
                                hidden_size=self.num_lstm_out,
                                num_layers=self.num_lstm_layers,
                                batch_first=True,
                                bidirectional=False,
                                dropout=self.lstm_drop_p)
    
        self.conv1 = nn.Conv1d(self.max_seq_len, self.conv1_nf, 8)
        self.conv2 = nn.Conv1d(self.conv1_nf, self.conv2_nf, 5)
        self.conv3 = nn.Conv1d(self.conv2_nf, self.conv3_nf, 3)

        self.bn1 = nn.BatchNorm1d(self.conv1_nf)
        self.bn2 = nn.BatchNorm1d(self.conv2_nf)
        self.bn3 = nn.BatchNorm1d(self.conv3_nf)

        self.se1 = SELayer(self.conv1_nf)  
        self.se2 = SELayer(self.conv2_nf)

        self.fc = nn.Linear(self.conv3_nf+self.num_lstm_out, self.num_lstm_out)
        self.out_layer = nn.Linear(self.num_lstm_out + self.num_lstm_out, self.num_classes)
    
    def forward(self, enc_out, x):

        x1, (ht,ct) = self.lstm(x)
        x1 = x1[:,-1,:]
        
        x2 = x.transpose(2,1)
        # x2 = self.convDrop(self.gelu(self.bn1(self.conv1(x2))))
        x2 = self.convDrop(self.gelu(self.conv1(x2)))
        x2 = self.se1(x2)
        # x2 = self.convDrop(self.gelu(self.bn2(self.conv2(x2))))
        x2 = self.convDrop(self.gelu(self.conv2(x2)))
        x2 = self.se2(x2)
        # x2 = self.convDrop(self.gelu(self.bn3(self.conv3(x2))))
        x2 = self.convDrop(self.gelu(self.conv3(x2)))
        x2 = torch.mean(x2,2)
        
        x_all = torch.cat((x1,x2),dim=1)
        x_out = self.fc(x_all)
        concat = torch.cat([enc_out, x_out], dim=1) 
        output = self.dropout(concat)
        x_output = self.gelu(self.out_layer(output))
        return x_output


class CNN2RNN(nn.Module):
    def __init__(self, config):
        super(CNN2RNN, self).__init__()
        if config['model'] == 'etc':
            self.cnn = CNNRegressor(config)
        else:
            self.cnn = CNN_Encoder(config)
        self.rnn = RNN_Decoder(config)
        self.relu = nn.ReLU()
        
    def forward(self, img, seq):
        output = self.cnn(img)
        # output = self.relu(output)
        output = self.rnn(output, seq)
        
        return output
    
