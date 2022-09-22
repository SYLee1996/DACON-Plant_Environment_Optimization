import timm 
import torch
from torch import nn



class CNN_Encoder(torch.nn.Module):
    def __init__(self, config):
        super(CNN_Encoder, self).__init__()
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
            nn.ReLU())

    def forward(self, x):
        # Simple CNN Model (Batch, 3, 128, 128 -> Batch, 64, 7, 7)
        
        x = self.layer1(x)                    # (Batch, 3, 128, 128)
        x = self.layer2(x)                    # (Batch, 8, 64, 64)
        x = self.layer3(x)                    # (Batch, 16, 32, 32)
        x = self.layer4(x)                    # (Batch, 32, 16, 16)
        x = torch.flatten(x, start_dim=1)     # (Batch, 64, 7, 7) -> Flatten (Batch, 64*7*7(=3136))
        out = self.regressor(x)
        return out
    
    
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
    def __init__(self, config, 
                    conv1_nf=128, conv2_nf=256, conv3_nf=128,
                    lstm_drop_p=0.8, fc_drop_p=0.1):
        
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

        self.se1 = SELayer(self.conv1_nf)  # ex 128
        self.se2 = SELayer(self.conv2_nf)  # ex 256

        self.fc = nn.Linear(self.conv3_nf+self.num_lstm_out, self.num_lstm_out)
        self.out_layer = nn.Linear(self.num_lstm_out + self.num_lstm_out, self.num_classes)
    
    def forward(self, enc_out, x):

        x1, (ht,ct) = self.lstm(x)
        # print('x1 shape: ', x1.shape)
        # x1, _ = nn.utils.rnn.pad_packed_sequence(x1, batch_first=True, 
        #                                         padding_value=0.0)
        x1 = x1[:,-1,:]
        
        x2 = x.transpose(2,1)
        x2 = self.convDrop(self.gelu(self.bn1(self.conv1(x2))))
        x2 = self.se1(x2)
        x2 = self.convDrop(self.gelu(self.bn2(self.conv2(x2))))
        x2 = self.se2(x2)
        x2 = self.convDrop(self.gelu(self.bn3(self.conv3(x2))))
        x2 = torch.mean(x2,2)
        
        x_all = torch.cat((x1,x2),dim=1)
        x_out = self.fc(x_all)
        concat = torch.cat([enc_out, x_out], dim=1) 
        output = self.dropout(concat)
        x_output = self.out_layer(output)

        return x_output

    
class CNN2RNN(nn.Module):
    def __init__(self, config):
        super(CNN2RNN, self).__init__()
        self.cnn = CNN_Encoder(config)
        self.rnn = RNN_Decoder(config)
        
    def forward(self, img, seq):
        output = self.cnn(img)
        output = self.rnn(output, seq)
        
        return output