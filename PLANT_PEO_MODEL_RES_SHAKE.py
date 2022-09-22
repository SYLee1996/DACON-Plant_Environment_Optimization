import math
import torch
import torch.nn.functional as F

from torch import nn
from torch.autograd import Variable


class ShakeShake(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1, x2, training=True):
        if training:
            alpha = torch.cuda.FloatTensor(x1.size(0)).uniform_()
            alpha = alpha.view(alpha.size(0), 1, 1, 1).expand_as(x1)
        else:
            alpha = 0.5
        return alpha * x1 + (1 - alpha) * x2

    @staticmethod
    def backward(ctx, grad_output):
        beta = torch.cuda.FloatTensor(grad_output.size(0)).uniform_()
        beta = beta.view(beta.size(0), 1, 1, 1).expand_as(grad_output)
        beta = Variable(beta)

        return beta * grad_output, (1 - beta) * grad_output, None


class Shortcut(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super(Shortcut, self).__init__()
        self.stride = stride
        self.conv1 = nn.Conv2d(in_ch, out_ch // 2, 1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(in_ch, out_ch // 2, 1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        h = F.relu(x)

        h1 = F.avg_pool2d(h, 1, self.stride)
        h1 = self.conv1(h1)

        h2 = F.avg_pool2d(F.pad(h, (-1, 1, -1, 1)), 1, self.stride)
        h2 = self.conv2(h2)

        h = torch.cat((h1, h2), 1)
        return self.bn(h)


class ShakeBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(ShakeBlock, self).__init__()
        self.equal_io = in_ch == out_ch
        self.shortcut = self.equal_io and None or Shortcut(in_ch, out_ch, stride=stride)

        self.branch1 = self._make_branch(in_ch, out_ch, stride)
        self.branch2 = self._make_branch(in_ch, out_ch, stride)

    def forward(self, x):
        h1 = self.branch1(x)
        h2 = self.branch2(x)
        h = ShakeShake.apply(h1, h2, self.training)
        h0 = x if self.equal_io else self.shortcut(x)
        return h + h0

    def _make_branch(self, in_ch, out_ch, stride=1):
        return nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(in_ch, out_ch, 3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_ch))


class ShakeResNet(nn.Module):
    def __init__(self, config):
        super(ShakeResNet, self).__init__()
        self.in_channels = config['in_channels']
        n_units = (config['depth'] - 2) / 6

        in_chs = [16, config['w_base'], config['w_base'] * 2, config['w_base'] * 4]
        self.in_chs = in_chs

        self.c_in = nn.Conv2d(self.in_channels, in_chs[0], 3, padding=1)
        self.layer1 = self._make_layer(n_units, in_chs[0], in_chs[1])
        self.layer2 = self._make_layer(n_units, in_chs[1], in_chs[2], 2)
        self.layer3 = self._make_layer(n_units, in_chs[2], in_chs[3], 2)
        # self.fc_out = nn.Linear(in_chs[3], class_num)
        self.fc_out = nn.Linear(int((512 / 2 / 16)**2 *in_chs[3]) , config['embedding_dim'])

        # Initialize paramters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        # print('x: ', x.shape)
        h = self.c_in(x)
        # print('h: ', h.shape)
        h = self.layer1(h)
        # print('layer1: ', h.shape)
        h = self.layer2(h)
        # print('layer2: ', h.shape)
        h = self.layer3(h)
        # print('layer3: ', h.shape)
        h = F.relu(h)
        h = F.avg_pool2d(h, 8)
        # print('avg_pool2d: ', h.shape)
        # h = h.view(-1, self.in_chs[0]*self.in_chs[3]*self.in_chs[3])
        # h = h.view(-1, self.in_chs[3])
        h = torch.flatten(h, start_dim=1)
        h = self.fc_out(h)
        
        return h

    def _make_layer(self, n_units, in_ch, out_ch, stride=1):
        layers = []
        for i in range(int(n_units)):
            layers.append(ShakeBlock(in_ch, out_ch, stride=stride))
            in_ch, stride = out_ch, 1
        return nn.Sequential(*layers)
    
    
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
        x2 = self.convDrop(self.gelu(self.conv1(x2)))
        x2 = self.se1(x2)
        x2 = self.convDrop(self.gelu(self.conv2(x2)))
        x2 = self.se2(x2)
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
        self.cnn = ShakeResNet(config)
        self.rnn = RNN_Decoder(config)
        self.relu = nn.ReLU()
        
    def forward(self, img, seq):
        output = self.cnn(img)
        output = self.rnn(output, seq)
        
        return output
    
