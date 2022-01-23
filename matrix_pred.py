import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from utils.metrics import FocalLoss, dice_loss

class conv_block_nested(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, ksize):
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=ksize, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=ksize, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        identity = x
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x + identity)
        return output

class predMatrixLinear(nn.Module):
    def __init__(self, num_layer):
        super(predMatrix, self).__init__()
        self.layers = nn.ModuleList()

        for i in range(num_layer):
            if i == 0:
                layer = nn.Linear(25,100,bias=True)#conv_block_nested(2, 16, 16)
            else:
                layer = nn.Linear(100,100,bias=True)#conv_block_nested(16, 16, 16)
            self.layers.append(layer)
        self.xcor_conv = nn.Linear(100,25,bias=True)#nn.Conv2d(100, 25, kernel_size=3, padding=1, bias=True)
        #self.ycor_conv = nn.Conv2d(16, 5, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        out = x.view(-1)
        for layer in self.layers:
            out = layer(out)
        #xcor = F.softmax(self.xcor_conv(out), dim=1)
        #ycor = F.softmax(self.xcor_conv(out), dim=1)
        #return xcor, ycor
        return self.xcor_conv(out).view(5,5)


class predMatrix(nn.Module):
    def __init__(self, num_layer,input_size):
        super(predMatrix, self).__init__()
        self.layers = nn.ModuleList()

        for i in range(num_layer):
            if i == 0:
                layer = nn.Conv2d(2, input_size, kernel_size=input_size, bias=True)
            else:
                layer = nn.Conv2d(input_size, input_size, kernel_size=1, bias=True)
            self.layers.append(layer)
        self.xcor_conv = nn.Conv2d(input_size, input_size*input_size, kernel_size=1, bias=True)

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return self.xcor_conv(out).view(5,5)

if __name__ == '__main__':
    x = torch.randn([1,1,5, 5])
    y = torch.from_numpy(x.numpy()[..., ::-1].copy())
    x.requires_grad_(True)
    y.requires_grad_(True)
    #xcor_gt = torch.tensor([[]])
    data = torch.cat([x,y],dim=1)
    data.requires_grad_(True)
    print(data.shape)
    model = predMatrix(1,5)
    model.train()
    criteria = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=0.01)
    print(x)
    print(y)
    for epoch in range(50):
        for step in range(10):
            # Zero the gradient
            optimizer.zero_grad()

            out = model(data)
            # xcor_value, xcor_label = xcor.topk(1, dim=1)
            # ycor_value, ycor_label = ycor.topk(1, dim=1)
            # x2y = x[:, :, xcor_label.view(5,5), ycor_label.view(5,5)]
            loss = criteria(out,y)
            loss.backward()
            optimizer.step()
        #print('Epoch {}: {}'.format(epoch,loss.item()))
    print(out)
    print(y)
