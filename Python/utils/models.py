from torch import optim, flatten, save, cat
from torch.nn import Module, Sequential, Conv2d, Linear, BatchNorm2d, ReLU, ELU, GELU, CrossEntropyLoss, ModuleList, GroupNorm
from torch.nn.functional import max_pool2d, adaptive_avg_pool2d, softmax, log_softmax, dropout, relu

class MyContextNet(Module):
    def __init__(self, input_channels, final_channels, c=16, growths=[2,2,4,4,8,8,16,16],
                 dilations=[1,1,2,2,3,3,3,1], strides=[1,2,1,2,1,2,1,1], additional_features=None, dropout=0.25):
        super(MyContextNet, self).__init__()
        self.blocks = []
        self.dropout = dropout
        for growth, dilation, stride in zip(growths, dilations, strides):
            out_channels = c * growth
            self.blocks.append(MyContextNet._conv_block(input_channels, out_channels, c//2, dilation=dilation, stride=stride))
            input_channels = out_channels

        if additional_features is not None:
            self.intermediate = Linear(out_channels + additional_features, out_channels // 2)
            self.final = Linear(out_channels // 2, final_channels)
        else:
            self.intermediate = None
            self.final = Linear(out_channels, final_channels)
        self.blocks = ModuleList(self.blocks)

    def _conv_block(inp, out, g_size, dilation=1, stride=1, kernel=3):
        padding = (kernel-1) * dilation // 2
        block = Sequential(
            Conv2d(inp, out, kernel, padding=padding, stride=stride, dilation=dilation),
            GroupNorm(g_size, out),
            ReLU())
        return block

    def forward(self, x, add_features=None):
        for l in self.blocks:
            x = l(x)
        x = adaptive_avg_pool2d(x, 1)
        x = flatten(x, start_dim=1)
        if not hasattr(self, 'intermediate') or self.intermediate is None:
            x = dropout(x, self.dropout, self.training)
            x = self.final(x)
        else:
            x = cat([x, add_features], dim=1)
            x = dropout(x, self.dropout, self.training)
            x = self.intermediate(x)
            x = relu(x)
            x = self.final(x)
        return log_softmax(x, dim=-1)
