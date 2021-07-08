from ..models.inception import Inception, InceptionBlock
import torch.nn as nn





class Flatten(nn.Module):
    def __init__(self, out_features):
        super(Flatten, self).__init__()
        self.output_dim = out_features

    def forward(self, x):
        return x.view(-1, self.output_dim)
    
class Reshape(nn.Module):
    def __init__(self, out_shape):
        super(Reshape, self).__init__()
        self.out_shape = out_shape

    def forward(self, x):
        return x.view(-1, *self.out_shape)

def build_model(input_size, n_classes):
    InceptionTime = nn.Sequential(
                        Reshape(out_shape=(1, input_size)),
                        InceptionBlock(
                            in_channels=1, 
                            n_filters=32, 
                            kernel_sizes=[9, 19, 39],
                            bottleneck_channels=32,
                            use_residual=True,
                            activation=nn.ReLU()
                        ),
                        InceptionBlock(
                            in_channels=32*4, 
                            n_filters=32, 
                            kernel_sizes=[9, 19, 39],
                            bottleneck_channels=32,
                            use_residual=True,
                            activation=nn.ReLU()
                        ),
                        nn.AdaptiveAvgPool1d(output_size=1),
                        Flatten(out_features=32*4*1),
                        nn.Linear(in_features=4*32*1, out_features=n_classes)
            )
    return InceptionTime
