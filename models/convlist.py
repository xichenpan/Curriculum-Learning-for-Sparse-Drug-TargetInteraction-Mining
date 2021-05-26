import torch.nn as nn


class TransposeLast(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.transpose(-2, -1)


class ConvFeatureExtractionModel(nn.Module):
    def __init__(self, in_d, conv_layers, dropout: float = 0.0, conv_bias: bool = False):
        super().__init__()

        def block(n_in, n_out, kernel, stride, is_layer_norm=False, conv_bias=False):
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, kernel, stride=stride, bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv

            if is_layer_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    nn.Sequential(
                        TransposeLast(),
                        nn.LayerNorm(dim, elementwise_affine=True),
                        TransposeLast(),
                    ),
                    nn.GELU(),
                )
            else:
                return nn.Sequential(make_conv(), nn.Dropout(p=dropout), nn.GELU())

        self.conv_layers = nn.ModuleList()
        for i, cl in enumerate(conv_layers):
            (dim, kernel, stride) = cl

            self.conv_layers.append(
                block(
                    in_d,
                    dim,
                    kernel,
                    stride,
                    is_layer_norm=True,
                    conv_bias=conv_bias,
                )
            )
            in_d = dim

    def forward(self, x):
        for conv in self.conv_layers:
            x = conv(x)

        return x
