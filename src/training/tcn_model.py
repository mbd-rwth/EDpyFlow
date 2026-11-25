import torch
import torch.nn as nn


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()

        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation
        )

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        if in_channels != out_channels:
            self.residual = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=1
            )
        else:
            self.residual = nn.Identity()

        self.padding = padding

    def forward(self, x):
        residual = self.residual(x)

        out = self.conv1(x)

        if self.padding > 0:
            out = out[:, :, :-self.padding]

        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)

        if self.padding > 0:
            out = out[:, :, :-self.padding]

        out = self.relu(out)
        out = self.dropout(out)

        return self.relu(out + residual)


class TCNSurrogate(nn.Module):
    def __init__(
        self,
        weather_features=6,
        static_features=11,
        channels=(16, 32, 32),
        kernel_size=3,
        dropout=0.2
    ):
        super().__init__()

        blocks = []
        in_channels = weather_features

        for i, out_channels in enumerate(channels):
            blocks.append(
                TemporalBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=2 ** i,
                    dropout=dropout
                )
            )

            in_channels = out_channels

        self.tcn = nn.Sequential(*blocks)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.static_net = nn.Sequential(
            nn.Linear(static_features, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU()
        )

        self.regressor = nn.Sequential(
            nn.Linear(channels[-1] + 16, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, weather, static):
        # weather initially:
        # [batch, time, features]

        weather = weather.transpose(1, 2)

        # now:
        # [batch, features, time]

        temporal_features = self.tcn(weather)

        temporal_features = self.pool(
            temporal_features
        ).squeeze(-1)

        static_features = self.static_net(static)

        combined = torch.cat(
            [temporal_features, static_features],
            dim=1
        )

        prediction = self.regressor(combined)

        return prediction.squeeze(1)