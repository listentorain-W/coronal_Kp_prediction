from torch import nn


class FeedForwardNN(nn.Module):
    def __init__(self, input_size=94, hidden_size=20, output_size=1):
        super(FeedForwardNN, self).__init__()
        self.FF_NN = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Sigmoid(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        y = self.FF_NN(x)
        return y