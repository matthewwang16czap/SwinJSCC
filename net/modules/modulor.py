import torch.nn as nn


class AdaptiveModulator(nn.Module):
    def __init__(self, M):
        super(AdaptiveModulator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, M),
            nn.ReLU(),
            nn.Linear(M, M),
            nn.ReLU(),
            nn.Linear(M, M),
            nn.Sigmoid(),
        )

    def forward(self, cond):
        return self.fc(cond)


class ModulatorLayer(nn.Module):
    def __init__(self, in_dim, hidden_dim, layer_num=7):
        super(ModulatorLayer, self).__init__()
        self.layer_num = layer_num
        self.sm_list = nn.ModuleList(
            [nn.Linear(in_dim, hidden_dim)]
            + [nn.Linear(hidden_dim, hidden_dim) for _ in range(layer_num - 1)]
            + [nn.Linear(hidden_dim, in_dim)]
        )
        self.bm_list = nn.ModuleList(
            [AdaptiveModulator(hidden_dim) for _ in range(layer_num)]
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, cond):
        temp = self.sm_list[0](x.detach())
        for i in range(self.layer_num):
            bm = self.bm_list[i](cond)
            temp = temp * bm
            temp = self.sm_list[i + 1](temp)
        return x * self.sigmoid(temp)
