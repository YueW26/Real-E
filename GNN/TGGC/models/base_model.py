import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, units, stack_cnt, time_step, multi_layer, horizon=1, dropout_rate=0.5, leaky_rate=0.2, device='cpu'):
        super(Model, self).__init__()
        self.unit = units
        self.stack_cnt = stack_cnt
        self.alpha = leaky_rate
        self.time_step = time_step
        self.horizon = horizon
        self.freq_sample = 6
        self.poly_order = 3
        self.alpha_gegen = 1.0

        self.weight_key = nn.Parameter(torch.zeros(size=(self.unit, 1)))
        nn.init.xavier_uniform_(self.weight_key.data, gain=1.414)
        self.weight_query = nn.Parameter(torch.zeros(size=(self.unit, 1)))
        nn.init.xavier_uniform_(self.weight_query.data, gain=1.414)

        self.fc = nn.Sequential(
            nn.Linear(int(self.time_step), int(self.time_step)),
            nn.LeakyReLU(),
            nn.Linear(int(self.time_step), self.horizon),
        )
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.to(device)

    def get_laplacian(self, graph, normalize=True):
        if normalize:
            D = torch.diag(torch.sum(graph, dim=-1) ** (-1 / 2))
            L = torch.eye(graph.size(0), device=graph.device) - torch.mm(torch.mm(D, graph), D)
        else:
            D = torch.diag(torch.sum(graph, dim=-1))
            L = D - graph
        return L

    def gegenbauer_polynomial(self, L, K=3, alpha=1.0):
        N = L.size(0)
        L_list = [torch.eye(N, device=L.device), 2 * alpha * (torch.eye(N, device=L.device) - L)]
        for k in range(2, K + 1):
            P_k = ((2 * (k + alpha - 1)) * (torch.eye(N, device=L.device) - L) @ L_list[-1] -
                   (k + 2 * alpha - 2) * L_list[-2]) / k
            L_list.append(P_k)
        return torch.stack(L_list, dim=0)

    def temporal_fdm(self, X, W=None):
        B, C, N, T = X.shape
        X = X.view(B * C * N, T)
        X_freq = torch.fft.fft(X, dim=1)
        X_freq_sampled = X_freq[:, :self.freq_sample]
        if W is None:
            W = torch.randn(self.freq_sample, self.freq_sample, device=X.device)
        X_filtered = X_freq_sampled @ W
        X_ifft = torch.fft.ifft(X_filtered, n=T, dim=1).real
        return X_ifft.view(B, C, N, T)

    def latent_correlation_layer(self, x):
        input = x.permute(2, 0, 1)
        input = input.permute(1, 0, 2)
        attention = self.self_graph_attention(input)
        attention = torch.mean(attention, dim=0)
        degree = torch.sum(attention, dim=1)
        attention = 0.5 * (attention + attention.T)
        degree_l = torch.diag(degree)
        diagonal_degree_hat = torch.diag(1 / (torch.sqrt(degree) + 1e-7))
        laplacian = torch.matmul(diagonal_degree_hat, torch.matmul(degree_l - attention, diagonal_degree_hat))
        mul_L = self.gegenbauer_polynomial(laplacian, K=self.poly_order, alpha=self.alpha_gegen)
        return mul_L, attention

    def self_graph_attention(self, input):
        input = input.permute(0, 2, 1).contiguous()
        bat, N, fea = input.size()
        key = torch.matmul(input, self.weight_key)
        query = torch.matmul(input, self.weight_query)
        data = key.repeat(1, 1, N).view(bat, N * N, 1) + query.repeat(1, N, 1)
        data = data.squeeze(2).view(bat, N, -1)
        data = self.leakyrelu(data)
        attention = F.softmax(data, dim=2)
        attention = self.dropout(attention)
        return attention

    def forward(self, x):
        mul_L, attention = self.latent_correlation_layer(x)
        X = x.unsqueeze(1).permute(0, 1, 3, 2).contiguous()
        X = torch.einsum('knm,bcmf->bkcnf', mul_L, X)  # GSF
        X = self.temporal_fdm(X)  # TSF
        X = X.mean(1)  # Reduce channel dimension
        forecast = self.fc(X)
        if forecast.size()[-1] == 1:
            return forecast.unsqueeze(1).squeeze(-1), attention
        else:
            return forecast.permute(0, 2, 1).contiguous(), attention