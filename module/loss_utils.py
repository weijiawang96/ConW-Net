import torch
import torch.nn as nn


class contrastiveLoss(nn.Module):
    def __init__(self, args):
        super(contrastiveLoss, self).__init__()
        self.batch_size = args.batch_size
        self.temperature = args.temperature
        self.device = args.device
        self.mask = self.mask(self.batch_size)
        self.similarity_f = nn.CosineSimilarity(dim=-1)

    def mask(self, batch_size):
        N = batch_size
        mask = torch.ones(2 * N, 2 * N)
        mask = (mask - torch.eye(2 * N)).bool()
        return mask

    def forward(self, z_i, z_j):
        N = self.batch_size

        # [2*B, D]
        z = torch.cat((z_i, z_j), dim=0)
        # [2*B, 1, D]
        z1 = z.unsqueeze(1)
        # [1, 2*B, D]
        z2 = z.unsqueeze(0)

        # [2*B, 2*B]
        sim = self.similarity_f(z1, z2) / self.temperature
        # if self.task == 'seg':
        #     sim = torch.mean(sim, -1)
        # [B, ]
        sim_i_j = torch.diag(sim, self.batch_size)
        # [B, ]
        sim_j_i = torch.diag(sim, -self.batch_size)

        # 2N samples
        # [2*B, 1]
        p_sim = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(2 * N, 1)

        # [2*B, 2*(B-1)]
        all_sim = sim[self.mask].reshape(2 * N, -1)
        # [2*B, ]
        m = torch.exp(all_sim).sum(dim=-1).reshape(-1, 1)

        # [2*B, 1]
        S = torch.exp(p_sim) / m

        C_loss = -torch.log(S).mean()

        return C_loss

