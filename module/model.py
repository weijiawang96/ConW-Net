import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib


# ================================================ PointNet ============================================================
# feature extraction encoder.
class PointNet(nn.Module):
    def __init__(self, in_dim):
        super(PointNet, self).__init__()

        self.conv1 = torch.nn.Conv1d(in_dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 64, 1)
        self.conv3 = torch.nn.Conv1d(64, 64, 1)
        self.conv4 = torch.nn.Conv1d(64, 128, 1)
        self.conv5 = torch.nn.Conv1d(128, 1024, 1)
        # self.mp1 = torch.nn.MaxPool1d(num_points)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))

        # x = self.mp1(x)  # no need for 1 point
        x = x.view(-1, 1024)
        return x


# ================================================ DGCNN ==============================================================
# This DGCNN should be correct
def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature


class DGCNN(nn.Module):
    def __init__(self, emb_dims, k=20):
        print("DGCNN_KNN =", k)
        super(DGCNN, self).__init__()
        # self.args = args
        self.k = k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))


    def forward(self, x):
        batch_size = x.size(0)

        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        return x
# ==============================================================================================================


# Points and normals proj head
class ProjHead_DGCNN_PointNet(nn.Module):
    def __init__(self, out_dim):
        super(ProjHead_DGCNN_PointNet, self).__init__()
        self.out_dim = out_dim

        # for DGCNN
        self.projector1 = nn.Sequential(
            nn.Linear(1024*2, 512, bias=False),  # 1024*2, 512 for DGCNN
            nn.BatchNorm1d(512),
            # nn.Dropout(p=0.3),
            nn.ELU(inplace=True),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.3),
            nn.ELU(inplace=True),

            nn.Linear(256, self.out_dim),
        )

        # for PoitnNet
        self.projector2 = nn.Sequential(
            nn.Linear(1024, 512, bias=False),  # 1024, 512 for pointnet
            nn.BatchNorm1d(512),
            # nn.Dropout(p=0.3),
            nn.ELU(inplace=True),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.3),
            nn.ELU(inplace=True),

            nn.Linear(256, self.out_dim),
        )

    def forward(self, h_i, h_j):

        z_i = self.projector1(h_i)  # pts
        z_j = self.projector2(h_j)  # nmls

        return z_i, z_j



class DGCNN_Weight(nn.Module):
    def __init__(self, emb_dims, k=20):
        print("DGCNN_KNN =", k)
        super(DGCNN_Weight, self).__init__()
        # self.args = args
        self.k = k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))


    def forward(self, x, weighted=False, weights=None):
        batch_size = x.size(0)
        pts_per_patch = x.size(2)

        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x)  # (B, 1024, N)

        if weighted:  # weighting switch
            for i in range(batch_size):
                x[i] = torch.mul(x[i], weights[i])  # per-batch add per-point weight. e.g. point[0](1024-dim)<-weight[0]

        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        global_feature = torch.cat((x1, x2), 1)  # cat ([32, 1024], [32, 1024])

        global_feature_repeated = global_feature.unsqueeze(2).repeat(1, 1, pts_per_patch)  # (B, 1024, 700)

        # cat global feature and local feature (unpocessed global feat)
        cat = torch.cat((global_feature_repeated, x), 1)  # repeated global feat + local feat(B, 2048, 700)

        return global_feature, cat


# Weight Regression layer
class Weight_Regressor(nn.Module):
    def __init__(self, num_points):
        super(Weight_Regressor, self).__init__()
        self.num_points = num_points

        self.conv1 = nn.Conv1d(3072, 1024, 1)
        self.conv2 = nn.Conv1d(1024, 512, 1)
        self.conv3 = nn.Conv1d(512, 256, 1)
        self.conv4 = nn.Conv1d(256, 128, 1)

        self.conv_weight = nn.Conv1d(128, 1, 1)  # weight head

        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(128)


    def forward(self, encoded_pts_global_local):

        x = encoded_pts_global_local
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        # per point weight
        weights = 0.01 + torch.sigmoid(self.conv_weight(x))  # (B, 1, 700)

        return weights.squeeze()


# Normal Regressor
class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 3)

        self.bn_fc1 = nn.BatchNorm1d(1024)
        self.bn_fc2 = nn.BatchNorm1d(512)
        self.bn_fc3 = nn.BatchNorm1d(256)

        self.dropout_1 = nn.Dropout(0.3)
        self.dropout_2 = nn.Dropout(0.3)
        self.dropout_3 = nn.Dropout(0.3)

    def forward(self, h):
        # pass in global feature

        h = F.relu(self.bn_fc1(self.fc1(h)))
        h = self.dropout_1(h)

        h = F.relu(self.bn_fc2(self.fc2(h)))
        h = self.dropout_2(h)

        h = F.relu(self.bn_fc3(self.fc3(h)))
        h = self.dropout_3(h)

        pred_normal = torch.tanh(self.fc4(h))  # (B, 3)

        return pred_normal


