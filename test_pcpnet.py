"""
Test set, contrast finetune + L_cos + alpha * L_weight
"""

import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import datetime

from pathlib import Path
import numpy as np
from tqdm import tqdm
import importlib

from dataloader import PointcloudPatchDataset, RandomPointcloudPatchSampler
from module import Regressor, DGCNN_Weight, Weight_Regressor


def setup_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def config_params():
    parser = argparse.ArgumentParser(description='Downstream Parameters')

    ## dataset (from my pcrnet trainer)
    parser.add_argument('--indir', type=str, default="./data/pclouds",
                        help='the data path')
    parser.add_argument('--outdir', type=str, default="./data/Result/",
                        help='the data path')

    parser.add_argument('--patch_radius', type=float, default=[0.05], nargs='+',
                        help='patch radius in multiples of the shape\'s bounding box diagonal, multiple values for multi-scale.')
    parser.add_argument('--points_per_patch', type=int, default=700, help='knn')
    parser.add_argument('--patch_point_count_std', type=float, default=0,
                        help='standard deviation of the number of points in a patch')
    parser.add_argument('--identical_epochs', type=int, default=False,
                        help='use same patches in each epoch, mainly for debugging')
    parser.add_argument('--use_pca', type=int, default=True,
                        help='use pca on point clouds, must be true for jet fit type')
    parser.add_argument('--patch_center', type=str, default='point', help='center patch at...\n'
                                                                          'point: center point\n'
                                                                          'mean: patch mean')
    parser.add_argument('--point_tuple', type=int, default=1,
                        help='use n-tuples of points as input instead of single points')
    parser.add_argument('--cache_capacity', type=int, default=100,
                        help='Max. number of dataset elements (usually shapes) to hold in the cache at the same time.')
    parser.add_argument('--neighbor_search', type=str, default='k', help='[k | r] for k nearest and radius')


    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40], help='training on ModelNet10/40')
    # parser.add_argument('--num_points', type=int, default=1024, help='input batch size')
    parser.add_argument('--use_normals', action='store_true', default=True, help='use normals')
    parser.add_argument('--augmentation', default='None',
                        help='augmentation name [default: dual_jitter_all_with_normal]')

    parser.add_argument('--epochs', default=200, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--batch_size', default=16, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--lr', type=int, default=1e-3, help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=1e-6, help='decay rate [default: 1e-4]')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--input_dim', default=3, type=int, help='Input dim for training')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--seed', type=int, default=1000, help='input manual_seed')
    parser.add_argument('--cuda', type=str, default="0", help="cuda")

    parser.add_argument('--saved_path', default='trained_models/',
                        help='the path to save training logs and checkpoints')

    parser.add_argument('--sym_op', type=str, default='max', help='symmetry operation')
    parser.add_argument('--jet_order', type=int, default=2, help='jet polynomial fit order')
    parser.add_argument('--weight_mode', type=str, default="sigmoid",
                        help='which function to use on the weight output: softmax, tanh, sigmoid')
    parser.add_argument('--use_consistency', type=int, default=True, help='flag to use consistency loss')
    parser.add_argument('--outputs', type=str, nargs='+', default=['unoriented_normals', 'neighbor_normals'],
                        help='outputs of the network, a list with elements of:\n'
                             'unoriented_normals: unoriented (flip-invariant) point normals\n'
                             'oriented_normals: oriented point normals\n'
                             'max_curvature: maximum curvature\n'
                             'min_curvature: mininum curvature')

    # DGCNN Knn
    parser.add_argument('--dgcnn_knn', type=int, default=20, help="knn of DGCNN backbone")

    # Weighting coefficient for L_weight
    parser.add_argument('--L_weight_coef', type=float, default=1.0, help='')

    args = parser.parse_args()
    args.device = torch.device("cuda:" + args.cuda if torch.cuda.is_available() else "cpu")
    return args


# My get dataloaders func. useful
def get_data_loaders(opt, target_features, shapename):
    # create train and test dataset loaders
    test_dataset = PointcloudPatchDataset(
        root=opt.indir,
        shape_list_filename=shapename,
        patch_radius=opt.patch_radius,
        points_per_patch=opt.points_per_patch,
        patch_features=target_features,
        point_count_std=opt.patch_point_count_std,
        seed=opt.seed,
        identical_epochs=opt.identical_epochs,
        use_pca=opt.use_pca,
        center=opt.patch_center,
        point_tuple=opt.point_tuple,
        cache_capacity=opt.cache_capacity,
        neighbor_search_method=opt.neighbor_search,
        train_state='test')



    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        # sampler=train_datasampler,
        batch_size=opt.batch_size,
        num_workers=int(opt.num_workers))

    return test_dataloader, test_dataset


def main():

    args = config_params()

    with open(os.path.join(args.indir, 'testset_synthetic_noise.txt'), 'r') as f:  # test
        shape_names = f.readlines()
    shape_names = [x.strip() for x in shape_names]
    shape_names = list(filter(None, shape_names))
    print("shape_names", shape_names)  # OK

    for shape_id, shapename in enumerate(shape_names):
        setup_seed(args.seed)
        if not os.path.exists(args.saved_path):
            os.makedirs(args.saved_path)

        output_directory = args.outdir
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        downstream_checkpoints_path = os.path.join(args.saved_path,
            'checkpoints_downstream_finetune_{}weight_cont_Lcos_{}Lweight_{}pts_dgcnn_knn{}').format(
            args.L_weight_coef, args.L_weight_coef, args.points_per_patch, args.dgcnn_knn)

        target_features = []
        test_dataloader, test_dataset = get_data_loaders(args, target_features, shapename)

        '''MODEL LOADING'''
        pretrained_path = os.path.join(
            downstream_checkpoints_path, "train_epoch_best.pth"  # best DGCNN model for upstream
        )
        checkpoint = torch.load(pretrained_path, map_location=args.device.type)


        pt_encoder = DGCNN_Weight(emb_dims=1024, k=args.dgcnn_knn)  # point encoder structure
        pt_encoder.load_state_dict(checkpoint['pt_encoder_state_dict'])
        pt_encoder = pt_encoder.to(args.device)
        pt_encoder.eval()

        Weighting_Net = Weight_Regressor(args.points_per_patch)
        Weighting_Net.load_state_dict(checkpoint['weighting_state_dict'])
        Weighting_Net = Weighting_Net.to(args.device)
        Weighting_Net.eval()

        Nml_Regression_Net = Regressor()
        Nml_Regression_Net.load_state_dict(checkpoint['nml_regressor_state_dict'])
        Nml_Regression_Net = Nml_Regression_Net.to(args.device)
        Nml_Regression_Net.eval()

        pred_normal = np.empty((0, 3), dtype='float32')

        nml_for_flip = np.loadtxt(os.path.join(args.indir, shapename + '.normals'))  # old ver

        print(len(test_dataloader))

        for batch_ind, combo in enumerate(test_dataloader):
            pts, invmat = combo

            pts = pts.float().cuda()
            invmat = invmat.float().cuda()

            _, encoded_pts_global_local = pt_encoder(pts.permute(0, 2, 1).contiguous(), False, None)

            weights = \
                Weighting_Net(encoded_pts_global_local)

            weights_normlized = torch.nn.functional.normalize(weights, p=2.0, dim=1)

            encoded_pts_global_weighted, _ = pt_encoder(pts.permute(0, 2, 1).contiguous(), True, weights_normlized.data)

            pred_normals = Nml_Regression_Net(encoded_pts_global_weighted)
            pred_nml = F.normalize(pred_normals, p=2, dim=1)
            pred_nml = pred_nml.unsqueeze(2)

            # map pred normal back by R^-1
            predict = torch.bmm(invmat, pred_nml)
            predict = predict.data.cpu().numpy()

            predict = predict.squeeze(2)

            # load normals for flip
            gt_normal_batch = nml_for_flip[batch_ind * args.batch_size:(batch_ind + 1) * args.batch_size]
            for i in range(len(predict)):
                dot_prod = np.dot(predict[i], gt_normal_batch[i])
                if dot_prod < 0:
                    predict[i] = -predict[i]

            pred_normal = np.append(pred_normal, predict, axis=0)


        points = np.loadtxt(os.path.join(args.indir, shapename + '.xyz'))
        np.savetxt(os.path.join(output_directory, shapename + '_pred.xyz'), np.hstack((points, pred_normal)))




if __name__ == '__main__':
    main()

