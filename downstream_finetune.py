"""
L_cos + L_weight + L_cont(pre-trained)
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
import logging # 处理日志的模块
from pathlib import Path
import numpy as np
from tqdm import tqdm
import importlib
import itertools

from dataloader import PointcloudPatchDataset, RandomPointcloudPatchSampler
from module import Regressor, DGCNN_Weight, Weight_Regressor
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

import time
import datetime


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
    parser.add_argument('--trainset', type=str, default='trainingset_whitenoise.txt', help='training set file name')
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
    parser.add_argument('--sym_op', type=str, default='max', help='symmetry operation')
    parser.add_argument('--point_tuple', type=int, default=1,
                        help='use n-tuples of points as input instead of single points')
    parser.add_argument('--cache_capacity', type=int, default=100,
                        help='Max. number of dataset elements (usually shapes) to hold in the cache at the same time.')
    parser.add_argument('--neighbor_search', type=str, default='k', help='[k | r] for k nearest and radius')


    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40], help='training on ModelNet10/40')
    parser.add_argument('--use_normals', action='store_true', default=True, help='use normals')
    parser.add_argument('--augmentation', default='None',
                        help='augmentation name [default: dual_jitter_all_with_normal]')

    parser.add_argument('--model_path', type=str, default='./trained_models/', help='The pretrained model path')
    parser.add_argument('--epochs', default=100, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--batch_size', default=32, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--lr', type=int, default=1e-3, help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=1e-6, help='decay rate [default: 1e-4]')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--input_dim', default=3, type=int, help='Input dim for training')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--seed', type=int, default=1000, help='input manual_seed')
    parser.add_argument('--cuda', type=str, default="0", help="cuda")
    parser.add_argument('--patches_per_shape', type=int, default=1000,
                        help='number of patches sampled from each shape in an epoch')
    parser.add_argument('--training_order', type=str, default='random',
                        help='order in which the training patches are presented:\n'
                             'random: fully random over the entire dataset (the set of all patches is permuted)\n'
                             'random_shape_consecutive: random over the entire dataset, but patches of a shape remain '
                             'consecutive (shapes and patches inside a shape are permuted)')

    parser.add_argument('--saved_path', default='trained_models/',
                        help='the path to save training logs and checkpoints')

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
def get_data_loaders(opt, target_features):
    # create train and test dataset loaders
    train_dataset = PointcloudPatchDataset(
        root=opt.indir,
        shape_list_filename=opt.trainset,
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
        train_state='normal_est')

    if opt.training_order == 'random':  # Yes, it's random.
        train_datasampler = RandomPointcloudPatchSampler(
            train_dataset,
            patches_per_shape=opt.patches_per_shape,
            seed=opt.seed,
            identical_epochs=opt.identical_epochs)
    # elif opt.training_order == 'random_shape_consecutive':
    #     train_datasampler = SequentialShapeRandomPointcloudPatchSampler(
    #         train_dataset,
    #         patches_per_shape=opt.patches_per_shape,
    #         seed=opt.seed,
    #         identical_epochs=opt.identical_epochs)
    else:
        raise ValueError('Unknown training order: %s' % (opt.training_order))

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=train_datasampler,
        batch_size=opt.batch_size,
        num_workers=int(opt.num_workers))

    return train_dataloader, train_dataset, train_datasampler


def compute_loss(pred_normals, gt_normals, point_weights_normalized, gt_neighbor_normals, bias, args):

    # L_cos ==> central normal angular loss (ctr_cosine_loss)
    cosine_similarity = 1 - (pred_normals * gt_normals).sum(1).pow(2)
    ctr_cosine_loss = torch.mean(cosine_similarity)  # take average

    # L_weight
    gt_ctr_normals_repeat = gt_normals.unsqueeze(1).repeat(1, point_weights_normalized.size(1), 1)
    # Hadamard product
    neighbor_cosine_loss = (gt_ctr_normals_repeat * gt_neighbor_normals).sum(-1).pow(2)  # smaller angle, bigger contribution

    neighbor_cosine_loss_norm = torch.nn.functional.normalize(neighbor_cosine_loss, p=2.0, dim=1)

    # calculate MSE of neighbour angles and predicted weights
    consistency_loss = nn.MSELoss(reduction='none')
    consistency_loss = torch.sum(consistency_loss(neighbor_cosine_loss_norm, point_weights_normalized), dim=1)
    L_weight = torch.mean(consistency_loss)  # take average value

    # final loss = L_cos + alpha * L_weight
    loss = ctr_cosine_loss + args.L_weight_coef * L_weight

    return loss


def main():
    start_time = time.time()
    args = config_params()

    setup_seed(args.seed)
    if not os.path.exists(args.saved_path):
        os.makedirs(args.saved_path)

    checkpoints_path = os.path.join(args.saved_path, 'checkpoints_downstream_finetune_{}weight_cont_Lcos_{}Lweight_{}pts_dgcnn_knn{}').format(
        args.L_weight_coef, args.L_weight_coef, args.points_per_patch, args.dgcnn_knn)
    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)

    target_features = []
    train_dataloader, train_dataset, train_datasampler = get_data_loaders(args, target_features)


    pretrained_path = os.path.join(
        args.model_path, "checkpoints_upstream_{}weighted_{}pts_dgcnn_knn{}/train_epoch_best.pth".format(
            args.L_weight_coef, args.points_per_patch, args.dgcnn_knn)
    )

    checkpoint = torch.load(pretrained_path, map_location=args.device.type)  # upstream trained ckpt

    # Point patch encoder & load weights
    pt_encoder = DGCNN_Weight(emb_dims=1024, k=args.dgcnn_knn)  # point encoder structure
    pt_encoder.load_state_dict(checkpoint['pt_encoder_state_dict'])
    pt_encoder = pt_encoder.to(args.device)

    # for weighting & load weights
    Weighting_Net = Weight_Regressor(args.points_per_patch)
    Weighting_Net.load_state_dict(checkpoint['weighting_state_dict'])
    Weighting_Net = Weighting_Net.to(args.device)

    # for normal regression
    Nml_Regression_Net = Regressor().to(args.device)

    optimizer = torch.optim.Adam(
        itertools.chain(pt_encoder.parameters(), Weighting_Net.parameters(), Nml_Regression_Net.parameters()),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.decay_rate
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_loss = float("inf")
    for epoch in range(args.epochs):
        loss_epoch = 0
        pt_encoder.train()
        Weighting_Net.train()
        Nml_Regression_Net.train()

        for pts, ctr_gt_nml, patch_gt_nmls, _ in tqdm(train_dataloader):

            optimizer.zero_grad()
            pts = pts.to(dtype=torch.float).cuda()
            ctr_gt_nml = ctr_gt_nml.to(dtype=torch.float).cuda()
            patch_gt_nmls = patch_gt_nmls.to(dtype=torch.float).cuda()

            _, encoded_pts_global_local = pt_encoder(pts.permute(0, 2, 1).contiguous(), False, None)

            # Get weights
            weights = Weighting_Net(encoded_pts_global_local)

            # Normalise weights.
            weights_normlized = torch.nn.functional.normalize(weights, p=2.0, dim=1)

            # weighted normal regression.
            encoded_pts_global_weighted, _ = pt_encoder(pts.permute(0, 2, 1).contiguous(), True, weights_normlized.data)

            pred_normals = Nml_Regression_Net(encoded_pts_global_weighted)

            # normalise predicted normals
            pred_normals = F.normalize(pred_normals, p=2, dim=1)

            loss = compute_loss(
                pred_normals=pred_normals,
                gt_normals=ctr_gt_nml,
                point_weights_normalized=weights_normlized,
                gt_neighbor_normals=patch_gt_nmls,
                bias=None,
                args=args
            )

            loss.backward()
            optimizer.step()
            loss_epoch += loss.item()

        loss = round(loss_epoch / len(train_dataloader), 4)
        scheduler.step()
        print("Down Loss :", loss)

        # Note down the loss
        f = open("[Downstream-finetune_{}weight_cont-L_cos-{}L_weight---Loss.txt".format(
            args.L_weight_coef, args.L_weight_coef), "a")
        f.write("= Epoch " + str(epoch) + " = , loss = " + str(loss) + "\n")
        f.close()

        with torch.no_grad():
            if (loss < best_loss):
                best_loss = loss
                best_epoch = epoch + 1
                # savepath = os.path.join(checkpoints_path, "train_epoch"+str(epoch)+".pth")  # save each best epoch
                savepath = os.path.join(checkpoints_path, "train_epoch_best.pth")  # overwrite the best epoch
                state = {
                    'epoch': best_epoch,
                    'pt_encoder_state_dict': pt_encoder.state_dict(),
                    'weighting_state_dict': Weighting_Net.state_dict(),
                    'nml_regressor_state_dict': Nml_Regression_Net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)

    end_time = time.time()
    lapsed_seconds = end_time - start_time
    print('End of training...')
    print("Training time: ", datetime.timedelta(seconds=lapsed_seconds))



if __name__ == '__main__':
    main()

