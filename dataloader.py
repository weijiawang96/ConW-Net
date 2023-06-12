
from __future__ import print_function
import os
import os.path
import sys
import torch
import torch.utils.data as data
import numpy as np
import scipy.spatial as spatial
from sklearn.neighbors import NearestNeighbors
import json
from utils import pca_alignment
import math
import random
import time
from scipy.spatial.transform import Rotation as R


# do NOT modify the returned points! kdtree uses a reference, not a copy of these points,
# so modifying the points would make the kdtree give incorrect results
def load_shape(point_filename, normals_filename, curv_filename, pidx_filename):
    pts = np.load(point_filename+'.npy')

    if normals_filename != None:
        normals = np.load(normals_filename+'.npy')
    else:
        normals = None

    if curv_filename != None:
        curvatures = np.load(curv_filename+'.npy')
    else:
        curvatures = None

    if pidx_filename != None:
        patch_indices = np.load(pidx_filename+'.npy')
    else:
        patch_indices = None

    sys.setrecursionlimit(int(max(1000, round(pts.shape[0]/10)))) # otherwise KDTree construction may run out of recursions
    kdtree = spatial.cKDTree(pts, 10)

    return Shape(pts=pts, kdtree=kdtree, normals=normals, curv=curvatures, pidx=patch_indices)

class SequentialPointcloudPatchSampler(data.sampler.Sampler):

    def __init__(self, data_source):
        self.data_source = data_source
        self.total_patch_count = None

        self.total_patch_count = 0
        for shape_ind, _ in enumerate(self.data_source.shape_names):
            self.total_patch_count = self.total_patch_count + self.data_source.shape_patch_count[shape_ind]

    def __iter__(self):
        return iter(range(self.total_patch_count))

    def __len__(self):
        return self.total_patch_count


class SequentialShapeRandomPointcloudPatchSampler(data.sampler.Sampler):

    def __init__(self, data_source, patches_per_shape, seed=None, sequential_shapes=False, identical_epochs=False):
        self.data_source = data_source
        self.patches_per_shape = patches_per_shape
        self.sequential_shapes = sequential_shapes
        self.seed = seed
        self.identical_epochs = identical_epochs
        self.total_patch_count = None
        self.shape_patch_inds = None

        if self.seed is None:
            self.seed = np.random.random_integers(0, 2**32-1, 1)[0]
        self.rng = np.random.RandomState(self.seed)

        self.total_patch_count = 0
        for shape_ind, _ in enumerate(self.data_source.shape_names):
            self.total_patch_count = self.total_patch_count + min(self.patches_per_shape, self.data_source.shape_patch_count[shape_ind])

    def __iter__(self):

        # optionally always pick the same permutation (mainly for debugging)
        if self.identical_epochs:
            self.rng.seed(self.seed)

        # global point index offset for each shape
        shape_patch_offset = list(np.cumsum(self.data_source.shape_patch_count))
        shape_patch_offset.insert(0, 0)
        shape_patch_offset.pop()

        shape_inds = range(len(self.data_source.shape_names))

        if not self.sequential_shapes:
            shape_inds = self.rng.permutation(shape_inds)

        # return a permutation of the points in the dataset where all points in the same shape are adjacent (for performance reasons):
        # first permute shapes, then concatenate a list of permuted points in each shape
        self.shape_patch_inds = [[]]*len(self.data_source.shape_names)
        point_permutation = []
        for shape_ind in shape_inds:
            start = shape_patch_offset[shape_ind]
            end = shape_patch_offset[shape_ind]+self.data_source.shape_patch_count[shape_ind]

            global_patch_inds = self.rng.choice(range(start, end), size=min(self.patches_per_shape, end-start), replace=False)
            point_permutation.extend(global_patch_inds)

            # save indices of shape point subset
            self.shape_patch_inds[shape_ind] = global_patch_inds - start

        return iter(point_permutation)

    def __len__(self):
        return self.total_patch_count

class RandomPointcloudPatchSampler(data.sampler.Sampler):

    def __init__(self, data_source, patches_per_shape, seed=None, identical_epochs=False):
        self.data_source = data_source
        self.patches_per_shape = patches_per_shape
        self.seed = seed
        self.identical_epochs = identical_epochs
        self.total_patch_count = None

        if self.seed is None:
            self.seed = np.random.random_integers(0, 2**32-1, 1)[0]
        self.rng = np.random.RandomState(self.seed)

        self.total_patch_count = 0
        for shape_ind, _ in enumerate(self.data_source.shape_names):
            self.total_patch_count = self.total_patch_count + min(self.patches_per_shape, self.data_source.shape_patch_count[shape_ind])

    def __iter__(self):

        # optionally always pick the same permutation (mainly for debugging)
        if self.identical_epochs:
            self.rng.seed(self.seed)

        return iter(self.rng.choice(sum(self.data_source.shape_patch_count), size=self.total_patch_count, replace=False))

    def __len__(self):
        return self.total_patch_count


class Shape():
    def __init__(self, pts, kdtree, normals=None, curv=None, pidx=None):
        self.pts = pts
        self.kdtree = kdtree
        self.normals = normals
        self.curv = curv
        self.pidx = pidx # patch center points indices (None means all points are potential patch centers)


class Cache():
    def __init__(self, capacity, loader, loadfunc):
        self.elements = {}
        self.used_at = {}
        self.capacity = capacity
        self.loader = loader
        self.loadfunc = loadfunc
        self.counter = 0

    def get(self, element_id):
        if element_id not in self.elements:
            # cache miss

            # if at capacity, throw out least recently used item
            if len(self.elements) >= self.capacity:
                remove_id = min(self.used_at, key=self.used_at.get)
                del self.elements[remove_id]
                del self.used_at[remove_id]

            # load element
            self.elements[element_id] = self.loadfunc(self.loader, element_id)

        self.used_at[element_id] = self.counter
        self.counter += 1

        return self.elements[element_id]


class PointcloudPatchDataset(data.Dataset):

    # patch radius as fraction of the bounding box diagonal of a shape
    def __init__(self, root, shape_list_filename, patch_radius, points_per_patch, patch_features,
                 seed=None, identical_epochs=False, use_pca=False, center='point', point_tuple=1, cache_capacity=1,
                 point_count_std=0.0, sparse_patches=False, neighbor_search_method='k', train_state='pretrain'):

        start_time = time.time()

        # initialize parameters
        self.root = root
        self.shape_list_filename = shape_list_filename
        self.patch_features = patch_features
        self.patch_radius = patch_radius
        self.points_per_patch = points_per_patch
        self.identical_epochs = identical_epochs
        self.use_pca = use_pca
        self.sparse_patches = sparse_patches
        self.center = center
        self.point_tuple = point_tuple
        self.point_count_std = point_count_std
        self.seed = seed
        self.neighbor_search_method = neighbor_search_method
        # self.include_normals = False  # False for test
        self.include_curvatures = False
        self.include_neighbor_normals = False
        self.train_state = train_state
        if train_state == "test":
            self.include_normals = False
        else:
            self.include_normals = True

        # append additional information
        self.kdtrees = []
        self.point_cloud_bbdiags = []

        # self.loaded_shape = None
        self.load_iteration = 0
        self.shape_cache = Cache(cache_capacity, self, PointcloudPatchDataset.load_shape_by_index)

        # get all shape names in the dataset
        if train_state == "test":  # only test 1 shape each time
            self.shape_names = [self.shape_list_filename]
        else:  # not testing
            self.shape_names = []
            with open(os.path.join(root, self.shape_list_filename)) as f:
                self.shape_names = f.readlines()
            self.shape_names = [x.strip() for x in self.shape_names]
            self.shape_names = list(filter(None, self.shape_names))

        # initialize rng for picking points in a patch
        if self.seed is None:
            self.seed = np.random.random_integers(0, 2**32-1, 1)[0]
        self.rng = np.random.RandomState(self.seed)

        # get basic information for each shape in the dataset
        self.shape_patch_count = []  # each element is the number of points in that shape. Equals num of patches
        # self.patch_radius_absolute = []  # don't do ball query
        print("now getting shapes in ", self.shape_list_filename)
        for shape_ind, shape_name in enumerate(self.shape_names):
            print('getting information for shape %s' % (shape_name))

            # Points
            # load from text file and save in more efficient numpy format
            point_filename = os.path.join(self.root, shape_name+'.xyz')
            pts = np.loadtxt(point_filename).astype('float32')
            np.save(point_filename+'.npy', pts)

            # Normals
            if self.include_normals:
                normals_filename = os.path.join(self.root, shape_name+'.normals')
                normals = np.loadtxt(normals_filename).astype('float32')
                np.save(normals_filename+'.npy', normals)
            else:
                normals_filename = None

            shape = self.shape_cache.get(shape_ind)

            if shape.pidx is None:
                self.shape_patch_count.append(shape.pts.shape[0])
            else:
                self.shape_patch_count.append(len(shape.pidx))

            # append neighbour dict
            pts = shape.pts[:, :3]

            # query neighbours
            # imitate triplet loss, which query neighbours and append as list.
            pts_kdtree = spatial.cKDTree(pts)
            self.kdtrees.append(pts_kdtree)

            # append point cloud size
            bbdiag = float(np.linalg.norm(pts.max(0) - pts.min(0), 2))
            self.point_cloud_bbdiags.append(bbdiag)

            ################################################# End init #################################################


    # returns a patch centered at the point with the given global index
    # and the ground truth normal the the patch center
    def __getitem__(self, index):
        shape_ind, patch_ind = self.shape_index(index)

        shape = self.shape_cache.get(shape_ind)
        # exit()

        # query neighbour index
        kd_tree = self.kdtrees[shape_ind]
        query_pt = shape.pts[patch_ind]

        _, patch_neighbour_idx = kd_tree.query(query_pt, k=self.points_per_patch)
        patch_pts = shape.pts[patch_neighbour_idx]  # no normals

        # centralize at patch[0]
        patch_pts -= query_pt  # so central point is 0,0,0

        # Find the furthest pt from center
        internal_radius = [np.linalg.norm(d) for d in patch_pts][-1]  # find the longest distance, i.e. furthest point
        patch_pts /= internal_radius  # unify the patch by /= the furthest distance

        # Do PCA alignment rotation
        patch_pts_pcarot, patch_pts_inv = pca_alignment(patch_pts)

        if self.train_state == "test":  # No need to include normals
            return patch_pts_pcarot, patch_pts_inv  # downstream, only return central noraml

        else:  # require normals
            patch_nmls = shape.normals[patch_neighbour_idx]  # normals only
            patch_normals_pcarot = np.array(np.linalg.inv(patch_pts_inv) * np.matrix(patch_nmls.T)).T  # new

            ctr_nml_pca_rot = patch_normals_pcarot[0]  # get central gt normal
            ctr_nml_pca_rot_expdim = np.expand_dims(ctr_nml_pca_rot, 0)  # expand dim (unsqueeze)
            ctr_nml_pca_rot_repeat = np.repeat(ctr_nml_pca_rot_expdim, self.points_per_patch, axis=0)

            if self.train_state == "pretrain":
                # return patch_pts_pcarot, ctr_nml_pca_rot_expdim, patch_pts_inv  # for PointNet, only return ctr nml  --> Original simple ver

                # patch_normals_pcarot[0] & patch_normals_pcarot -- for weight calculation
                return patch_pts_pcarot, patch_normals_pcarot[0], patch_normals_pcarot, patch_pts_inv

            elif self.train_state == "normal_est":
                return patch_pts_pcarot, patch_normals_pcarot[0], patch_normals_pcarot, patch_pts_inv  # downstream, only return central noraml and patch gt nml
            else:
                raise NotImplementedError


    def __len__(self):
        return sum(self.shape_patch_count)


    # translate global (dataset-wide) point index to shape index & local (shape-wide) point index
    def shape_index(self, index):
        shape_patch_offset = 0
        shape_ind = None
        for shape_ind, shape_patch_count in enumerate(self.shape_patch_count):
            if index >= shape_patch_offset and index < shape_patch_offset + shape_patch_count:
                shape_patch_ind = index - shape_patch_offset
                break
            shape_patch_offset = shape_patch_offset + shape_patch_count

        return shape_ind, shape_patch_ind

    # load shape from a given shape index
    def load_shape_by_index(self, shape_ind):
        point_filename = os.path.join(self.root, self.shape_names[shape_ind]+'.xyz')
        normals_filename = os.path.join(self.root, self.shape_names[shape_ind]+'.normals') if self.include_normals else None
        curv_filename = os.path.join(self.root, self.shape_names[shape_ind]+'.curv') if self.include_curvatures else None
        pidx_filename = os.path.join(self.root, self.shape_names[shape_ind]+'.pidx') if self.sparse_patches else None
        return load_shape(point_filename, normals_filename, curv_filename, pidx_filename)
