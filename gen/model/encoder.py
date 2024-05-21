import torch.nn as nn
import torch
import torch.nn.functional as F
from .layers.transformer import *
from .layers.improved_transformer import *
from gen.config import *
from gen.model.network import *

from typing import Optional, Tuple, List


class SketchEncoder(nn.Module):
  """
  Transformer Encoder 
  """
  def __init__(self):
    super(SketchEncoder, self).__init__()
    self.embed_dim = ENCODER_CONFIG['embed_dim']
    self.coord_embed_x = Embedder(2**CAD_BIT+SKETCH_PAD, self.embed_dim)
    self.coord_embed_y = Embedder(2**CAD_BIT+SKETCH_PAD, self.embed_dim)
    self.pixel_embeds = Embedder(2**CAD_BIT * 2**CAD_BIT+SKETCH_PAD, self.embed_dim)
    self.pos_embed = PositionalEncoding(max_len=MAX_CAD, d_model=self.embed_dim)
    layers = TransformerEncoderLayerImproved(d_model=self.embed_dim, nhead=ENCODER_CONFIG['num_heads'], 
        dim_feedforward=ENCODER_CONFIG['hidden_dim'], dropout=ENCODER_CONFIG['dropout_rate'])
    self.encoder = TransformerEncoder(layers, ENCODER_CONFIG['num_layers'], LayerNorm(self.embed_dim))
   
  
  def forward(self, pixel, coord, mask):
    """ forward pass """
    coord_embed = self.coord_embed_x(coord[...,0]) + self.coord_embed_y(coord[...,1]) # [bs, vlen, dim]
    pixel_embed = self.pixel_embeds(pixel)
    embed_inputs = pixel_embed + coord_embed 
    input_embeds = self.pos_embed(embed_inputs.transpose(0,1))
    outputs = self.encoder(src=input_embeds, src_key_padding_mask=mask)  # [seq_len, bs, dim]    
    return outputs.transpose(0,1)



class ExtEncoder(nn.Module):
  """
  Transformer Encoder 
  """
  def __init__(self):
    super(ExtEncoder, self).__init__()
    self.embed_dim = ENCODER_CONFIG['embed_dim']
    self.ext_embed = Embedder(2**CAD_BIT+EXT_PAD, self.embed_dim)
    self.pos_embed = PositionalEncoding(max_len=MAX_EXT, d_model=self.embed_dim)
    layers = TransformerEncoderLayerImproved(d_model=self.embed_dim, nhead=ENCODER_CONFIG['num_heads'], 
        dim_feedforward=ENCODER_CONFIG['hidden_dim'], dropout=ENCODER_CONFIG['dropout_rate'])
    self.encoder = TransformerEncoder(layers, ENCODER_CONFIG['num_layers'], LayerNorm(self.embed_dim))
   

  def forward(self, extrude, mask):
    """ forward pass """
    embed_inputs = self.ext_embed(extrude)
    input_embeds = self.pos_embed(embed_inputs.transpose(0,1))
    outputs = self.encoder(src=input_embeds, src_key_padding_mask=mask)  # [seq_len, bs, dim]    
    return outputs.transpose(0,1)


##########################
##########################
##########################
##########################
##########################
##########################
##########################

def build_shared_mlp(mlp_spec: List[int], bn: bool = True):
    layers = []
    for i in range(1, len(mlp_spec)):
        layers.append(
            nn.Conv2d(mlp_spec[i - 1], mlp_spec[i], kernel_size=1, bias=not bn)
        )
        if bn:
            layers.append(nn.BatchNorm2d(mlp_spec[i]))
        layers.append(nn.ReLU(True))

    return nn.Sequential(*layers)

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


class QueryAndGroup(nn.Module):
    r"""
    Groups with a ball query of radius

    Parameters
    ---------
    radius : float32
        Radius of ball
    nsample : int32
        Maximum number of features to gather in the ball
    """

    def __init__(self, radius, nsample, use_xyz=True):
        # type: (QueryAndGroup, float, int, bool) -> None
        super(QueryAndGroup, self).__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz

    def forward(self, xyz, new_xyz, features=None):
        # type: (QueryAndGroup, torch.Tensor. torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor]
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            centriods (B, npoint, 3)
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, 3 + C, npoint, nsample) tensor
        """

        idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
        grouped_xyz -= new_xyz.unsqueeze(-2)
        # xyz_trans = xyz.transpose(1, 2).contiguous()
        # grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        # grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)

        if features is not None:
            grouped_features = index_points(features.transpose(1, 2), idx)
            # grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz, grouped_features], dim=-1
                )  # (B, npoint, nsample, C + 3) ## (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert (
                self.use_xyz
            ), "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        return new_features.permute(0, 3, 1, 2)

class GroupAll(nn.Module):
    r"""
    Groups all features

    Parameters
    ---------
    """

    def __init__(self, use_xyz=True):
        # type: (GroupAll, bool) -> None
        super(GroupAll, self).__init__()
        self.use_xyz = use_xyz

    def forward(self, xyz, new_xyz, features=None):
        # type: (GroupAll, torch.Tensor, torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor]
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            Ignored
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, C + 3, 1, N) tensor
        """

        grouped_xyz = xyz.transpose(1, 2).unsqueeze(2)
        if features is not None:
            grouped_features = features.unsqueeze(2)
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz, grouped_features], dim=1
                )  # (B, 3 + C, 1, N)
            else:
                new_features = grouped_features
        else:
            new_features = grouped_xyz

        return new_features


class PointnetSAModule(nn.Module):
    r"""Pointnet set abstrction layer

    Parameters
    ----------
    npoint : int
        Number of features
    radius : float
        Radius of ball
    nsample : int
        Number of samples in the ball query
    mlp : list
        Spec of the pointnet before the global max_pool
    bn : bool
        Use batchnorm
    """
    def __init__(
        self, mlp, npoint=None, radius=None, nsample=None, bn=True, use_xyz=True
    ):
        # type: (PointnetSAModule, list[int], int, float, int, bool, bool) -> None
        mlps=[mlp]
        radii=[radius]
        nsamples=[nsample]

        super(PointnetSAModule, self).__init__()

        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(
                QueryAndGroup(radius, nsample, use_xyz=use_xyz)
                if npoint is not None
                else GroupAll(use_xyz)
            )
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            self.mlps.append(build_shared_mlp(mlp_spec, bn))        

    def forward(
    self, xyz: torch.Tensor, features: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, C, N) tensor of the descriptors of the the features

        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B,  \sum_k(mlps[k][-1]), npoint) tensor of the new_features descriptors
        """

        new_features_list = []

        new_xyz = None
        if self.npoint is not None:
            fps_idx = farthest_point_sample(xyz, self.npoint) # [B, npoint]
            new_xyz = index_points(xyz, fps_idx) # [B, npoints, 3]

        # xyz_flipped = xyz.transpose(1, 2).contiguous()
        # new_xyz = (
        #     pointnet2_utils.gather_operation(
        #         xyz_flipped, pointnet2_utils.furthest_point_sample(xyz, self.npoint)
        #     )
        #     .transpose(1, 2)
        #     .contiguous()
        #     if self.npoint is not None
        #     else None
        # )

        for i in range(len(self.groupers)):
            new_features = self.groupers[i](
                xyz, new_xyz, features
            )  # (B, C, npoint, nsample)

            new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
            new_features = F.max_pool2d(
                new_features, kernel_size=[1, new_features.size(3)]
            )  # (B, mlp[-1], npoint, 1)
            new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)

            new_features_list.append(new_features)

        return new_xyz, torch.cat(new_features_list, dim=1)


class PointNet2(nn.Module):
    def __init__(self):
        super(PointNet2, self).__init__()

        self.use_xyz = True

        self._build_model()

    def _build_model(self):
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=512,
                radius=0.1,
                nsample=64,
                mlp=[0, 32, 32, 64],
                # bn=False,
                use_xyz=self.use_xyz,
            )
        )

        self.SA_modules.append(
            PointnetSAModule(
                npoint=256,
                radius=0.2,
                nsample=64,
                mlp=[64, 64, 64, 128],
                # bn=False,
                use_xyz=self.use_xyz,
            )
        )

        self.SA_modules.append(
            PointnetSAModule(
                npoint=128,
                radius=0.4,
                nsample=64,
                mlp=[128, 128, 128, 256],
                # bn=False,
                use_xyz=self.use_xyz,
            )
        )

        self.SA_modules.append(
            PointnetSAModule(
                mlp=[256, 256, 512, 1024],
                # bn=False,
                use_xyz=self.use_xyz
            )
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(True),
            nn.Linear(512, 256),
            nn.LeakyReLU(True),
            nn.Linear(256, 256),
            nn.Tanh()
        )

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)

        for module in self.SA_modules:
            xyz, features = module(xyz, features)

        return self.fc_layer(features.squeeze(-1))


#################
#################
#################
#################
#################
#################

class PointnetFPModule(nn.Module):
    r"""Propigates the features of one set to another

    Parameters
    ----------
    mlp : list
        Pointnet module parameters
    bn : bool
        Use batchnorm
    """

    def __init__(self, mlp, bn=True):
        # type: (PointnetFPModule, List[int], bool) -> None
        super(PointnetFPModule, self).__init__()
        self.mlp = build_shared_mlp(mlp, bn=bn)

    def forward(self, unknown, known, unknow_feats, known_feats):
        # type: (PointnetFPModule, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""
        Parameters
        ----------
        unknown : torch.Tensor
            (B, n, 3) tensor of the xyz positions of the unknown features
        known : torch.Tensor
            (B, m, 3) tensor of the xyz positions of the known features
        unknow_feats : torch.Tensor
            (B, C1, n) tensor of the features to be propigated to
        known_feats : torch.Tensor
            (B, C2, m) tensor of features to be propigated

        Returns
        -------
        new_features : torch.Tensor
            (B, mlp[-1], n) tensor of the features of the unknown features
        """

        B, N, _ = unknown.shape

        if known is not None:
            dists = square_distance(unknown, known)
            dists, idx = dists.sort(dim=-1)
            dist, idx = dists[:, :, :3], idx[:, :, :3].reshape(B, -1)  # [B, N, 3]
            # dist, idx = pointnet2_utils.three_nn(unknown, known)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = (dist_recip / norm).view(B, N, 3, 1)

            known_feats = known_feats.transpose(1, 2)
            chosen_known_feats = index_points(known_feats, idx).view(B, N, 3, -1)
            interpolated_feats = torch.sum(chosen_known_feats * weight, dim=2).transpose(1, 2)
            # interpolated_feats = pointnet2_utils.three_interpolate(
            #     known_feats, idx, weight
            # )

        else:
            interpolated_feats = known_feats.expand(
                *(known_feats.size()[0:2] + [unknown.size(1)])
            )

        if unknow_feats is not None:
            new_features = torch.cat(
                [interpolated_feats, unknow_feats], dim=1
            )  # (B, C2 + C1, n)
        else:
            new_features = interpolated_feats

        new_features = new_features.unsqueeze(-1)
        new_features = self.mlp(new_features)

        return new_features.squeeze(-1)

class PointNet2_SEG(nn.Module):
    def __init__(self):
        super(PointNet2_SEG, self).__init__()

        self.use_xyz = True

        self._build_model()

    def _build_model(self):
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=512,
                radius=0.1,
                nsample=64,
                mlp=[0, 32, 32, 64],
                # bn=False,
                use_xyz=self.use_xyz,
            )
        )

        self.SA_modules.append(
            PointnetSAModule(
                npoint=256,
                radius=0.2,
                nsample=64,
                mlp=[64, 64, 64, 128],
                # bn=False,
                use_xyz=self.use_xyz,
            )
        )

        self.SA_modules.append(
            PointnetSAModule(
                npoint=128,
                radius=0.4,
                nsample=64,
                mlp=[128, 128, 128, 256],
                # bn=False,
                use_xyz=self.use_xyz,
            )
        )

        self.SA_modules.append(
            PointnetSAModule(
                npoint=32,
                radius=0.8,
                nsample=64,
                mlp=[256, 256, 256, 512],
                # bn=False,
                use_xyz=self.use_xyz,
            )
        )

        # self.SA_modules.append(
        #     PointnetSAModule(
        #         mlp=[256, 256, 512, 1024],
        #         # bn=False,
        #         use_xyz=self.use_xyz
        #     )
        # )

        self.FP_modules = nn.ModuleList()
        # self.FP_modules.append(PointnetFPModule(mlp=[128 + 0, 128, 128, 128]))
        # self.FP_modules.append(PointnetFPModule(mlp=[256 + 64, 256, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 128, 256, 256]))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + 256, 256, 256]))

        self.fc_layer = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv1d(512, 256, kernel_size=1),
        )

        # self.fc_layer = nn.Sequential(
        #     nn.Linear(256, 512),
        #     nn.BatchNorm1d(512),
        #     nn.LeakyReLU(True),
        #     nn.Linear(512, 512),
        #     nn.BatchNorm1d(512),
        #     nn.LeakyReLU(True),
        #     nn.Dropout(0.5),
        #     nn.Linear(512, 256),
        #     nn.Tanh()
        # )

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        return self.fc_layer(l_features[2])

