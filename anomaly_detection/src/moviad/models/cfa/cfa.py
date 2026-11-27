from __future__ import annotations
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from einops import rearrange
from tqdm import tqdm
from sklearn.cluster import KMeans
from scipy.ndimage import gaussian_filter
from sklearn.metrics import precision_recall_curve
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from moviad.models.components.cfa.descriptor import Descriptor
from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
from moviad.models.vad_model import VADModel
from moviad.models.training_args import TrainingArgs
from moviad.models.cfa.cfa_loss import soft_boundary
from moviad.utilities.get_sizes import count_params, params_to_mb

@dataclass
class CFATrainArgs(TrainingArgs):

    def init_train(self, model: VADModel):
        if self.optimizer is None:
            learning_rate = 1e-3
            weight_decay = 5e-4
            self.optimizer = AdamW(params=model.parameters(),
                            lr=learning_rate,
                            weight_decay=weight_decay,
                            amsgrad=True)
        if self.loss_function is None:
            self.loss_function = soft_boundary


class CFA(VADModel):

    def __init__(
            self,
            feature_extractor: CustomFeatureExtractor,
            backbone: str,
            gamma_c:int = 1,
            gamma_d:int = 1,
            feature_maps_channels: int = 1792,
            K: int = 5,
            J: int = 3,
            bank_init: str = "mean",
            bank_patches_per_image: int = 64,
            bank_size: int | None = None,
            blur_sigma: float = 4.0,
            score_top_k: int = 0,
            foreground_fn=None,
        ):

        """
        Args:
            feature_extractor () : feature extractor to be used
            training_dataloader (torch.utils.data.DataLoader) : training dataloader
            backbone (str) : name of the used backbone in the feature extractor
            gamma_c (int) : model parameter
            gamma_d (int) : model parameter
            K (int) : nearest memorized features used by the attraction loss and by the score
            J (int) : following nearest memorized features used by the repulsion loss
            bank_init (str) : how the memory bank is initialized
                "mean"   : per-position mean of the training features (CFA paper,
                           suited to aligned objects), reduced with KMeans if gamma_c > 1
                "kmeans" : KMeans centroids of patches sampled from the whole training
                           set (position independent, for objects that are not aligned)
            bank_patches_per_image (int) : patches sampled per image when bank_init is "kmeans"
            bank_size (int) : memorized features when bank_init is "kmeans"
                (None = number of patches of an image // gamma_c, as in the paper)
            blur_sigma (float) : sigma of the gaussian smoothing of the anomaly map (0 = none)
            score_top_k (int) : image score = mean of the top-k pixels of the anomaly map
                (0 = maximum, as in the CFA paper)
            foreground_fn (callable) : images (b, 3, H, W) -> object mask (b, 1, H, W).
                If given, only the patches of the object are used for the "kmeans"
                memory bank, the loss and the anomaly map (zero on the background)
        """

        super().__init__()

        self.memory_bank = 0
        self.nu = 1e-3
        self.scale = None

        self.gamma_c = gamma_c
        self.gamma_d = gamma_d
        self.alpha = 1e-1
        self.K = K
        self.J = J
        if bank_init not in ("mean", "kmeans"):
            raise ValueError(f"bank_init must be 'mean' or 'kmeans', got {bank_init}")
        self.bank_init = bank_init
        self.bank_patches_per_image = bank_patches_per_image
        self.bank_size = bank_size
        self.foreground_fn = foreground_fn
        self.blur_sigma = blur_sigma
        self.score_top_k = score_top_k
        self.r = nn.Parameter(1e-5*torch.ones(1), requires_grad=True)

        self.device = feature_extractor.device

        self.backbone = backbone
        self.feature_maps_shape: tuple = None

        self.feature_extractor = feature_extractor
        self.feature_maps_channels = feature_maps_channels
        self.Descriptor = Descriptor(self.gamma_d, self.feature_maps_channels, self.backbone, self.device).to(self.device)

    def to(self, device: torch.device):
        super().to(device)
        self.feature_extractor.to(device)
        self.device = device

    def initialize_memory_bank(self, training_dataloader: DataLoader):
        """
        Initialize the memory bank

        Args:
            training_dataloader (DataLoader) : training dataloader
        """

        if self.bank_init == "kmeans":
            patches = self.sample_patches(self.feature_extractor, training_dataloader)
            memory_bank = CFA.kmeans_torch(patches, self.bank_size or max(1, (self.scale**2)//self.gamma_c))
            return memory_bank.transpose(-1, -2).detach()

        memory_bank = self.init_centroid(self.feature_extractor, training_dataloader)
        memory_bank = rearrange(memory_bank, 'b c h w -> (b h w) c').detach()

        if self.gamma_c > 1:
            memory_bank = memory_bank.cpu().detach().numpy()
            memory_bank = KMeans(n_clusters=(self.scale**2)//self.gamma_c, max_iter=3000).fit(memory_bank).cluster_centers_
            memory_bank = torch.Tensor(memory_bank).to(self.device)

        memory_bank = memory_bank.transpose(-1, -2).detach()
        return memory_bank

    def forward(self, x: torch.Tensor) -> float | tuple[torch.Tensor, torch.Tensor]:
        """
        CFA forward pass

        Args:
            x (torch.Tensor) : batch of images

        Returns:
            loss (float) if the model is in training mode
            else a tuple:
                [0] : tensor with the anomaly maps (one for every input image)
                [1] : tensor with image level anomaly scores (one for every input image)
        """

        phi_p = self.describe(x)

        if self.training:
            return phi_p

        heatmaps = self.distance_map(phi_p, self.patch_foreground(x)).squeeze(1)
        heatmaps = F.interpolate(heatmaps.unsqueeze(1), size=x.size(2), mode="bilinear",
                                 align_corners=False).squeeze(1)
        if self.blur_sigma > 0:
            heatmaps = CFA.gaussian_smooth_torch(heatmaps, sigma=self.blur_sigma)

        # image score = maximum of the smoothed heatmap, as in the CFA paper
        # (over the whole image, no margin), or mean of the top-k pixels
        flat = heatmaps.reshape(x.size(0), -1)
        if self.score_top_k > 0:
            img_scores = flat.topk(self.score_top_k, dim=1).values.mean(dim=1)
        else:
            img_scores = flat.amax(dim=1)

        return heatmaps.unsqueeze(dim=1).cpu().detach(), img_scores.cpu().detach()

    def describe(self, x: torch.Tensor) -> torch.Tensor:
        """Adapted patch features phi(p) of a batch of images, shape (b, h*w, c)."""

        p = self.feature_extractor(x)
        if isinstance(p, dict):
            p = list(p.values())

        if not self.feature_maps_shape:
            self.feature_maps_shape = (1, self.feature_maps_channels, p[0].shape[2], p[0].shape[3])
        self.scale = p[0].size(2)

        phi_p = self.Descriptor(p)
        return rearrange(phi_p, 'b c h w -> b (h w) c')

    def patch_foreground(self, x: torch.Tensor) -> torch.Tensor | None:
        """Patches of the object (bool, shape (b, h*w)); None without foreground_fn."""

        if self.foreground_fn is None:
            return None

        mask = self.foreground_fn(x).float()
        return (F.adaptive_avg_pool2d(mask, self.scale) > 0.5).flatten(1)

    def distance_map(self, phi_p: torch.Tensor, foreground: torch.Tensor | None = None) -> torch.Tensor:
        """
        Anomaly map at the resolution of the feature maps, shape (b, 1, h, w):
        distance from the nearest memorized feature, weighted with the softmin
        over the K nearest ones. With foreground (bool, shape (b, h*w)) the
        distances are computed only for those patches, the others are zero.
        """

        if foreground is None:
            dist = self.patch_distance(phi_p)
        else:
            dist = phi_p.new_zeros(phi_p.shape[:2])
            if foreground.any():
                dist[foreground] = self.patch_distance(phi_p[foreground].unsqueeze(0)).squeeze(0)

        return rearrange(dist, 'b (h w) -> b 1 h w', h=self.scale)

    def patch_distance(self, phi_p: torch.Tensor, chunk: int = 2 ** 28) -> torch.Tensor:
        """Anomaly score of the patches phi_p (b, n, c), shape (b, n)."""

        # at most chunk distances (patch x memory bank) at a time
        n = max(1, chunk // (phi_p.size(0) * self.memory_bank.size(1)))
        if phi_p.size(1) > n:
            return torch.cat([self.patch_distance(part, chunk) for part in phi_p.split(n, dim=1)], dim=1)

        features = torch.sum(torch.pow(phi_p, 2), 2, keepdim=True)
        centers  = torch.sum(torch.pow(self.memory_bank, 2), 0, keepdim=True)
        # in-place operations: a single (patch x memory bank) matrix in memory
        dist     = torch.matmul(phi_p, (self.memory_bank)).mul_(-2).add_(features).add_(centers)

        # the square root is monotonic: computing it on the K nearest is enough
        dist     = dist.topk(self.K, largest=False).values
        dist     = torch.sqrt(dist.clamp_min(0))  # avoids NaN from numerical cancellation

        return (F.softmin(dist, dim=-1)[:, :, 0]) * dist[:, :, 0]

    def train_step(self, batch: torch.Tensor, training_args: TrainingArgs):
        """
        Training step

        Args:
            x (torch.Tensor) : batch of images

        Returns:
            loss (float) : CFA loss
        """

        self.train()
        batch = batch.to(self.device)
        phi_p = self(batch)
        foreground = self.patch_foreground(batch)
        if foreground is not None:
            # loss only on the object's patches
            phi_p = phi_p[foreground].unsqueeze(0)
        training_args.optimizer.zero_grad()

        # at most 2**28 distances (patch x memory bank) at a time: the gradient
        # w.r.t. phi_p is accumulated in chunks and then propagated to the descriptor
        # (same gradient as the loss computed in one go)
        n = max(1, 2 ** 28 // (phi_p.size(0) * self.memory_bank.size(1)))
        if phi_p.size(1) <= n:
            loss = training_args.loss_function(phi_p, self.memory_bank, self.K, self.J, self.r, self.alpha, self.nu)
            loss.backward()
            loss = loss.item()
        else:
            leaf = phi_p.detach().requires_grad_()
            loss = 0.0
            for part in leaf.split(n, dim=1):
                part_loss = training_args.loss_function(part, self.memory_bank, self.K, self.J, self.r, self.alpha, self.nu)
                part_loss = part_loss * (part.size(1) / leaf.size(1))
                part_loss.backward()
                loss += part_loss.item()
            phi_p.backward(leaf.grad)

        training_args.optimizer.step()

        return loss

    def train_epoch(self, epoch: int, train_dataloader: torch.utils.data.DataLoader, training_args: TrainingArgs):

        self.train()

        if epoch == 0:
            self.memory_bank = nn.Parameter(self.initialize_memory_bank(train_dataloader), requires_grad=False)

        avg_batch_loss = 0

        # train the model
        for batch in tqdm(train_dataloader):
            avg_batch_loss += self.train_step(batch, training_args)

        avg_batch_loss /= len(train_dataloader)
        return avg_batch_loss

    @torch.no_grad()
    def init_centroid(self, feature_extractor:CustomFeatureExtractor, data_loader:DataLoader):
        """
        This method initializes the memory bank points

        Args:
            feature_extractor (CustomFeatureExtractor) : feature extractor to be used
            training_dataloader (torch.utils.data.DataLoader) : training dataloader
        """

        memory_bank = 0

        for i, x in enumerate(tqdm(data_loader)):
            x = x.to(self.device)
            p = feature_extractor(x)

            if isinstance(p, dict):
                p = list(p.values())

            self.scale = p[0].size(2)
            phi_p = self.Descriptor(p)
            memory_bank = ((memory_bank * i) + torch.mean(phi_p, dim=0, keepdim=True).detach()) / (i+1)

        return memory_bank

    @torch.no_grad()
    def sample_patches(self, feature_extractor:CustomFeatureExtractor, data_loader:DataLoader):
        """
        Adapted features of bank_patches_per_image random patches of every
        training image, shape (n, c)
        """

        patches = []

        for x in tqdm(data_loader):
            x = x.to(self.device)
            p = feature_extractor(x)

            if isinstance(p, dict):
                p = list(p.values())

            self.scale = p[0].size(2)
            phi_p = rearrange(self.Descriptor(p), 'b c h w -> b (h w) c')

            # random patches of every image (only on the object if foreground_fn is set)
            order = torch.rand(phi_p.shape[:2], device=phi_p.device)
            foreground = self.patch_foreground(x)
            if foreground is not None:
                order = order + foreground
            idx = order.argsort(dim=1, descending=True)[:, :self.bank_patches_per_image]
            sampled = torch.gather(phi_p, 1, idx.unsqueeze(-1).expand(-1, -1, phi_p.size(2)))
            if foreground is not None:
                sampled = sampled[torch.gather(foreground, 1, idx)]
            patches.append(sampled.reshape(-1, phi_p.size(2)))

        return torch.cat(patches)

    @staticmethod
    @torch.no_grad()
    def kmeans_torch(points: torch.Tensor, n_clusters: int, iters: int = 20, chunk: int = 16384):
        """KMeans (Lloyd) on the device of points; centroids initialized with random points."""

        n_clusters = min(n_clusters, points.size(0))
        centers = points[torch.randperm(points.size(0), device=points.device)[:n_clusters]].clone()

        for _ in range(iters):
            sums = torch.zeros_like(centers)
            counts = torch.zeros(centers.size(0), device=points.device)
            for i in range(0, points.size(0), chunk):
                batch = points[i:i + chunk]
                nearest = torch.cdist(batch, centers).argmin(dim=1)
                sums.index_add_(0, nearest, batch)
                counts.index_add_(0, nearest, torch.ones_like(nearest, dtype=counts.dtype))
            # centroids left without points are not moved
            filled = counts > 0
            centers[filled] = sums[filled] / counts[filled].unsqueeze(1)

        return centers

    def get_model_size(self):
        feature_extractor_params = self.feature_extractor.get_size()
        descriptor_params = params_to_mb(count_params(self.Descriptor))
        memory_bank_params = params_to_mb(count_params(self.memory_bank))

        return {
            "feature_extractor": feature_extractor_params,
            "descriptor": descriptor_params,
            "memory_bank": memory_bank_params,
            "total": feature_extractor_params + descriptor_params + memory_bank_params
        }

    def load_model(self, path):

        """
        Load the CFA memory bank

        Parameters:
        ----------
            path (str): where the pt file containing the memory bank is stored
        """

        state_dict = torch.load(path)

        if "C" not in state_dict.keys():
            raise RuntimeError("Memory Bank tensor not in model checkpoint")

        # load the memory bank
        self.memory_bank = state_dict["C"]

        # load the Patch Descriptor
        self.Descriptor = Descriptor(self.gamma_d, int(state_dict["feature_maps_channels"]), self.backbone, self.device)
        desc_dict = {
            'layer.weight'  : state_dict['Descriptor.layer.weight'],
            'layer.bias' : state_dict['Descriptor.layer.bias'],
            'layer.conv.weight' : state_dict['Descriptor.layer.conv.weight'],
            'layer.conv.bias' : state_dict['Descriptor.layer.conv.bias']
        }
        self.Descriptor.load_state_dict(desc_dict)

        self.feature_maps_channels = nn.Parameter(state_dict["feature_maps_channels"], requires_grad=False)

    def save_anomaly_map(self, dirpath, anomaly_map, pred_score, filepath, x_type, mask):
        """
        Args:
            dirpath     (str)       : Output directory path.
            anomaly_map (np.ndarray): Anomaly map with the same size as the input image.
            filepath    (str)       : Path of the input image.
            x_type      (str)       : Anomaly type (e.g. "good", "crack", etc).
            mask
        """

        def cvt2heatmap(gray):
            return cv.applyColorMap(np.uint8(gray), cv.COLORMAP_JET)

        # Get the image file name.
        filename = os.path.basename(filepath)

        # Load the image file and resize.
        original_image = cv.imread(filepath)
        original_image = cv.resize(original_image, anomaly_map.shape[:2])

        # Normalize anomaly map for easier visualization.
        anomaly_map_norm = cvt2heatmap(255 * CFA.rescale(anomaly_map))

        # Overlay the anomaly map to the origimal image.
        output_image = (anomaly_map_norm / 2 + original_image / 2).astype(np.uint8)

        # Create a figure and axes
        fig, axes = plt.subplots(1, 3, figsize=(10, 5))

        #convert the images to RGB
        original_image = cv.cvtColor(original_image, cv.COLOR_BGR2RGB)
        output_image = cv.cvtColor(output_image, cv.COLOR_BGR2RGB)

        # Display the input image
        axes[0].imshow(original_image)
        axes[0].set_title(f'Original Image {x_type}')
        axes[0].axis('off')

        # Display the mask image
        axes[1].imshow(mask.squeeze(), cmap ='gray')
        axes[1].set_title(f'Mask')
        axes[1].axis('off')

        # Display the final image
        axes[2].imshow(output_image)
        axes[2].set_title(f'Heatmap {pred_score}')
        axes[2].axis('off')

        # Show the plot
        plt.savefig(str(dirpath / f"{x_type}_{filename}.jpg"))


    # --------------- SEGMENTATION MASK PRODUCTION ---------------- #

    @staticmethod
    def upsample(x, size, mode):
        return (
            F.interpolate(x.unsqueeze(1), size=size, mode=mode, align_corners=False)
            .squeeze()
        )

    @staticmethod
    def gaussian_smooth(x, sigma=4):
        bs = x.shape[0]
        for i in range(0, bs):
            x[i] = gaussian_filter(x[i], sigma=sigma)
        return x

    @staticmethod
    def gaussian_smooth_torch(x, sigma=4):
        if x.ndim == 2:
            x = x.unsqueeze(0) # if processing (working on) one image

        # kernel 2*int(4*sigma+0.5)+1 (33 for sigma=4), like scipy's gaussian_filter:
        # with kernel 3 the filter was effectively a 3x3 box.
        # Separable filter (rows, then columns) with reflected border: same
        # result as torchvision GaussianBlur, but much faster
        radius = int(4.0 * sigma + 0.5)
        t = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
        kernel = torch.exp(-0.5 * (t / sigma) ** 2)
        kernel = kernel / kernel.sum()

        x = x.unsqueeze(1)
        x = F.conv2d(F.pad(x, (radius, radius, 0, 0), mode="reflect"), kernel.view(1, 1, 1, -1))
        x = F.conv2d(F.pad(x, (0, 0, radius, radius), mode="reflect"), kernel.view(1, 1, -1, 1))
        return x.squeeze(1)

    @staticmethod
    def rescale(x):
        return (x - x.min()) / (x.max() - x.min())

    @staticmethod
    def get_threshold(gt: np.ndarray, score: np.ndarray) -> float:
        """
        Calculate the segmentation threshold

        Args:
            gt (np.array)    : ground truth masks
            score (np.array) : predicted masks

        Returns:
            threshold (float) : segmentation threshold
        """

        gt_mask = np.asarray(gt)
        precision, recall, thresholds = precision_recall_curve(
            gt_mask.flatten(), score.flatten()
        )
        a = 2 * precision * recall
        b = precision + recall
        f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)

        # consider the threshold with the highest f1 score
        threshold = thresholds[np.argmax(f1)]

        return threshold
