"""PyTorch model for the PatchCore model implementation."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import os

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
#from memory_profiler import profile
from torch import Tensor, autocast, nn
from tqdm import tqdm

from moviad.models.patchcore.product_quantizer import ProductQuantizer
from moviad.models.vad_model import VADModel
from moviad.models.patchcore.anomaly_map import AnomalyMapGenerator
from moviad.models.patchcore.kcenter_greedy import CoresetExtractor
from moviad.models.training_args import TrainingArgs
from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
from moviad.utilities.get_sizes import params_to_mb, count_params

class StructCore(VADModel):
    """StructCore Module."""

    def __init__(
        self,
        feature_extractor: CustomFeatureExtractor,
        coreset_extractor: CoresetExtractor = None,
        num_neighbors: int = 9,
        apply_quantization: bool = False,
        memory_bank_size: int = 10000
    ) -> None:

        """
        Constructor of the patch-core model

        Args:
            device (torch.device): device to be used during the training
            input_size (tuple[int]): size of the input images
            feature_extractor (CustomFeatureExtractor): feature extractor to be used
            num_neighbors (int): number of neighbors to be considered in the k-nn search
        """

        super().__init__()

        self.num_neighbors = num_neighbors
        self.device = feature_extractor.device

        self.feature_extractor = feature_extractor
        self.feature_pooler = torch.nn.AvgPool2d(3,1,1)
        self.anomaly_map_generator = AnomalyMapGenerator()
        self.memory_bank_size = memory_bank_size
        self.memory_bank: Tensor
        self.train_mu = None
        self.train_sigma = None

        self.apply_quantization = apply_quantization
        if apply_quantization:
            self.product_quantizer = ProductQuantizer()

        if coreset_extractor is None:
            self.coreset_extractor = CoresetExtractor(False, self.device, k=self.memory_bank_size)
        else:
            self.coreset_extractor = coreset_extractor

    def to(self, device: torch.device):
        super().to(device)
        self.feature_extractor.to(device)

        if hasattr(self, "memory_bank"):
            if type(self.memory_bank) is dict:
                for k,v in self.memory_bank.items():
                    self.memory_bank[k] = v.to(self.device)
            else:
                self.memory_bank = self.memory_bank.to(self.device)

        if hasattr(self, "train_mu") and self.train_mu is not None:
            self.train_mu = self.train_mu.to(self.device)
        if hasattr(self, "train_sigma") and self.train_sigma is not None:
            self.train_sigma = self.train_sigma.to(self.device)

        self.device = device

    def extract_embedding(self, batch:torch.Tensor):
        with torch.no_grad():
            features = self.feature_extractor(batch.to(self.device))

        #concatenate the embeddings
        if isinstance(features, dict):
            features = list(features.values())

        # Apply smoothing (3x3 average pooling) to the features.
        smoothing = torch.nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        features  = [smoothing(feature) for feature in features]

        # Compute maximum shape.
        H_max = max([f.shape[2] for f in features])
        W_max = max([f.shape[3] for f in features])

        # Create resize function instance.
        resizer = torch.nn.Upsample(size=(H_max, W_max), mode="bilinear", align_corners=False)

        # Apply resize function for all input tensors.
        features = [resizer(f) for f in features]

        embedding =  torch.cat(features, dim=1)

        batch_size, _, width, height = embedding.shape
        embedding = self.reshape_embedding(embedding)

        return embedding, batch_size, width, height

    def forward(self, input_tensor: Tensor) -> Tensor | dict[str, Tensor]:
        """Return Embedding during training, or a tuple of anomaly map and anomaly score during testing.

        Steps performed:
        1. Get features from a CNN.
        2. Generate embedding based on the features.
        3. Compute anomaly map in test mode, return the embedding in training mode

        Args:
            input_tensor (Tensor): Input tensor

        Returns:
            Tensor | dict[str, Tensor]: Embedding for training,
                anomaly map and anomaly score for testing.
        """

        image_size = input_tensor.shape[2:]

        #extract the features for the input tensor
        embedding, batch_size, width, height = self.extract_embedding(input_tensor)

        #embedding shape: (#num_patches, emb_dim)

        if self.training:
            return embedding
        else:
            return self.calculate_anomaly_maps_scores(
                embedding=embedding,
                batch_size=batch_size,
                width=width,
                height=height,
                image_size=image_size,
                memory_bank=self.memory_bank
            )

    @staticmethod
    def get_structural_descriptor(anomaly_map: torch.Tensor) -> torch.Tensor:
        # expect a 2d tensor map (H, W)
        # il descrittore usa tutta la mappa (come nel paper): una varroa sul bordo
        # del crop deve poter alzare lo score quanto una al centro
        # fp32: in calibrazione la mappa arriva in bf16 (autocast) e std/TV perderebbero precisione
        active_region = anomaly_map.float()
        flat_map = active_region.flatten()
        
        # 1. global dispersion
        sigma_s = flat_map.std()
        
        # 2. tail concentration (top 1%)
        k = max(1, int(0.01 * flat_map.numel()))
        topk_vals, _ = torch.topk(flat_map, k)
        topk_mean = topk_vals.mean()
        
        # 3. spatial roughness (total variation)
        tv_h = torch.abs(active_region[1:, :] - active_region[:-1, :]).mean()
        tv_w = torch.abs(active_region[:, 1:] - active_region[:, :-1]).mean()
        tv_s = tv_h + tv_w
        
        return torch.tensor([sigma_s, topk_mean, tv_s], device=anomaly_map.device)

    @staticmethod
    def compute_anomaly_score(descriptor: torch.Tensor, train_mu: torch.Tensor, train_sigma: torch.Tensor) -> torch.Tensor:
        # standard diagonal mahalanobis distance
        z_scores = (descriptor - train_mu) / (train_sigma + 1e-8)
        struct_score = torch.sqrt(torch.sum(z_scores ** 2))
        return struct_score

    def calculate_anomaly_maps_scores(
        self,
        embedding: Tensor,
        memory_bank: Tensor,
        batch_size: int,
        width: int,
        height: int,
        image_size: tuple,
    ) -> tuple[Tensor, Tensor]:

        if self.feature_extractor.quantized:
            embedding = torch.int_repr(embedding).to(torch.float64)

        # apply nearest neighbor search
        if self.apply_quantization:
            patch_scores, locations = self.nearest_neighbors_quantized(embedding=embedding, n_neighbors=1)
        else:
            patch_scores, locations = self.nearest_neighbors(embedding=embedding, n_neighbors=1, memory_bank=memory_bank)

        # reshape to w,h
        patch_scores = patch_scores.reshape((batch_size, 1, width, height))

        # get the anomaly map
        anomaly_maps = self.anomaly_map_generator(patch_scores, image_size = image_size)

        # compute the structcore anomaly score of the images
        pred_scores = []
        for i in range(batch_size):
            descriptor = StructCore.get_structural_descriptor(anomaly_maps[i].squeeze())
            if self.train_mu is not None and self.train_sigma is not None:
                score = StructCore.compute_anomaly_score(descriptor, self.train_mu, self.train_sigma)
            else:
                score = descriptor.sum()
            pred_scores.append(score)
            
        pred_scores = torch.stack(pred_scores)

        return anomaly_maps, pred_scores

    def train_epoch(
        self, epoch, train_dataloader, training_args: TrainingArgs
    ):
        embeddings = []

        with torch.no_grad():

            print("Embedding Extraction:")
            for batch in tqdm(iter(train_dataloader)):
                with autocast(device_type="cuda" if "cuda" in str(self.device) else "cpu", enabled=True):
                    embedding = self(batch.to(self.device))
                    embeddings.append(embedding)

            embeddings = torch.cat(embeddings, dim = 0)
            torch.cuda.empty_cache()

            #apply coreset reduction
            print("Coreset Extraction:")
            coreset = self.coreset_extractor.extract_coreset(embeddings)

            if self.apply_quantization:
                assert self.product_quantizer is not None, "Product Quantizer not initialized"

                self.product_quantizer.fit(coreset)
                coreset = self.product_quantizer.encode(coreset)

            self.memory_bank = coreset

            print("Fitting StructCore Calibration:")
            descriptors = []
            for batch in tqdm(iter(train_dataloader)):
                with autocast(device_type="cuda" if "cuda" in str(self.device) else "cpu", enabled=True):
                    embedding, batch_size, width, height = self.extract_embedding(batch.to(self.device))
                    anomaly_maps, _ = self.calculate_anomaly_maps_scores(
                        embedding=embedding,
                        memory_bank=self.memory_bank,
                        batch_size=batch_size,
                        width=width,
                        height=height,
                        image_size=batch.shape[2:]
                    )
                    for i in range(batch_size):
                        descriptors.append(StructCore.get_structural_descriptor(anomaly_maps[i].squeeze()))
            if len(descriptors) > 0:
                descriptors = torch.stack(descriptors)
                self.train_mu = descriptors.mean(dim=0)
                self.train_sigma = descriptors.std(dim=0)

    def train_chunk(
        self, train_dataloader, training_args: TrainingArgs
    ):
        embeddings = []

        with torch.no_grad():

            print("Embedding Extraction:")
            for batch in tqdm(iter(train_dataloader)):
                with autocast(device_type="cuda" if "cuda" in str(self.device) else "cpu", enabled=True):
                    embedding = self(batch.to(self.device))
                    embeddings.append(embedding)

            embeddings = torch.cat(embeddings, dim = 0)
            torch.cuda.empty_cache()

            total_embbeddings = torch.cat([embeddings, self.memory_bank], dim=0) if hasattr(self, "memory_bank") else embeddings

            #apply coreset reduction
            print("Coreset Extraction:")
            coreset = self.coreset_extractor.extract_coreset(total_embbeddings)

            if self.apply_quantization:
                assert self.product_quantizer is not None, "Product Quantizer not initialized"

                self.product_quantizer.fit(coreset)
                coreset = self.product_quantizer.encode(coreset)

            self.memory_bank = coreset

            print("Fitting StructCore Calibration:")
            descriptors = []
            for batch in tqdm(iter(train_dataloader)):
                with autocast(device_type="cuda" if "cuda" in str(self.device) else "cpu", enabled=True):
                    embedding, batch_size, width, height = self.extract_embedding(batch.to(self.device))
                    anomaly_maps, _ = self.calculate_anomaly_maps_scores(
                        embedding=embedding,
                        memory_bank=self.memory_bank,
                        batch_size=batch_size,
                        width=width,
                        height=height,
                        image_size=batch.shape[2:]
                    )
                    for i in range(batch_size):
                        descriptors.append(StructCore.get_structural_descriptor(anomaly_maps[i].squeeze()))
            if len(descriptors) > 0:
                descriptors = torch.stack(descriptors)
                self.train_mu = descriptors.mean(dim=0)
                self.train_sigma = descriptors.std(dim=0)

            del embeddings
            del total_embbeddings
            torch.cuda.empty_cache()

    def generate_embedding(self, features: dict[str, Tensor]) -> Tensor:
        """Generate embedding from hierarchical feature map.

        Args:
            features: dict[str:Tensor]: Hierarchical feature map from a CNN

        Returns:
            Embedding vector [Tensor]
        """

        embeddings = [features[self.layers[0]]]

        for layer in self.layers[1:]:
            layer_embedding = features[layer]
            layer_embedding = F.interpolate(layer_embedding, size = embeddings[0].shape[-2], mode = "bilinear")
            if self.quantize:
                layer_embedding = layer_embedding.quantize()
            embeddings.append(layer_embedding)

        embeddings = torch.cat(embeddings, 1)

        return embeddings

    @staticmethod
    def reshape_embedding(embedding: Tensor) -> Tensor:
        """
        Reshape Embedding.

        Reshapes Embedding to the following format:
        [Batch, Embedding, Patch, Patch] to [Batch*Patch*Patch, Embedding]

        Args:
            embedding (Tensor): Embedding tensor extracted from CNN features.

        Returns:
            Tensor: Reshaped embedding tensor.
        """

        embedding_size = embedding.size(1)
        embedding = embedding.permute(0,2,3,1).reshape(-1, embedding_size)
        return embedding

    def euclidean_distance(x: Tensor, y: Tensor, quantized:bool) -> Tensor:
        """
        Calculates pair-wise distance between row vectors in x and those in y.

        Args:
            x: input tensor 1
            y: input tensor 2
            quantized: bool, True if x and y are quantized tensors

        Returns:
            Matrix of distances between row vectors in x and y.
        """

        if quantized:
            return torch.cdist(x.dequantize(), y.dequantize())

        return torch.cdist(x, y)

    def nearest_neighbors(self, embedding: Tensor, n_neighbors: int, memory_bank: torch.Tensor = None) -> tuple[Tensor, Tensor]:
        """
        Nearest neighbors using brute force method and euclidean norm, processed in chunks.
        Uses float16 for ~3x speedup and 2x less VRAM usage.
        """
        if memory_bank is None:
            memory_bank = self.memory_bank

        # Move both to GPU in float16 for fast computation
        device = self.device
        emb_f16 = embedding.to(device).half()
        bank_f16 = memory_bank.to(device).half()

        chunk_size = 4096
        patch_scores_list = []
        locations_list = []

        for i in range(0, emb_f16.size(0), chunk_size):
            chunk = emb_f16[i : i + chunk_size]
            distances = torch.cdist(chunk, bank_f16)

            if n_neighbors == 1:
                scores, locs = distances.min(1)
            else:
                scores, locs = distances.topk(k=n_neighbors, largest=False, dim=1)

            patch_scores_list.append(scores.float())
            locations_list.append(locs)

        patch_scores = torch.cat(patch_scores_list, dim=0)
        locations = torch.cat(locations_list, dim=0)

        return patch_scores, locations

    def nearest_neighbors_quantized(self, embedding: Tensor, n_neighbors: int) -> tuple[Tensor, Tensor]:

        device = self.device
        embedding = embedding.to(device)
        self.memory_bank = self.memory_bank.to(device)

        quantized_embedding = self.product_quantizer.encode(embedding).to(device)

        distances = StructCore.euclidean_distance(
            quantized_embedding,
            self.memory_bank,
            quantized=self.feature_extractor.quantized,
        )

        _, top_locations = distances.topk(
            k=1000, largest=False, dim=1
        )

        quantized_neighbors = self.memory_bank[top_locations]

        B, K, Dq = quantized_neighbors.shape

        decoded_neighbors = self.product_quantizer.decode(
            quantized_neighbors.view(-1, Dq)
        ).view(B, K, -1).to(device)

        embedding_expanded = embedding.unsqueeze(1)

        exact_distances = StructCore.euclidean_distance(
            embedding_expanded,
            decoded_neighbors,
            quantized=False,
        )

        exact_distances = exact_distances.squeeze(1)  # [B, top_k]

        patch_scores, locations = exact_distances.topk(
            k=n_neighbors, largest=False, dim=1
        )

        return patch_scores, locations


    def save(self, save_path: str):
        """
        Save the Patchcore model

        Parameters:
        ----------
            save_path (str): where the model will be saved
        """
        if self.apply_quantization:
            assert self.product_quantizer is not None
            self.product_quantizer.save(save_path + "/product_quantizer.bin")
        state_dict = {
            "memory_bank": self.memory_bank,
            "train_mu": self.train_mu,
            "train_sigma": self.train_sigma
        }
        torch.save(state_dict, save_path)

    def load(self, model_state_dict_patch, quantizer_state_dict_path):
        """
        Load the Patchcore model

        Parameters:
        ----------
            model_state_dict_patch (dict): model state dictionary
            quantizer_state_dict_patch (dict): quantizer state dictionary
        """

        self._load_model(model_state_dict_patch)
        if quantizer_state_dict_path is not None:
            self.product_quantizer = ProductQuantizer()
            self.product_quantizer.load(quantizer_state_dict_path)
            self.apply_quantization = True

    def _load_model(self, path):
        """
        Load the Patchcore memory bank

        Parameters:
        ----------
            path (str): where the pt file containing the memory bank is stored
        """

        state_dict = torch.load(path)

        ## TODO: MemoryBank quantization

        if type(state_dict) is dict and "memory_bank" in state_dict:
            self.memory_bank = state_dict["memory_bank"]
            self.train_mu = state_dict.get("train_mu", None)
            self.train_sigma = state_dict.get("train_sigma", None)
        else:
            self.memory_bank = state_dict
            self.train_mu = None
            self.train_sigma = None

    def reset_model(self):
        self.memory_bank = None

    def save_anomaly_map(self, dirpath, anomaly_map, pred_score, filepath, x_type, mask):
        """
        Args:
            dirpath     (str)       : Output directory path.
            anomaly_map (np.ndarray): Anomaly map with the same size as the input image.
            filepath    (str)       : Path of the input image.
            x_type      (str)       : Anomaly type (e.g. "good", "crack", etc).
            contour     (float)     : Threshold of contour, or None.
        """
        def min_max_norm(image):
            a_min, a_max = image.min(), image.max()
            return (image - a_min) / (a_max - a_min)

        def cvt2heatmap(gray):
            return cv.applyColorMap(np.uint8(gray), cv.COLORMAP_JET)

        # Get the image file name.
        filename = os.path.basename(filepath)

        # Load the image file and resize.
        original_image = cv.imread(filepath)
        original_image = cv.resize(original_image, anomaly_map.shape[:2])

        # Normalize anomaly map for easier visualization.
        anomaly_map_norm = cvt2heatmap(255 * min_max_norm(anomaly_map))

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
        plt.savefig(str(dirpath + f"/{x_type}_{filename}.jpg"))

    def get_model_size(self):

        feautre_extractor_params = self.feature_extractor.get_size()
        memory_bank_params = params_to_mb(count_params(self.memory_bank))

        return {
            "feature_extractor": feautre_extractor_params,
            "memory_bank": memory_bank_params,
            "total": feautre_extractor_params + memory_bank_params
        }
