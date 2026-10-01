import torch.nn as nn
import torch
import torch.nn.functional as F

class UpscalingFeatureExtractor(nn.Module):
    """Feature extractor module.

    Args:
        backbone (str): backbone name.
        layers (list[str]): list of layers used for extraction.
        patch_size (int): side of the neighboring patch aggregation (average pooling).
        upscale (float): the feature map has `upscale` x the size of the first (largest) layer.
    """

    def __init__(self, featurre_extractor, patch_size: int = 3, upscale: float = 2) -> None:
        super().__init__()

        self.feature_extractor = featurre_extractor
        self.upscale = upscale

        self.pooler = nn.AvgPool2d(
            kernel_size=patch_size,
            stride=1,
            padding=patch_size // 2,
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Extract features from input tensor.

        Args:
            input_tensor: input tensor (images)

        Returns:
            (torch.Tensor): extracted feature map.
        """
        features = self.feature_extractor(input_tensor)

        _, _, h, w = features[0].shape
        size = (int(h * self.upscale), int(w * self.upscale))
        feature_map = []
        for layer in features:
            # upscale all to `upscale` x the size of the first (largest), 2x in the paper
            resized = F.interpolate(
                layer,
                size=size,
                mode="bilinear",
            )
            feature_map.append(resized)
        # channel-wise concat
        feature_map = torch.cat(feature_map, dim=1)

        # neighboring patch aggregation
        return self.pooler(feature_map)

    def get_channels_dim(self) -> int:
        return self.feature_extractor.get_channels_dim()