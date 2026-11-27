"""
Domain Shift Transforms — Transforms that simulate environmental changes.

These transforms operate on already-normalized tensors (after ToTensor and Normalize)
and are meant to simulate progressive degradation of the acquisition conditions.

Usage:
    from moviad.datasets.domain_shift_transforms import DirtyLensTransform, LowLightTransform

    dirty = DirtyLensTransform(blur_kernel=7, blur_sigma=2.0, num_spots=5, spot_opacity=0.3)
    low_light = LowLightTransform(brightness_factor=0.6, contrast_factor=0.8, noise_std=0.05)

    # Apply to an already-normalized (C, H, W) tensor
    transformed = dirty(image_tensor)
"""

import torch
import torch.nn.functional as F
import random


class DirtyLensTransform:
    """Simulates dust or propolis deposited on the camera.

    Applies a Gaussian blur localized on the image borders and adds
    random semi-transparent spots to simulate dirt on the lens.

    Args:
        blur_kernel (int): Kernel size for the border blur (must be odd).
        blur_sigma (float): Sigma of the Gaussian blur.
        border_ratio (float): Fraction of the image considered "border" (0.0-0.5).
        num_spots (int): Number of semi-transparent spots to add.
        spot_radius_range (tuple): (min, max) range of the spot radius in pixels.
        spot_opacity (float): Spot opacity (0.0 = transparent, 1.0 = opaque).
    """

    def __init__(
        self,
        blur_kernel: int = 7,
        blur_sigma: float = 2.0,
        border_ratio: float = 0.15,
        num_spots: int = 5,
        spot_radius_range: tuple = (10, 30),
        spot_opacity: float = 0.3,
    ):
        self.blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        self.blur_sigma = blur_sigma
        self.border_ratio = border_ratio
        self.num_spots = num_spots
        self.spot_radius_range = spot_radius_range
        self.spot_opacity = spot_opacity

    def _create_gaussian_kernel_2d(self, kernel_size: int, sigma: float) -> torch.Tensor:
        """Creates a 2D Gaussian kernel."""
        x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        gauss_1d = torch.exp(-0.5 * (x / sigma) ** 2)
        gauss_2d = gauss_1d[:, None] * gauss_1d[None, :]
        return gauss_2d / gauss_2d.sum()

    def _apply_border_blur(self, image: torch.Tensor) -> torch.Tensor:
        """Applies a Gaussian blur only to the image borders."""
        C, H, W = image.shape

        # Create the blurred image
        kernel = self._create_gaussian_kernel_2d(self.blur_kernel, self.blur_sigma)
        kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(C, 1, 1, 1).to(image.device)
        padding = self.blur_kernel // 2
        blurred = F.conv2d(
            image.unsqueeze(0), kernel, padding=padding, groups=C
        ).squeeze(0)

        # Create the border mask (1.0 = border, 0.0 = center)
        border_h = int(H * self.border_ratio)
        border_w = int(W * self.border_ratio)

        mask = torch.ones(H, W, device=image.device)
        if border_h > 0 and border_w > 0:
            mask[border_h:-border_h, border_w:-border_w] = 0.0

        # Feather the mask for a smooth transition
        smooth_kernel = self._create_gaussian_kernel_2d(15, 5.0)
        smooth_kernel = smooth_kernel.unsqueeze(0).unsqueeze(0).to(image.device)
        mask_smooth = F.conv2d(
            mask.unsqueeze(0).unsqueeze(0), smooth_kernel, padding=7
        ).squeeze(0).squeeze(0).clamp(0, 1)

        # Blend: blurred borders, sharp center
        return image * (1 - mask_smooth) + blurred * mask_smooth

    def _add_spots(self, image: torch.Tensor) -> torch.Tensor:
        """Adds random semi-transparent spots (simulates drops/propolis)."""
        C, H, W = image.shape

        for _ in range(self.num_spots):
            # Random position, preferably towards the borders
            if random.random() < 0.7:
                # 70% probability: spot on the borders
                side = random.choice(["top", "bottom", "left", "right"])
                if side == "top":
                    cy = random.randint(0, int(H * 0.25))
                    cx = random.randint(0, W - 1)
                elif side == "bottom":
                    cy = random.randint(int(H * 0.75), H - 1)
                    cx = random.randint(0, W - 1)
                elif side == "left":
                    cy = random.randint(0, H - 1)
                    cx = random.randint(0, int(W * 0.25))
                else:
                    cy = random.randint(0, H - 1)
                    cx = random.randint(int(W * 0.75), W - 1)
            else:
                cy = random.randint(0, H - 1)
                cx = random.randint(0, W - 1)

            radius = random.randint(*self.spot_radius_range)

            # Create a feathered circular mask
            y_grid, x_grid = torch.meshgrid(
                torch.arange(H, device=image.device, dtype=torch.float32),
                torch.arange(W, device=image.device, dtype=torch.float32),
                indexing="ij",
            )
            dist = ((y_grid - cy) ** 2 + (x_grid - cx) ** 2).sqrt()
            spot_mask = (1 - (dist / radius).clamp(0, 1)) * self.spot_opacity

            # Spot color (beige/yellowish, simulates propolis)
            spot_color = torch.tensor(
                [0.6 + random.random() * 0.2,
                 0.5 + random.random() * 0.2,
                 0.3 + random.random() * 0.1],
                device=image.device,
            ).view(3, 1, 1)

            if C >= 3:
                image[:3] = image[:3] * (1 - spot_mask) + spot_color * spot_mask
            else:
                image = image * (1 - spot_mask) + spot_color[:C] * spot_mask

        return image

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        image = self._apply_border_blur(image)
        image = self._add_spots(image)
        return image

    def __repr__(self) -> str:
        return (
            f"DirtyLensTransform(blur_kernel={self.blur_kernel}, "
            f"blur_sigma={self.blur_sigma}, border_ratio={self.border_ratio}, "
            f"num_spots={self.num_spots}, spot_opacity={self.spot_opacity})"
        )


class LowLightTransform:
    """Simulates low-light conditions.

    Reduces global brightness, alters contrast and adds
    Gaussian noise to simulate a change in lighting conditions.

    Args:
        brightness_factor (float): Brightness reduction factor (0.0-1.0, lower = darker).
        contrast_factor (float): Contrast factor (< 1.0 = less contrast).
        noise_std (float): Standard deviation of the Gaussian noise.
    """

    def __init__(
        self,
        brightness_factor: float = 0.6,
        contrast_factor: float = 0.8,
        noise_std: float = 0.05,
    ):
        self.brightness_factor = brightness_factor
        self.contrast_factor = contrast_factor
        self.noise_std = noise_std

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        # 1. Brightness reduction
        image = image * self.brightness_factor

        # 2. Contrast change (towards the mean)
        mean = image.mean()
        image = (image - mean) * self.contrast_factor + mean

        # 3. Add Gaussian noise
        noise = torch.randn_like(image) * self.noise_std
        image = image + noise

        return image

    def __repr__(self) -> str:
        return (
            f"LowLightTransform(brightness={self.brightness_factor}, "
            f"contrast={self.contrast_factor}, noise_std={self.noise_std})"
        )
