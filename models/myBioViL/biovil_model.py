import time
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from health_multimodal.text import TextInferenceEngine
from health_multimodal.image import ImageInferenceEngine
from typing import Callable, List, Optional
from math import ceil, floor

import pdb


class ImageTextModel(nn.Module):
    def __init__(
        self,
        image_inference_engine: ImageInferenceEngine,
        text_inference_engine: TextInferenceEngine,
        width,
        height,
        print_inference_time=True,
        train_image=False
    ) -> None:
        super(ImageTextModel, self).__init__()
        self.image_inference_engine = image_inference_engine.model
        self.text_inference_engine = text_inference_engine
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.box_head = nn.Sequential(
            # Layer 1
            nn.ConvTranspose2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm([64, 28, 28]),
            nn.ReLU(inplace=True),

            # Layer 2
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm([32, 56, 56]),
            nn.ReLU(inplace=True),

            # Layer 3
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm([16, 112, 112]),
            nn.ReLU(inplace=True),

            # Layer 4
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm([8, 224, 224]),
            nn.ReLU(inplace=True),

            # Layer 5
            nn.ConvTranspose2d(8, 4, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm([4, 448, 448]),
            nn.ReLU(inplace=True),

            # Layer 6
            nn.ConvTranspose2d(4, 2, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm([2, 896, 896]),
            nn.ReLU(inplace=True),

            # Final Conv to ensure correct dimensions
            nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1)
        ).to(device)

        self.width = width
        self.height = height
        self.resize_size, self.crop_size = (
            image_inference_engine.resize_size,
            image_inference_engine.crop_size,
        )
        self.transform = image_inference_engine.transform
        self.print_time = print_inference_time
        self.train_image = train_image


    def get_similarity_maps_from_raw_data(
        self,
        images: torch.Tensor,
        query_text: List[str],
        interpolation: str = "nearest",
    ) -> torch.Tensor:
        """Return a heatmap of the similarities between each patch embedding from the image and the text embedding.

        :param image_path: Path to the input chest X-ray, either a DICOM or JPEG file.
        :param query_text: Input radiology text phrase.
        :param interpolation: Interpolation method to upsample the heatmap so it matches the input image size.
            See :func:`torch.nn.functional.interpolate` for more details.
        :return: A heatmap of the similarities between each patch embedding from the image and the text embedding,
            with the same shape as the input image.
        """
        # Get embedding
        text_embedding = self.text_inference_engine.get_embeddings_from_prompt(
            query_text
        )
        image_embedding = self.image_inference_engine.get_patchwise_projected_embeddings(
            images, normalize=True
        )

        # Compute similarity map
        sim = self._get_similarity_maps_from_embeddings(image_embedding, text_embedding)

        # Resize to original dimension
        resized_sim_maps = self.convert_multiple_similarity_to_image_size(
            sim,
            width=self.width,
            height=self.height,
            resize_size=self.resize_size,
            crop_size=self.crop_size,
            val_img_transform=self.transform,
            interpolation=interpolation,
        )
        return resized_sim_maps
    

    def get_bbox_from_raw_data(
        self,
        images: torch.Tensor,
        query_text: List[str],
    ) -> torch.Tensor:
        """
        Get the bounding box from image and text pairs.
        """
        with torch.no_grad():
            # Get embedding
            text_embedding = self.text_inference_engine.get_embeddings_from_prompt(
                query_text
            )
            image_embedding = self.image_inference_engine.get_patchwise_projected_embeddings(
                images, normalize=True
            )

        # Generate similarity map
        sim = self._get_similarity_maps_from_embeddings(image_embedding, text_embedding)

        # Convert similarity map to bounding box
        bbox = self.convert_similarity_to_bbox(sim)
        return bbox
    

    def convert_similarity_to_bbox(self, similarity_map: torch.Tensor):
        # Flatten similarity map        
        B = similarity_map.shape[0]
        x = similarity_map.unsqueeze(1)

        # Run object detection head
        x = self.box_head(x)

        # Pad due to center crop
        margins_for_pad = (64,64,64,64)

        # Pad with zeros for differentiability instead of NaNs
        x = F.pad(x[:, 0, :], margins_for_pad, value=0.0)
        return x


    @staticmethod
    def _get_similarity_maps_from_embeddings(
        projected_patch_embeddings: torch.Tensor,
        projected_text_embeddings: torch.Tensor,
        sigma: float = 1.5,
    ) -> torch.Tensor:
        """Get smoothed similarity map for a given image patch embeddings and text embeddings.

        :param projected_patch_embeddings: [batch_size, n_patches_h, n_patches_w, feature_size]
        :param projected_text_embeddings: [batch_size, feature_size]
        :return: similarity_map: similarity map of shape [batch_size, n_patches_h, n_patches_w]
        """
        (
            batch_size,
            n_patches_h,
            n_patches_w,
            feature_size,
        ) = projected_patch_embeddings.shape
        assert feature_size == projected_text_embeddings.shape[1]
        assert projected_text_embeddings.shape[0] == batch_size
        assert projected_text_embeddings.dim() == 2
        # patch_wise_similarity = projected_patch_embeddings.view(batch_size, -1, feature_size) @ projected_text_embeddings.t().unsqueeze(0) #TODO: check
        # print(projected_patch_embeddings.shape)
        # print(projected_text_embeddings.shape)
        # similarity_map = patch_wise_similarity.reshape(batch_size, n_patches_h, n_patches_w)
        # return similarity_map
        # similarity_map = torch.einsum("bhwc,bc->bhw", projected_patch_embeddings, projected_text_embeddings)
        cos_sim = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
        similarity_map = cos_sim(projected_patch_embeddings, projected_text_embeddings.unsqueeze(1).unsqueeze(2))
        return similarity_map

    @staticmethod
    def convert_multiple_similarity_to_image_size(
        similarity_map: torch.Tensor,
        width: int,
        height: int,
        resize_size: Optional[int],
        crop_size: Optional[int],
        val_img_transform: Optional[Callable] = None,
        interpolation: str = "nearest",
    ) -> torch.tensor:
        """
        Convert similarity map from raw patch grid to original image size,
        taking into account whether the image has been resized and/or cropped prior to entering the network.
        """
        batch_size, n_patches_h, n_patches_w = (
            similarity_map.shape[0],
            similarity_map.shape[1],
            similarity_map.shape[2],
        )
        target_shape = batch_size, 1, n_patches_h, n_patches_w
        smallest_dimension = min(height, width)

        # TODO:
        # verify_resize_params(val_img_transforms, resize_size, crop_size)

        reshaped_similarity = similarity_map.reshape(target_shape)
        align_corners_modes = "linear", "bilinear", "bicubic", "trilinear"
        align_corners = False if interpolation in align_corners_modes else None

        if crop_size is not None:
            if resize_size is not None:
                cropped_size_orig_space = int(
                    crop_size * smallest_dimension / resize_size
                )
                target_size = cropped_size_orig_space, cropped_size_orig_space
            else:
                target_size = crop_size, crop_size
            # return reshaped_similarity[0, 0]
            similarity_map = F.interpolate(
                reshaped_similarity,
                size=target_size,
                mode=interpolation,
                align_corners=align_corners,
            )
            margin_w, margin_h = (width - target_size[0]), (height - target_size[1])
            margins_for_pad = (
                floor(margin_w / 2),
                ceil(margin_w / 2),
                floor(margin_h / 2),
                ceil(margin_h / 2),
            )

            # Pad with zeros for differentiability instead of NaNs
            similarity_map = F.pad(similarity_map[:, 0, :], margins_for_pad, value=0.0)
        else:
            similarity_map = F.interpolate(
                reshaped_similarity,
                size=(height, width),
                mode=interpolation,
                align_corners=align_corners,
            )[:, 0, :]
        return similarity_map  # .numpy()

    def to(self, device: torch.device) -> None:
        """Move models to the specified device."""
        self.image_inference_engine.to(device)
        self.text_inference_engine.to(device)
