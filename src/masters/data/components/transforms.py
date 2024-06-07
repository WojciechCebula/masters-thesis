from typing import Hashable

import cv2
import numpy as np
import torch

from monai.config import KeysCollection
from monai.data.meta_tensor import MetaTensor
from monai.transforms.transform import MapTransform
from monai.utils.type_conversion import convert_to_numpy
from scipy.ndimage import distance_transform_edt
from skimage.morphology import binary_closing, skeletonize


class LoadYOLOLabeld(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        reference_image_key: Hashable,
        class_num: int,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.reference_image_key = reference_image_key
        self.class_num = class_num

    def __call__(self, data):
        d = dict(data)
        _, H, W = d[self.reference_image_key].shape
        for key in self.key_iterator(d):
            with open(d[key], 'r') as file:
                lines = file.readlines()

            label_mask = np.zeros((self.class_num, H, W), np.int32)
            for line in lines:
                cls, *contour = line.split()
                contour = np.array(contour).reshape(-1, 2).astype(np.float32)
                contour[:, 0] = contour[:, 0] * H
                contour[:, 1] = contour[:, 1] * W
                contour = contour.astype(np.int32)
                temp = np.zeros((512, 512), np.int32)
                cv2.fillPoly(temp, [contour], (1))
                label_mask[int(cls)] = temp
            d[key] = label_mask
        return d


class MaxPoold(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        axis: int = 0,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.axis = axis

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            image = d[key].max(axis=self.axis)
            image = np.expand_dims(image, axis=self.axis)
            d[key] = image
        return d


class FixChannelInconsistency(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            image = d[key]
            if image.shape[0] == 3:
                image = torch.mean(image, axis=0)
                image = torch.unsqueeze(image, axis=0)
            d[key] = MetaTensor(image).astype(d[key].dtype)
        return d


class ButterworthFilterd(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        gH: float = 1.5,
        gL: float = 0.5,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.gH = gH
        self.gL = gL

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            image = self._apply_filter(d[key].squeeze())
            image = np.expand_dims(image, axis=0)
            d[key] = image
        return d

    def _apply_filter(self, image: np.ndarray) -> np.ndarray:
        image_log = np.log1p(np.array(image, dtype='d'))
        image_fft = np.fft.fft2(image_log)
        image_fft = np.fft.fftshift(image_fft)

        filter = self._butterworth_filter(image_fft.shape)
        if self.gH >= 1 and self.gL < 1:
            filter = (self.gH - self.gL) * filter + self.gL

        image_fft_filtered = filter * image_fft
        image_fft_filtered = np.fft.fftshift(image_fft_filtered)
        image_filtered = np.fft.ifft2(image_fft_filtered)
        result_image = np.expm1(np.real(image_filtered))
        result_image = 255 * ((result_image - np.min(result_image)) / (np.max(result_image) - np.min(result_image)))
        return result_image

    @staticmethod
    def _butterworth_filter(image_shape: tuple[int, int], n: float = 2.0, c: float = 1.0, D0: float = 12):
        P = image_shape[0] / 2
        Q = image_shape[1] / 2
        U, V = np.meshgrid(range(image_shape[0]), range(image_shape[1]), sparse=False, indexing='ij')
        Duv = (((U - P) ** 2 + (V - Q) ** 2) ** (1 / 2)).astype(np.dtype('d'))
        h = 1 / (1 + ((c * Duv) / D0) ** (2 * n))
        H = 1 - h
        return H

    def adjust_scale(self, image):
        min_val = np.min(image)
        max_val = np.max(image)
        adjusted_image = ((image - min_val) / (max_val - min_val)) * 255
        return adjusted_image

    def normalize(self, img, M0=128, VAR0=100):  # Desired variance):
        uniform_image = img.copy()

        mean = np.mean(uniform_image)
        variance = np.var(uniform_image)

        mask = uniform_image > mean

        normalized_image = np.zeros_like(uniform_image, dtype=np.float32)
        normalized_image[mask] = M0 + np.sqrt(VAR0 * (uniform_image[mask] - mean) ** 2 / variance)
        normalized_image[~mask] = M0 - np.sqrt(VAR0 * (uniform_image[~mask] - mean) ** 2 / variance)
        normalized_image = self.adjust_scale(normalized_image)

        return normalized_image


class DistanceTransformEDTd(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        distance_upper_bound: float = 20.0,
        allow_missing_keys: bool = False,
        result_dtype: type = np.float32,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.distance_upper_bound = distance_upper_bound
        self.result_dtype = result_dtype

    def __call__(self, data):
        d = dict(data)

        for key in self.key_iterator(d):
            label = d[key]
            affine = label.affine
            label = convert_to_numpy(label).squeeze()
            label = (label == 0).astype(np.uint8)
            distance_map = distance_transform_edt(label)
            distance_map[distance_map > self.distance_upper_bound] = self.distance_upper_bound
            distance_map = distance_map.astype(self.result_dtype)

            d[key].set_array(distance_map[np.newaxis])
            d[key].affine = affine
        return d


class SkeletonizeLabeld(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        threshold: float = 0.5,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.threshold = threshold

    def __call__(self, data):
        d = dict(data)

        for key in self.key_iterator(d):
            label = d[key]
            affine = label.affine
            label = convert_to_numpy(label).squeeze()
            mask = label >= self.threshold
            mask = binary_closing(mask)
            skeleton = skeletonize(mask)
            d[key].set_array(skeleton[np.newaxis])
            d[key].affine = affine
        return d
