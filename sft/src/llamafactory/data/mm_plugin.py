# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's Transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/llava/processing_llava.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import math
import re
from copy import deepcopy
from dataclasses import dataclass
from io import BytesIO
from typing import TYPE_CHECKING, BinaryIO, Literal, Optional, TypedDict, Union, Any, Dict, List

import numpy as np
import torch
from transformers.image_utils import get_image_size, to_numpy_array
from typing_extensions import override

from torchvision import transforms as TF
from datetime import datetime

import os
import av
from PIL import Image, ImageDraw, ImageFont

from ..extras.constants import AUDIO_PLACEHOLDER, IGNORE_INDEX, IMAGE_PLACEHOLDER, VIDEO_PLACEHOLDER
from ..extras.packages import (
    is_librosa_available,
    is_pillow_available,
    is_pyav_available,
    is_transformers_version_greater_than,
)

import json

if is_librosa_available():
    import librosa


if is_pillow_available():
    from PIL import Image
    from PIL.Image import Image as ImageObject


if is_pyav_available():
    import av


if is_transformers_version_greater_than("4.45.0"):
    from transformers.models.mllama.processing_mllama import (
        convert_sparse_cross_attention_mask_to_dense,
        get_cross_attention_token_mask,
    )


if is_transformers_version_greater_than("4.49.0"):
    from transformers.image_utils import make_batched_videos, make_flat_list_of_images


if TYPE_CHECKING:
    from av.stream import Stream
    from numpy.typing import NDArray
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
    from transformers.image_processing_utils import BaseImageProcessor

    class EncodedImage(TypedDict):
        path: Optional[str]
        bytes: Optional[bytes]

    ImageInput = Union[str, bytes, EncodedImage, BinaryIO, ImageObject]
    VideoInput = Union[str, BinaryIO, list[str]]  # Modified to reflect new flexible type
    AudioInput = Union[str, BinaryIO, NDArray]

    class MMProcessor(ProcessorMixin):
        patch_size: int
        image_seq_length: int
        num_additional_image_tokens: int
        vision_feature_select_strategy: Literal["default", "full"]

        def _get_number_of_features(self, orig_height: int, orig_width: int, height: int, width: int) -> int:
            pass


def _get_paligemma_token_type_ids(imglens: list[int], seqlens: list[int], processor: "MMProcessor") -> list[list[int]]:
    r"""Get paligemma token type ids for computing loss.

    It is slightly different with the original token type ids where the prompt part is 0.

    Returns:
        batch_token_type_ids: shape (batch_size, seq_length)

    """
    batch_token_type_ids = []
    for imglen, seqlen in zip(imglens, seqlens):
        image_seqlen = imglen * processor.image_seq_length
        batch_token_type_ids.append([0] * image_seqlen + [1] * (seqlen - image_seqlen))

    return batch_token_type_ids


def _get_gemma3_token_type_ids(batch_ids: list[list[int]], processor: "MMProcessor"):
    r"""Get gemma3 token type ids for computing loss.

    Returns:
        batch_token_type_ids: shape (batch_size, seq_length)

    """
    image_token_id: int = getattr(processor, "image_token_id")
    batch_token_type_ids = []
    for token_ids in batch_ids:
        token_ids = np.array(token_ids)
        token_type_ids = np.zeros_like(token_ids)
        token_type_ids[token_ids == image_token_id] = 1
        batch_token_type_ids.append(token_type_ids.tolist())

    return batch_token_type_ids


def _make_batched_images(images: list["ImageObject"], imglens: list[int]) -> list[list["ImageObject"]]:
    r"""Make nested list of images."""
    batch_images = []
    for imglen in imglens:
        batch_images.append(images[:imglen])
        images = images[imglen:]

    return batch_images


@dataclass
class MMPluginMixin:
    image_token: Optional[str]
    video_token: Optional[str]
    audio_token: Optional[str]
    expand_mm_tokens: bool = True

    def _validate_input(
        self,
        processor: Optional["MMProcessor"],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
    ) -> None:
        r"""Validate if this model accepts the input modalities."""
        image_processor: BaseImageProcessor = getattr(processor, "image_processor", None)
        video_processor: BaseImageProcessor = getattr(
            processor, "video_processor", getattr(processor, "image_processor", None)
        )
        feature_extractor: SequenceFeatureExtractor = getattr(processor, "feature_extractor", None)
        if len(images) != 0 and self.image_token is None:
            raise ValueError(
                "This model does not support image input. Please check whether the correct `template` is used."
            )

        if len(videos) != 0 and self.video_token is None:
            raise ValueError(
                "This model does not support video input. Please check whether the correct `template` is used."
            )

        if len(audios) != 0 and self.audio_token is None:
            raise ValueError(
                "This model does not support audio input. Please check whether the correct `template` is used."
            )

        if self.image_token is not None and processor is None:
            raise ValueError("Processor was not found, please check and update your model file.")

        if self.image_token is not None and image_processor is None:
            raise ValueError("Image processor was not found, please check and update your model file.")

        if self.video_token is not None and video_processor is None:
            raise ValueError("Video processor was not found, please check and update your model file.")

        if self.audio_token is not None and feature_extractor is None:
            raise ValueError("Audio feature extractor was not found, please check and update your model file.")

    def _validate_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
    ):
        r"""Validate if the number of images, videos and audios match the number of placeholders in messages."""
        num_image_tokens, num_video_tokens, num_audio_tokens = 0, 0, 0
        for message in messages:
            num_image_tokens += message["content"].count(IMAGE_PLACEHOLDER)
            num_video_tokens += message["content"].count(VIDEO_PLACEHOLDER)
            num_audio_tokens += message["content"].count(AUDIO_PLACEHOLDER)

        if len(images) != num_image_tokens:
            raise ValueError(
                f"The number of images does not match the number of {IMAGE_PLACEHOLDER} tokens in {messages}."
            )

        if len(videos) != num_video_tokens:
            raise ValueError(
                f"The number of videos does not match the number of {VIDEO_PLACEHOLDER} tokens in {messages}."
            )

        if len(audios) != num_audio_tokens:
            raise ValueError(
                f"The number of audios does not match the number of {AUDIO_PLACEHOLDER} tokens in {messages}."
            )

    def _preprocess_image(
        self, image: "ImageObject", image_max_pixels: int, image_min_pixels: int, **kwargs
    ) -> "ImageObject":
        r"""Pre-process a single image."""
        if (image.width * image.height) > image_max_pixels:
            resize_factor = math.sqrt(image_max_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if (image.width * image.height) < image_min_pixels:
            resize_factor = math.sqrt(image_min_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if image.mode != "RGB":
            image = image.convert("RGB")

        return image

    def _get_video_sample_indices(
        self, video_stream: "Stream", video_fps: float, video_maxlen: int, **kwargs
    ) -> list[int]:
        r"""Compute video sample indices according to fps."""
        total_frames = video_stream.frames
        if total_frames == 0:  # infinite video
            return np.linspace(0, video_maxlen - 1, video_maxlen).astype(np.int32)

        sample_frames = max(1, math.floor(float(video_stream.duration * video_stream.time_base) * video_fps))
        sample_frames = min(total_frames, video_maxlen, sample_frames)
        return np.linspace(0, total_frames - 1, sample_frames).astype(np.int32)

    def _regularize_images(self, images: list["ImageInput"], **kwargs) -> dict[str, list["ImageObject"]]:
        r"""Regularize images to avoid error. Including reading and pre-processing."""
        results = []
        for image in images:
            if isinstance(image, (str, BinaryIO)):
                image = Image.open(image)
            elif isinstance(image, bytes):
                image = Image.open(BytesIO(image))
            elif isinstance(image, dict):
                if image["bytes"] is not None:
                    image = Image.open(BytesIO(image["bytes"]))
                else:
                    image = Image.open(image["path"])

            if not isinstance(image, ImageObject):
                raise ValueError(f"Expect input is a list of images, but got {type(image)}.")

            results.append(self._preprocess_image(image, **kwargs))

        return {"images": results}

    def _regularize_videos(self, videos: list["VideoInput"], **kwargs) -> dict[str, list[list["ImageObject"]]]:
        r"""Regularizes videos to avoid error. Including reading, resizing and converting."""
        results = []
        for video in videos:
            container = av.open(video, "r")
            video_stream = next(stream for stream in container.streams if stream.type == "video")
            sample_indices = self._get_video_sample_indices(video_stream, **kwargs)
            frames: list[ImageObject] = []
            container.seek(0)
            for frame_idx, frame in enumerate(container.decode(video_stream)):
                if frame_idx in sample_indices:
                    frames.append(frame.to_image())

            frames = self._regularize_images(frames, **kwargs)["images"]
            results.append(frames)

        return {"videos": results}

    def _regularize_audios(
        self, audios: list["AudioInput"], sampling_rate: float, **kwargs
    ) -> dict[str, Union[list["NDArray"], list[float]]]:
        r"""Regularizes audios to avoid error. Including reading and resampling."""
        results, sampling_rates = [], []
        for audio in audios:
            if isinstance(audio, (str, BinaryIO)):
                audio, sampling_rate = librosa.load(audio, sr=sampling_rate)

            if not isinstance(audio, np.ndarray):
                raise ValueError(f"Expect input is a list of audios, but got {type(audio)}.")

            results.append(audio)
            sampling_rates.append(sampling_rate)

        return {"audios": results, "sampling_rates": sampling_rates}

    def _get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: "MMProcessor",
        imglens: Optional[list[int]] = None,
    ) -> dict[str, "torch.Tensor"]:
        r"""Process visual inputs.

        Returns: (llava and paligemma)
            pixel_values: tensor with shape (B, C, H, W)

        Returns: (qwen2-vl)
            pixel_values: tensor with shape (num_patches, patch_dim)
            image_grid_thw: tensor with shape (num_images, 3), where the three numbers are time, width, height
                            where num_patches == torch.prod(image_grid_thw)

        Returns: (mllama)
            pixel_values: tensor with shape
                          (batch_size, max_num_images, max_image_tiles, channels, tile_height, tile_width)
                          For example, (2, 1, 4, 3, 560, 560).
            aspect_ratio_ids: tensor with shape (batch_size, max_num_images). For example, (2, 1).
            aspect_ratio_mask: tensor with shape (batch_size, max_num_images, max_image_tiles). For example, (2, 1, 4).
            num_tiles: List[List[int]] with shape (batch_size, num_images_in_batch). For example, (2, 1).

        """
        mm_inputs = {}
        if len(images) != 0:
            image_processor: BaseImageProcessor = getattr(processor, "image_processor", None)
            images = self._regularize_images(
                images,
                image_max_pixels=getattr(processor, "image_max_pixels", 768 * 768),
                image_min_pixels=getattr(processor, "image_min_pixels", 32 * 32),
            )["images"]
            if imglens is not None:  # if imglens are provided, make batched images
                images = _make_batched_images(images, imglens)

            image_processor_kwargs = {}
            if getattr(processor, "image_do_pan_and_scan", False):  # gemma3 image processor
                image_processor_kwargs.update(
                    {
                        "do_pan_and_scan": True,
                        "pan_and_scan_min_crop_size": 256,
                        "pan_and_scan_max_num_crops": 4,
                        "pan_and_scan_min_ratio_to_activate": 1.2,
                    }
                )

            mm_inputs.update(image_processor(images, return_tensors="pt", **image_processor_kwargs))

        if len(videos) != 0:
            video_processor: BaseImageProcessor = getattr(
                processor, "video_processor", getattr(processor, "image_processor", None)
            )
            videos = self._regularize_videos(
                videos,
                image_max_pixels=getattr(processor, "video_max_pixels", 256 * 256),
                image_min_pixels=getattr(processor, "video_min_pixels", 16 * 16),
                video_fps=getattr(processor, "video_fps", 1.0),
                video_maxlen=getattr(processor, "video_maxlen", 128),
            )["videos"]
            if "videos" in inspect.signature(video_processor.preprocess).parameters:  # for qwen2_vl and video_llava
                mm_inputs.update(video_processor(images=None, videos=videos, return_tensors="pt"))
            else:  # for llava_next_video
                mm_inputs.update(video_processor(videos, return_tensors="pt"))

        if len(audios) != 0:
            feature_extractor: SequenceFeatureExtractor = getattr(processor, "feature_extractor", None)
            audios = self._regularize_audios(
                audios,
                sampling_rate=getattr(processor, "audio_sampling_rate", 16000),
            )["audios"]
            mm_inputs.update(
                feature_extractor(
                    audios,
                    sampling_rate=getattr(processor, "audio_sampling_rate", 16000),
                    return_attention_mask=True,
                    padding="max_length",
                    return_tensors="pt",
                )
            )
            mm_inputs["feature_attention_mask"] = mm_inputs.pop("attention_mask")  # prevent conflicts

        return mm_inputs


@dataclass
class BasePlugin(MMPluginMixin):
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
        r"""Pre-process input messages before tokenization for VLMs."""
        self._validate_input(processor, images, videos, audios)
        return messages

    def process_token_ids(
        self,
        input_ids: list[int],
        labels: Optional[list[int]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        tokenizer: "PreTrainedTokenizer",
        processor: Optional["MMProcessor"],
    ) -> tuple[list[int], Optional[list[int]]]:
        r"""Pre-process token ids after tokenization for VLMs."""
        self._validate_input(processor, images, videos, audios)
        return input_ids, labels

    def get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        extra_info: list[Any],
        imglens: list[int],
        vidlens: list[int],
        audlens: list[int],
        batch_ids: list[list[int]],
        processor: Optional["MMProcessor"],
    ) -> dict[str, Union[list[int], "torch.Tensor"]]:
        r"""Build batched multimodal inputs for VLMs.

        Arguments:
            images: a list of image inputs, shape (num_images,)
            videos: a list of video inputs, shape (num_videos,)
            audios: a list of audio inputs, shape (num_audios,)
            imglens: number of images in each sample, shape (batch_size,)
            vidlens: number of videos in each sample, shape (batch_size,)
            audlens: number of audios in each sample, shape (batch_size,)
            batch_ids: token ids of input samples, shape (batch_size, seq_len)
            processor: a processor for pre-processing images and videos

        """
        self._validate_input(processor, images, videos, audios)
        return self._get_mm_inputs(images, videos, audios, extra_info, processor)


@dataclass
class Gemma3Plugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        num_image_tokens = 0
        messages = deepcopy(messages)
        boi_token: str = getattr(processor, "boi_token")
        full_image_sequence: str = getattr(processor, "full_image_sequence")
        image_str = full_image_sequence if self.expand_mm_tokens else boi_token

        do_pan_and_scan: bool = getattr(processor, "image_do_pan_and_scan", False)
        if do_pan_and_scan:
            mm_inputs = self._get_mm_inputs(images, videos, audios, processor)

        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                if do_pan_and_scan:
                    image_placeholder_str = (
                        "Here is the original image {{image}} and here are some crops to help you see better "
                        + " ".join(["{{image}}"] * mm_inputs["num_crops"][0][num_image_tokens])
                    )
                else:
                    image_placeholder_str = "{{image}}"

                content = content.replace(IMAGE_PLACEHOLDER, image_placeholder_str, 1)
                num_image_tokens += 1

            message["content"] = content.replace("{{image}}", image_str)

        return messages

    @override
    def get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        imglens: list[int],
        vidlens: list[int],
        audlens: list[int],
        batch_ids: list[list[int]],
        processor: Optional["MMProcessor"],
    ) -> dict[str, Union[list[int], "torch.Tensor"]]:
        self._validate_input(processor, images, videos, audios)
        mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
        mm_inputs.pop("num_crops", None)
        mm_inputs["token_type_ids"] = _get_gemma3_token_type_ids(batch_ids, processor)
        return mm_inputs


@dataclass
class InternVLPlugin(BasePlugin):
    @override
    def _get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: "ProcessorMixin",
        **kwargs,
    ) -> dict[str, "torch.Tensor"]:
        image_processor: BaseImageProcessor = getattr(processor, "image_processor")
        image_processor_kwargs = {}
        if getattr(processor, "crop_to_patches", False):
            image_processor_kwargs.update(
                {
                    "crop_to_patches": True,
                    "max_patches": 12,
                    "min_patches": 1,
                }
            )

        mm_inputs = {}
        image_video_patches = []

        if len(images) != 0 and isinstance(images[0], str):
            images = self._regularize_images(
                images,
                image_max_pixels=getattr(processor, "image_max_pixels", 1024 * 1024),
                image_min_pixels=getattr(processor, "image_min_pixels", 32 * 32),
            )["images"]

        if len(videos) != 0 and isinstance(videos[0], str):
            videos = self._regularize_videos(
                videos,
                image_max_pixels=getattr(processor, "video_max_pixels", 256 * 256),
                image_min_pixels=getattr(processor, "video_min_pixels", 16 * 16),
                video_fps=getattr(processor, "video_fps", 2.0),
                video_maxlen=getattr(processor, "video_maxlen", 128),
            )["videos"]

        if len(images) != 0:
            images = make_flat_list_of_images(images)
            image_inputs = image_processor(images=images, return_tensors="pt", **image_processor_kwargs)
            image_num_patches = image_inputs.pop("num_patches")
            image_pixel_values = image_inputs.pop("pixel_values")
            image_num_patches_indices = np.cumsum(image_num_patches)

        if len(videos) != 0:
            videos = make_batched_videos(videos)
            num_frames_per_video = [len(video) for video in videos]
            patch_indices = np.cumsum(num_frames_per_video)
            image_processor_kwargs["crop_to_patches"] = False
            video_inputs = image_processor(images=videos, return_tensors="pt", **image_processor_kwargs)
            video_num_patches = video_inputs.pop("num_patches")
            video_pixel_values = video_inputs.pop("pixel_values")
            video_num_patches_indices = np.cumsum(video_num_patches)

        # NOT SUPPORT IMAGE VIDEO INTERLEAVED
        if len(images) != 0 and image_pixel_values is not None:
            for i in range(len(images)):
                start_index = image_num_patches_indices[i - 1] if i > 0 else 0
                end_index = image_num_patches_indices[i]
                image_video_patches.append(image_pixel_values[start_index:end_index])

        if len(videos) != 0 and video_pixel_values is not None:
            patch_indices_with_prefix = [0] + list(patch_indices)
            for i in range(len(videos)):
                current_patch_index = patch_indices_with_prefix[i]
                end_patch_index = patch_indices_with_prefix[i + 1]
                start_index = video_num_patches_indices[current_patch_index - 1] if i > 0 else 0
                end_index = video_num_patches_indices[end_patch_index - 1]
                image_video_patches.append(video_pixel_values[start_index:end_index])

        if len(images) != 0 or len(videos) != 0:
            mm_inputs["pixel_values"] = torch.cat(image_video_patches, dim=0)

        if len(images) != 0:
            mm_inputs.update({"image_num_patches": image_num_patches})

        if len(videos) != 0:
            mm_inputs.update({"video_patch_indices": patch_indices})
            mm_inputs.update({"video_num_patches": video_num_patches})

        return mm_inputs

    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: Optional["ProcessorMixin"],
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        num_image_tokens, num_video_tokens = 0, 0
        image_seqlen = getattr(processor, "image_seq_length") if self.expand_mm_tokens else 1
        messages = deepcopy(messages)
        mm_inputs = self._get_mm_inputs(images, videos, audios, processor)

        image_pixel_patch_list = mm_inputs.get("image_num_patches")  # patches of images
        video_num_patches = mm_inputs.get("video_num_patches")  # all patches for frames of videos
        video_patch_indices = mm_inputs.get("video_patch_indices")  # num frames per video

        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                content = content.replace(
                    IMAGE_PLACEHOLDER,
                    f"<img>{'<IMG_CONTEXT>' * image_seqlen * image_pixel_patch_list[num_image_tokens]}</img>",
                    1,
                )
                num_image_tokens += 1

            while VIDEO_PLACEHOLDER in content:
                current_patch_index = video_patch_indices[num_video_tokens - 1] if num_video_tokens > 0 else 0
                end_patch_index = video_patch_indices[num_video_tokens]
                num_patches = list(video_num_patches[current_patch_index:end_patch_index])
                video_replaced_prompt = "\n".join(
                    f"Frame{i + 1}: <img>{'<IMG_CONTEXT>' * image_seqlen * num_patches[i]}</img>"
                    for i in range(len(num_patches))
                )
                content = content.replace(VIDEO_PLACEHOLDER, video_replaced_prompt, 1)
                num_video_tokens += 1

            message["content"] = content

        return messages

    @override
    def get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        imglens: list[int],
        vidlens: list[int],
        audlens: list[int],
        batch_ids: list[list[int]],
        processor: Optional["ProcessorMixin"],
    ) -> dict[str, Union[list[int], "torch.Tensor"]]:
        self._validate_input(processor, images, videos, audios)
        mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
        mm_inputs.pop("image_num_patches", None)
        mm_inputs.pop("video_patch_indices", None)
        mm_inputs.pop("video_num_patches", None)
        return mm_inputs


class KimiVLPlugin(BasePlugin):
    @override
    def process_messages(self, messages, images, videos, audios, processor):
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        if self.expand_mm_tokens:
            mm_inputs = self._get_mm_inputs(images, videos, audios, processor)

        image_grid_hws = mm_inputs.get("image_grid_hws", [])
        num_image_tokens = 0
        image_processor: BaseImageProcessor = getattr(processor, "image_processor")
        merge_length = math.prod(image_processor.merge_kernel_size)
        messages = deepcopy(messages)
        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                image_seqlen = image_grid_hws[num_image_tokens].prod() // merge_length if self.expand_mm_tokens else 1
                content = content.replace(
                    IMAGE_PLACEHOLDER,
                    f"<|media_start|>image<|media_content|>{self.image_token * image_seqlen}<|media_end|>",
                    1,
                )
                num_image_tokens += 1

            message["content"] = content

        return messages


@dataclass
class Llama4Plugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        if self.expand_mm_tokens:
            mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
            if "pixel_values" in mm_inputs:
                image_height, image_width = mm_inputs["pixel_values"][0].shape[-2:]
                num_patches_per_chunk = int(
                    (image_height // processor.patch_size)
                    * (image_width // processor.patch_size)
                    // processor.downsample_ratio
                )
                aspect_ratios = mm_inputs.pop("aspect_ratios")

        num_image_tokens = 0
        messages = deepcopy(messages)
        for message in messages:
            content = message["content"]
            if self.expand_mm_tokens:
                placeholder_count = content.count(IMAGE_PLACEHOLDER)
                prompt_splits = content.split(IMAGE_PLACEHOLDER)
                new_content = []
                for local_image_index, split_part in enumerate(prompt_splits):
                    new_content.append(split_part)
                    if local_image_index < placeholder_count:
                        tokens_for_this_image = processor._prompt_split_image(
                            aspect_ratios[num_image_tokens], num_patches_per_chunk
                        )
                        num_image_tokens += 1
                        new_content.append(tokens_for_this_image)

                content = "".join(new_content)
            else:
                content = content.replace(IMAGE_PLACEHOLDER, self.image_token)

            message["content"] = content

        return messages

    @override
    def get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        imglens: list[int],
        vidlens: list[int],
        audlens: list[int],
        batch_ids: list[list[int]],
        processor: Optional["MMProcessor"],
    ) -> dict[str, Union[list[int], "torch.Tensor"]]:
        self._validate_input(processor, images, videos, audios)
        mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
        mm_inputs.pop("aspect_ratios", None)
        return mm_inputs


@dataclass
class LlavaPlugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        messages = deepcopy(messages)
        if self.expand_mm_tokens:
            mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
            if "pixel_values" in mm_inputs:
                height, width = get_image_size(to_numpy_array(mm_inputs["pixel_values"][0]))
                image_seqlen = (height // processor.patch_size) * (
                    width // processor.patch_size
                ) + processor.num_additional_image_tokens
                if processor.vision_feature_select_strategy == "default":
                    image_seqlen -= 1
        else:
            image_seqlen = 1

        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                content = content.replace(IMAGE_PLACEHOLDER, "{{image}}" * image_seqlen, 1)

            message["content"] = content.replace("{{image}}", self.image_token)

        return messages


@dataclass
class LlavaNextPlugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        num_image_tokens = 0
        messages = deepcopy(messages)
        if self.expand_mm_tokens:
            mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
            if "pixel_values" in mm_inputs:
                image_sizes = iter(mm_inputs["image_sizes"].tolist())
                height, width = get_image_size(to_numpy_array(mm_inputs["pixel_values"][0][0]))

        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                if self.expand_mm_tokens:
                    orig_height, orig_width = next(image_sizes)
                    image_seqlen = processor._get_number_of_features(orig_height, orig_width, height, width)
                    if processor.vision_feature_select_strategy == "default":
                        image_seqlen -= 1
                else:
                    image_seqlen = 1

                content = content.replace(IMAGE_PLACEHOLDER, "{{image}}" * image_seqlen, 1)
                num_image_tokens += 1

            message["content"] = content.replace("{{image}}", self.image_token)

        return messages


@dataclass
class LlavaNextVideoPlugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        messages = deepcopy(messages)
        if self.expand_mm_tokens:
            mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
            if "pixel_values" in mm_inputs:
                image_sizes = iter(mm_inputs["image_sizes"].tolist())
                height, width = get_image_size(to_numpy_array(mm_inputs["pixel_values"][0][0]))

        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                if self.expand_mm_tokens:
                    orig_height, orig_width = next(image_sizes)
                    image_seqlen = processor._get_number_of_features(orig_height, orig_width, height, width)
                    if processor.vision_feature_select_strategy == "default":
                        image_seqlen -= 1
                else:
                    image_seqlen = 1

                content = content.replace(IMAGE_PLACEHOLDER, "{{image}}" * image_seqlen, 1)

            message["content"] = content.replace("{{image}}", self.image_token)

        if self.expand_mm_tokens:
            if "pixel_values_videos" in mm_inputs:
                one_video = to_numpy_array(mm_inputs.get("pixel_values_videos")[0])
                height, width = get_image_size(one_video[0])
                num_frames = one_video.shape[0]
                image_seqlen = (height // processor.patch_size) * (width // processor.patch_size)
                video_seqlen = image_seqlen // 4 * num_frames
        else:
            video_seqlen = 1

        for message in messages:
            content = message["content"]
            while VIDEO_PLACEHOLDER in content:
                content = content.replace(VIDEO_PLACEHOLDER, "{{video}}" * video_seqlen, 1)

            message["content"] = content.replace("{{video}}", self.video_token)

        return messages


@dataclass
class MiniCPMVPlugin(BasePlugin):
    @override
    def _get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: "MMProcessor",
        **kwargs,
    ) -> dict[str, "torch.Tensor"]:
        image_processor: BaseImageProcessor = getattr(processor, "image_processor")
        mm_inputs = {}
        if len(images) != 0:
            images = self._regularize_images(
                images,
                image_max_pixels=getattr(processor, "image_max_pixels", 768 * 768),
                image_min_pixels=getattr(processor, "image_min_pixels", 32 * 32),
            )["images"]
            if "valid_image_nums_ls" in kwargs:
                valid_image_nums_ls = kwargs["valid_image_nums_ls"]
                new_images = []
                idx = 0
                for valid_image_nums in valid_image_nums_ls:
                    new_images.append(images[idx : idx + valid_image_nums])
                    idx += valid_image_nums

                images = new_images

            image_inputs = image_processor(
                images, do_pad=True, max_slice_nums=image_processor.max_slice_nums, return_tensors="pt"
            )
            mm_inputs.update(image_inputs)

        if len(videos) != 0:
            videos = self._regularize_videos(
                videos,
                image_max_pixels=getattr(processor, "video_max_pixels", 256 * 256),
                image_min_pixels=getattr(processor, "video_min_pixels", 16 * 16),
                video_fps=getattr(processor, "video_fps", 2.0),
                video_maxlen=getattr(processor, "video_maxlen", 128),
            )["videos"]
            video_inputs = image_processor(videos, do_pad=True, max_slice_nums=2, return_tensors="pt")
            mm_inputs.update(video_inputs)

        if len(audios) != 0:
            audios = self._regularize_audios(
                audios,
                sampling_rate=getattr(processor, "audio_sampling_rate", 16000),
            )["audios"]
            if "valid_audio_nums_ls" in kwargs:
                valid_audio_nums_ls = kwargs["valid_audio_nums_ls"]
                audios_ls = []
                idx = 0
                for valid_audio_nums in valid_audio_nums_ls:
                    audios_ls.append(audios[idx : idx + valid_audio_nums])
                    idx += valid_audio_nums
            else:
                audios_ls = [audios]

            audio_features, audio_feature_lens, audio_phs = processor.audio_feature_extract(
                audios_ls,
                chunk_input=True,
                sampling_rate=getattr(processor, "audio_sampling_rate", 16000),
            )
            audio_feature_lens = [torch.tensor(audio_feature_len) for audio_feature_len in audio_feature_lens]
            mm_inputs.update({"audio_features": audio_features, "audio_feature_lens": audio_feature_lens})
            if kwargs.get("ret_phs", False):
                mm_inputs.update({"audio_phs": audio_phs})

        return mm_inputs

    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        num_image_tokens, num_video_tokens, num_audio_tokens = 0, 0, 0
        messages = deepcopy(messages)
        image_processor: BaseImageProcessor = getattr(processor, "image_processor")
        mm_inputs, audio_inputs = {}, {}
        if len(images) != 0 and len(videos) != 0:
            raise ValueError("MiniCPM-V model does not support input images and videos at the same time.")

        if len(videos) != 0:
            max_slice_nums = 2
            use_image_id = False
            mm_inputs = self._get_mm_inputs([], videos, [], processor)
        else:
            max_slice_nums = image_processor.max_slice_nums
            use_image_id = image_processor.use_image_id

        for i, message in enumerate(messages):
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                content = content.replace(IMAGE_PLACEHOLDER, "{{image}}", 1)
                num_image_tokens += 1

            while VIDEO_PLACEHOLDER in content:
                video_seqlen = len(mm_inputs["pixel_values"][num_video_tokens]) if self.expand_mm_tokens else 1
                content = content.replace(VIDEO_PLACEHOLDER, "{{image}}" * video_seqlen, 1)
                num_video_tokens += 1

            while AUDIO_PLACEHOLDER in content:
                content = content.replace(AUDIO_PLACEHOLDER, "{{audio}}", 1)
                num_audio_tokens += 1

            message["content"] = content.replace("{{image}}", "(<image>./</image>)").replace(
                "{{audio}}", "(<audio>./</audio>)"
            )

        if len(images):
            mm_inputs = self._get_mm_inputs(images, [], [], processor)

        if len(audios):
            audio_inputs = self._get_mm_inputs([], [], audios, processor, ret_phs=True)

        if self.expand_mm_tokens and mm_inputs:
            pattern = "(<image>./</image>)"
            image_sizes = mm_inputs["image_sizes"]
            idx = 0
            for index, message in enumerate(messages):
                text = message["content"]
                image_tags = re.findall(pattern, text)
                text_chunks = text.split(pattern)
                final_text = ""
                for i in range(len(image_tags)):
                    final_text = (
                        final_text
                        + text_chunks[i]
                        + image_processor.get_slice_image_placeholder(
                            image_sizes[0][idx], idx, max_slice_nums, use_image_id
                        )
                    )
                    idx += 1

                final_text += text_chunks[-1]
                messages[index]["content"] = final_text

        if self.expand_mm_tokens and audio_inputs:
            pattern = "(<audio>./</audio>)"
            idx = 0
            for index, message in enumerate(messages):
                text = message["content"]
                audio_tags = re.findall(pattern, text)
                text_chunks = text.split(pattern)
                final_text = ""
                for i in range(len(audio_tags)):
                    audio_placeholder = audio_inputs["audio_phs"][0][idx]
                    final_text = final_text + text_chunks[i] + audio_placeholder
                    idx += 1

                final_text += text_chunks[-1]
                messages[index]["content"] = final_text

        return messages

    @override
    def get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        imglens: list[int],
        vidlens: list[int],
        audlens: list[int],
        batch_ids: list[list[int]],
        processor: Optional["MMProcessor"],
    ) -> dict[str, Union[list[int], "torch.Tensor"]]:
        self._validate_input(processor, images, videos, audios)
        # image bound
        image_bounds_list = []
        valid_image_nums_ls = []
        for i, input_ids in enumerate(batch_ids):
            input_ids_ = torch.tensor(input_ids)
            start_cond = (input_ids_ == processor.tokenizer.im_start_id) | (
                input_ids_ == processor.tokenizer.slice_start_id
            )
            end_cond = (input_ids_ == processor.tokenizer.im_end_id) | (input_ids_ == processor.tokenizer.slice_end_id)
            image_start_tokens = torch.where(start_cond)[0]
            image_start_tokens += 1
            image_end_tokens = torch.where(end_cond)[0]
            valid_image_nums_ls.append(imglens[i])
            image_bounds = torch.hstack(
                [
                    image_start_tokens.unsqueeze(-1),
                    image_end_tokens.unsqueeze(-1),
                ]
            )
            image_bounds_list.append(image_bounds)

        mm_inputs = self._get_mm_inputs(images, videos, [], processor, valid_image_nums_ls=valid_image_nums_ls)
        if "tgt_sizes" not in mm_inputs:
            dummy_data = [torch.empty(0) for _ in range(len(batch_ids))]
            mm_inputs.update({"tgt_sizes": dummy_data, "pixel_values": dummy_data, "image_sizes": dummy_data})

        mm_inputs.update({"image_bound": image_bounds_list})

        if len(audios) > 0:
            # audio bound
            audio_bounds_ls = []
            spk_bounds_ls = []
            valid_audio_nums_ls = []

            for input_ids, audiolen in zip(batch_ids, audlens):
                input_ids_ = torch.tensor(input_ids)
                audio_start_idx = torch.where(input_ids_ == processor.tokenizer.audio_start_id)[0]
                audio_end_idx = torch.where(input_ids_ == processor.tokenizer.audio_end_id)[0]
                assert len(audio_start_idx) == len(audio_end_idx)
                audio_bounds = torch.hstack([(audio_start_idx + 1).unsqueeze(-1), audio_end_idx.unsqueeze(-1)])
                audio_bounds_ls.append(audio_bounds)
                valid_audio_nums_ls.append(audiolen)

                spk_start_idx = torch.where(input_ids_ == processor.tokenizer.spk_start_id)[0]
                spk_end_idx = torch.where(input_ids_ == processor.tokenizer.spk_end_id)[0]
                assert len(spk_start_idx) == len(spk_end_idx)
                spk_bounds = torch.hstack([(spk_start_idx + 1).unsqueeze(-1), spk_end_idx.unsqueeze(-1)])
                spk_bounds_ls.append(spk_bounds)

            audio_inputs = self._get_mm_inputs([], [], audios, processor, valid_audio_nums_ls=valid_audio_nums_ls)
            mm_inputs.update(audio_inputs)
            mm_inputs.update({"audio_bounds": audio_bounds_ls, "spk_bounds": spk_bounds_ls})

        return mm_inputs


@dataclass
class MllamaPlugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        num_image_tokens = 0
        messages = deepcopy(messages)
        for message in messages:
            content = message["content"]
            num_image_tokens += content.count(IMAGE_PLACEHOLDER)
            message["content"] = content.replace(IMAGE_PLACEHOLDER, self.image_token)

        return messages

    @override
    def get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        imglens: list[int],
        vidlens: list[int],
        audlens: list[int],
        batch_ids: list[list[int]],
        processor: Optional["MMProcessor"],
    ) -> dict[str, Union[list[int], "torch.Tensor"]]:
        self._validate_input(processor, images, videos, audios)
        mm_inputs = self._get_mm_inputs(images, videos, audios, processor, imglens)
        if mm_inputs:
            num_tiles = mm_inputs.pop("num_tiles")
            image_token_id: int = getattr(processor, "image_token_id")
            max_image_tiles: int = getattr(processor.image_processor, "max_image_tiles")
            cross_attention_token_mask = [
                get_cross_attention_token_mask(input_ids, image_token_id) for input_ids in batch_ids
            ]
            mm_inputs["cross_attention_mask"] = torch.from_numpy(
                convert_sparse_cross_attention_mask_to_dense(
                    cross_attention_token_mask,
                    num_tiles=num_tiles,
                    max_num_tiles=max_image_tiles,
                    length=max(len(input_ids) for input_ids in batch_ids),
                )
            )  # shape: (batch_size, length, max_num_images, max_num_tiles)

        return mm_inputs


@dataclass
class PaliGemmaPlugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        num_image_tokens = 0
        messages = deepcopy(messages)
        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                content = content.replace(IMAGE_PLACEHOLDER, "", 1)
                num_image_tokens += 1

            message["content"] = content

        return messages

    @override
    def process_token_ids(
        self,
        input_ids: list[int],
        labels: Optional[list[int]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        tokenizer: "PreTrainedTokenizer",
        processor: Optional["MMProcessor"],
    ) -> tuple[list[int], Optional[list[int]]]:
        self._validate_input(processor, images, videos, audios)
        num_images = len(images)
        image_seqlen = processor.image_seq_length if self.expand_mm_tokens else 0  # skip mm token
        image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        input_ids = [image_token_id] * num_images * image_seqlen + input_ids
        if labels is not None:
            labels = [IGNORE_INDEX] * num_images * image_seqlen + labels

        return input_ids, labels

    @override
    def get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        imglens: list[int],
        vidlens: list[int],
        audlens: list[int],
        batch_ids: list[list[int]],
        processor: Optional["MMProcessor"],
    ) -> dict[str, Union[list[int], "torch.Tensor"]]:
        self._validate_input(processor, images, videos, audios)
        seqlens = [len(input_ids) for input_ids in batch_ids]
        mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
        mm_inputs["token_type_ids"] = _get_paligemma_token_type_ids(imglens, seqlens, processor)
        return mm_inputs


@dataclass
class PixtralPlugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        messages = deepcopy(messages)
        if self.expand_mm_tokens:
            mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
            if "pixel_values" in mm_inputs:
                # BC for transformers < 4.49.0
                if isinstance(mm_inputs["image_sizes"], list):
                    image_sizes = iter(mm_inputs["image_sizes"][0])
                else:
                    image_sizes = iter(mm_inputs["image_sizes"].tolist())

                image_break_token: str = getattr(processor, "image_break_token")
                image_end_token: str = getattr(processor, "image_end_token")

        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                if self.expand_mm_tokens:
                    height, width = next(image_sizes)
                    num_height_tokens = height // processor.patch_size
                    num_width_tokens = width // processor.patch_size
                    replace_tokens = [[self.image_token] * num_width_tokens + [image_break_token]] * num_height_tokens
                    replace_tokens = [item for sublist in replace_tokens for item in sublist]
                    replace_tokens[-1] = image_end_token
                    replace_str = "".join(replace_tokens)
                else:
                    replace_str = self.image_token

                content = content.replace(IMAGE_PLACEHOLDER, replace_str, 1)

            message["content"] = content

        return messages

    @override
    def get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        imglens: list[int],
        vidlens: list[int],
        audlens: list[int],
        batch_ids: list[list[int]],
        processor: Optional["MMProcessor"],
    ) -> dict[str, Union[list[int], "torch.Tensor"]]:
        self._validate_input(processor, images, videos, audios)
        mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
        # ref to this commit https://github.com/huggingface/transformers/pull/35122
        # after transformers 4.49.0, the `image_sizes` is mandatory as an input parameter for Pixtral VisionEncoder forwarding.
        # it can be passed into `LlavaConditionalGeneration` as a parameter.
        if not is_transformers_version_greater_than("4.49.0"):
            mm_inputs.pop("image_sizes", None)
        return mm_inputs


@dataclass
class Qwen2AudioPlugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        bos_token: str = getattr(processor, "audio_bos_token")
        eos_token: str = getattr(processor, "audio_eos_token")
        messages = deepcopy(messages)
        if self.expand_mm_tokens:
            mm_inputs = self._get_mm_inputs([], [], audios, processor)
            if "feature_attention_mask" in mm_inputs:
                audio_lengths = mm_inputs["feature_attention_mask"].sum(-1).tolist()

        for message in messages:
            content = message["content"]
            while AUDIO_PLACEHOLDER in content:
                if self.expand_mm_tokens:
                    audio_length = audio_lengths.pop(0)
                    input_length = (audio_length - 1) // 2 + 1
                    audio_seqlen = (input_length - 2) // 2 + 1
                else:
                    audio_seqlen = 1

                content = content.replace(
                    AUDIO_PLACEHOLDER, f"{bos_token}{self.audio_token * audio_seqlen}{eos_token}", 1
                )

            message["content"] = content

        return messages

    @override
    def get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        imglens: list[int],
        vidlens: list[int],
        audlens: list[int],
        batch_ids: list[list[int]],
        processor: Optional["MMProcessor"],
    ) -> dict[str, Union[list[int], "torch.Tensor"]]:
        self._validate_input(processor, images, videos, audios)
        return self._get_mm_inputs(images, videos, audios, processor)
    
@dataclass
class Qwen2VLPlugin(BasePlugin):
    # User configurable: auto/old/new
    dataset_mode: str = "auto"

    # ========== Mode detection ==========
    def _detect_mode_from_extra(self, extra: Optional[Dict[str, Any]]) -> str:
        """
        Simple heuristics:
        - If contains old-version fields: input_point/input_bbox/input_mask/point_image_path/bbox_image_path/mask_image_path etc => old
        - If point/box format like [[x,y]] / [x1,y1,x2,y2] and values look like pixels => old
        - If point/box looks normalized (<=1 or around ~1000 scale) => new
        """
        if not isinstance(extra, dict):
            return "old"

        old_keys = [
            "input_point", "input_bbox", "input_mask",
            "point_image_path", "bbox_image_path", "mask_image_path"
        ]
        if any(k in extra and extra.get(k) for k in old_keys):
            return "old"

        def _looks_like_new_point(pt):
            try:
                p = pt
                if isinstance(p, list) and len(p) == 1 and isinstance(p[0], (list, tuple, np.ndarray)):
                    p = p[0]
                arr = np.array(p, dtype=float).reshape(-1)
                if arr.size == 2:
                    if (arr <= 1.0).all():
                        return True
                    if (arr <= 1200).all() and (arr >= 0.0).all():
                        if (arr > 1.0).any():
                            return True
                return False
            except Exception:
                return False

        def _looks_like_new_bbox(bb):
            try:
                b = bb
                if isinstance(b, list) and len(b) == 1:
                    b = b[0]
                arr = np.array(b, dtype=float).reshape(-1)
                if arr.size == 4:
                    if (arr <= 1.0).all():
                        return True
                    if (arr <= 2000).all() and (arr >= 0.0).all() and (arr > 1.0).any():
                        return True
                return False
            except Exception:
                return False

        for color in ["red", "green", "blue", "yellow"]:
            pt = extra.get(f"{color}_point")
            if pt is not None and _looks_like_new_point(pt):
                return "new"
            bb = extra.get(f"{color}_bbox")
            if bb is not None and _looks_like_new_bbox(bb):
                return "new"

        return "old"

    # ========== Common extreme aspect ratio protection ==========
    @override
    def _preprocess_image(self, image: "ImageObject", **kwargs) -> "ImageObject":
        image = super()._preprocess_image(image, **kwargs)
        if min(image.width, image.height) < 28:
            width, height = max(image.width, 28), max(image.height, 28)
            image = image.resize((width, height))
        if image.width / image.height > 200:
            width, height = int(image.height * 180), image.height
            image = image.resize((width, height))
        if image.height / image.width > 200:
            width, height = image.width, int(image.width * 180)
            image = image.resize((width, height))
        return image

    # ========== Lightweight JSON parsing ==========
    def _json_maybe_parse(self, value):
        if value is None:
            return None
        if isinstance(value, (list, dict, bool, int, float)):
            return value
        if isinstance(value, str):
            s = value.strip()
            if s == "":
                return None
            try:
                import json as _json
                return _json.loads(s)
            except Exception:
                pass
            try:
                import json as _json
                s2 = s.replace("True", "true").replace("False", "false")
                return _json.loads(s2)
            except Exception:
                return value
        return value

    # ========== Visualization/annotation: old logic ==========
    def _overlay_mask_rgba(self, pil_img: Image.Image, mask: np.ndarray, color=(255, 0, 0), alpha=0.35) -> Image.Image:
        if mask is None:
            return pil_img
        if mask.dtype != bool:
            mask = mask.astype(bool)
        if not mask.any():
            return pil_img

        base = pil_img.convert("RGBA")
        color_img = Image.new("RGBA", base.size, color + (int(255 * alpha),))

        mh, mw = mask.shape[:2]
        if (mw, mh) != (base.width, base.height):
            mask_pil = Image.fromarray((mask.astype(np.uint8) * 255)).resize((base.width, base.height), resample=Image.NEAREST)
        else:
            mask_pil = Image.fromarray(mask.astype(np.uint8) * 255)

        out = Image.composite(color_img, base, mask_pil)
        return out.convert("RGB")

    def _load_mask_array(self, mask_path: str) -> Optional[np.ndarray]:
        if not os.path.isfile(mask_path):
            return None
        try:
            return np.load(mask_path, allow_pickle=True)
        except Exception:
            return None

    def _select_mask_from_array(self, mask_arr: np.ndarray, object_ids: Union[List[bool], List[int], List[str], None]) -> Optional[np.ndarray]:
        if mask_arr is None:
            return None

        arr = mask_arr
        while arr.ndim > 2 and arr.shape[0] == 1:
            arr = arr[0]

        if arr.ndim == 2:
            if arr.dtype == bool:
                if object_ids is None:
                    return arr
                if isinstance(object_ids, list) and any(bool(x) for x in object_ids):
                    return arr
                return np.zeros_like(arr, dtype=bool)
            else:
                if object_ids is None:
                    return arr != 0
                ids = []
                for x in (object_ids if isinstance(object_ids, list) else [object_ids]):
                    try:
                        ids.append(int(x))
                    except Exception:
                        pass
                if not ids:
                    return arr != 0
                mask = np.zeros_like(arr, dtype=bool)
                for k in ids:
                    mask |= (arr == k)
                return mask

        if arr.ndim == 3:
            if arr.shape[0] not in (arr.shape[1],) and arr.shape[0] < 16 and arr.shape[0] != arr.shape[1]:
                arr = np.moveaxis(arr, 0, 2)
            if arr.dtype != bool:
                arr = arr != 0
            H, W, K = arr.shape
            if object_ids is None:
                return arr.any(axis=2)
            if isinstance(object_ids, list) and len(object_ids) == K and all(isinstance(x, (bool, np.bool_)) for x in object_ids):
                sel = [i for i, v in enumerate(object_ids) if v]
            else:
                tmp = []
                for x in (object_ids if isinstance(object_ids, list) else [object_ids]):
                    if isinstance(x, (bool, np.bool_)):
                        if x:
                            return arr.any(axis=2)
                    else:
                        try:
                            tmp.append(int(x))
                        except Exception:
                            pass
                sel = tmp
            if not sel:
                return arr.any(axis=2)
            sel = [i for i in sel if 0 <= i < K]
            if not sel:
                return arr.any(axis=2)
            return arr[:, :, sel].any(axis=2)

        return None

    # ========== Format/coordinate helpers (common) ==========
    def _safe_to_pil(self, img) -> Image.Image:
        if isinstance(img, Image.Image):
            return img
        if isinstance(img, np.ndarray):
            if img.ndim == 2:
                return Image.fromarray(img, mode="L")
            if img.ndim == 3:
                if img.shape[2] == 3:
                    if img.flags["C_CONTIGUOUS"]:
                        return Image.fromarray(img[:, :, ::-1])
                    else:
                        return Image.fromarray(np.ascontiguousarray(img)[:, :, ::-1])
                if img.shape[2] == 4:
                    rgba = img[:, :, [2, 1, 0, 3]]
                    return Image.fromarray(rgba, mode="RGBA")
        try:
            return Image.fromarray(np.array(img))
        except Exception:
            raise TypeError(f"Unsupported image type for conversion to PIL: {type(img)}")

    def _pil_to_numpy_bgr(self, img: Image.Image) -> np.ndarray:
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        arr = np.array(img)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]
        return arr[:, :, ::-1].copy()

    # ========== Coordinate scaling: old logic ==========
    def _scale_point_old(self, point, width, height):
        if isinstance(point, (list, tuple)) and len(point) == 1 and isinstance(point[0], (list, tuple, np.ndarray)):
            point = point[0]
        if not (isinstance(point, (list, tuple, np.ndarray)) and len(point) == 2):
            raise ValueError(f"Unsupported point format: {point}")

        y, x = float(point[0]), float(point[1])
        if 0 <= x < width and 0 <= y < height:
            return np.array([x, y], dtype=float)
        sx, sy = y, x
        if 0 <= sx < width and 0 <= sy < height:
            return np.array([sx, sy], dtype=float)
        return np.array([x, y], dtype=float)

    def _scale_bbox_old(self, bbox, width, height):
        if isinstance(bbox, (list, tuple)) and len(bbox) == 1:
            bbox = bbox[0]

        if isinstance(bbox, dict):
            keys = set(bbox.keys())
            if {"x", "y", "w", "h"} <= keys:
                x1 = float(bbox["x"]); y1 = float(bbox["y"])
                x2 = x1 + float(bbox["w"]); y2 = y1 + float(bbox["h"])
                return np.array([x1, y1, x2, y2], dtype=float)
            elif {"x1", "y1", "x2", "y2"} <= keys:
                x1 = float(bbox["x1"]); y1 = float(bbox["y1"]); x2 = float(bbox["x2"]); y2 = float(bbox["y2"])
                return np.array([x1, y1, x2, y2], dtype=float)
            else:
                raise ValueError(f"Unsupported bbox dict keys: {keys}")

        arr = np.array(bbox, dtype=float).reshape(-1)
        if arr.shape[0] != 4:
            raise ValueError(f"bbox must have 4 elements, got {bbox}")
        x1, y1, x2, y2 = arr.tolist()
        if x2 < x1 or y2 < y1:
            x2 = x1 + x2
            y2 = y1 + y2
        return np.array([x1, y1, x2, y2], dtype=float)

    # ========== Coordinate scaling: new logic (ratio or 1000-scale) ==========
    def _scale_point_new(self, point, width, height):
        if isinstance(point, (list, tuple)) and len(point) == 1 and isinstance(point[0], (list, tuple, np.ndarray)):
            point = point[0]
        p = np.array(point, dtype=float).reshape(-1)
        if p.size != 2:
            raise ValueError(f"Invalid point: {point}")

        if (p > 1.0).any():
            p = (p / 1000.0) * np.array([width, height], dtype=float)
        else:
            p = p * np.array([width, height], dtype=float)
        return p

    def _scale_bbox_new(self, bbox, width, height):
        b = bbox
        if isinstance(b, list) and len(b) == 1 and isinstance(b[0], (list, tuple, np.ndarray)):
            b = b[0]
        b = np.array(b, dtype=float).reshape(-1)
        if b.size != 4:
            raise ValueError(f"bbox must have 4 elements, got {bbox}")
        if (b > 1.0).any():
            b = (b / 1000.0) * np.array([width, height, width, height], dtype=float)
        else:
            b = b * np.array([width, height, width, height], dtype=float)
        x1, y1, x2, y2 = b.tolist()
        if x2 < x1 or y2 < y1:
            x2 = x1 + x2
            y2 = y1 + y2
        return np.array([x1, y1, x2, y2], dtype=float)

    def _index_of_image_in_frames(self, image_path: str, frames: List[str]) -> Optional[int]:
        if not image_path:
            return None
        apath = os.path.abspath(image_path)

        for i, f in enumerate(frames):
            if os.path.abspath(f) == apath:
                return i

        cand = []
        if "/images_8/" in apath:
            cand.append(apath.replace("/images_8/", "/images/"))
        if "/images/" in apath:
            cand.append(apath.replace("/images/", "/images_8/"))

        for c in cand:
            ac = os.path.abspath(c)
            for i, f in enumerate(frames):
                if os.path.abspath(f) == ac:
                    return i

        base = os.path.basename(apath)
        matches = [i for i, f in enumerate(frames) if os.path.basename(f) == base]
        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            for i in matches:
                af = os.path.abspath(frames[i])
                if "/images_8/" in af or "/images/" in af:
                    return i
            return matches[0]
        return None

    def _build_scene_extra_for_video(self, info_in: Dict[str, Any], frame_paths: List[str]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        ein = info_in or {}

        # POINT
        point_img_path = ein.get("point_image_path")
        point_points = self._json_maybe_parse(ein.get("point_points"))
        ip_old = ein.get("input_point")
        if (point_img_path and point_points) or (isinstance(ip_old, dict) and ip_old.get("image_path") and ip_old.get("points")):
            if not (point_img_path and point_points):
                point_img_path = ip_old.get("image_path")
                point_points = ip_old.get("points")
            if isinstance(point_points, list) and len(point_points) > 0:
                out["red_point"] = point_points[0]
                idx = self._index_of_image_in_frames(point_img_path, frame_paths)
                if idx is None:
                    idx = 0
                out["point_img_idx"] = [idx, None, None]

        # BBOX
        bbox_img_path = ein.get("bbox_image_path")
        bbox_bboxes = self._json_maybe_parse(ein.get("bbox_bboxes"))
        ib_old = ein.get("input_bbox")
        if (bbox_img_path and bbox_bboxes) or (isinstance(ib_old, dict) and ib_old.get("image_path") and ib_old.get("bboxes")):
            if not (bbox_img_path and bbox_bboxes):
                bbox_img_path = ib_old.get("image_path")
                bbox_bboxes = ib_old.get("bboxes")
            if isinstance(bbox_bboxes, list) and len(bbox_bboxes) > 0:
                first = bbox_bboxes[0]
                out["red_bbox"] = [first]
                idx = self._index_of_image_in_frames(bbox_img_path, frame_paths)
                if idx is None:
                    idx = 0
                out["bbox_img_idx"] = [idx, None, None]

        # MASK
        mask_img_path = ein.get("mask_image_path")
        mask_mask_path = ein.get("mask_mask_path")
        mask_object_ids = self._json_maybe_parse(ein.get("mask_object_ids"))
        im_old = ein.get("input_mask")
        selected_mask = None
        if (mask_img_path and mask_mask_path) or (isinstance(im_old, dict) and im_old.get("image_path") and im_old.get("mask_path")):
            if not (mask_img_path and mask_mask_path):
                mask_img_path = im_old.get("image_path")
                mask_mask_path = im_old.get("mask_path")
                mask_object_ids = im_old.get("object_ids", None)

            mask_arr = self._load_mask_array(mask_mask_path) if mask_mask_path else None
            if mask_arr is not None:
                selected_mask = self._select_mask_from_array(mask_arr, mask_object_ids)
            if selected_mask is not None:
                out["red_mask_info"] = {"mask_array": selected_mask}
                idx = self._index_of_image_in_frames(mask_img_path, frame_paths)
                if idx is None:
                    idx = 0
                out["mask_img_idx"] = [idx]

        return out

    def _normalize_extra_info(self, extra_info: Optional[Union[Dict[str, Any], List[Optional[Dict[str, Any]]]]], batch_size: int):
        if extra_info is None:
            return [None] * batch_size
        if isinstance(extra_info, list) and len(extra_info) == 0:
            return [None] * batch_size
        if isinstance(extra_info, dict):
            if batch_size != 1:
                raise ValueError(f"extra_info dict can only be broadcast when batch_size == 1 (got {batch_size}).")
            return [extra_info] * batch_size
        if isinstance(extra_info, list):
            if len(extra_info) != batch_size:
                raise ValueError(f"extra_info list length {len(extra_info)} != batch size {batch_size}")
            return extra_info
        raise TypeError(f"extra_info must be dict or list[dict], got {type(extra_info)}")

    def _draw_annotations_on_pil_old(self, img: Image.Image, extra_info: Dict[str, Any], current_img_idx: int) -> Image.Image:
        draw = ImageDraw.Draw(img)
        W, H = img.width, img.height
        colors_rgb = {"red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 128, 255)}
        order = ["red", "green", "blue"]

        def _normalize_indices(val):
            if val is None:
                return None
            if isinstance(val, list) and len(val) == 1 and isinstance(val[0], list):
                return val[0]
            return val

        point_img_idx = _normalize_indices(extra_info.get("point_img_idx"))
        bbox_img_idx = _normalize_indices(extra_info.get("bbox_img_idx"))
        mask_img_idx = _normalize_indices(extra_info.get("mask_img_idx"))

        # Points
        for i, color in enumerate(order):
            pt = extra_info.get(f"{color}_point")
            if pt is None:
                continue
            if point_img_idx is not None:
                if i >= len(point_img_idx) or point_img_idx[i] is None or point_img_idx[i] != current_img_idx:
                    continue
            xy = self._scale_point_old(pt, W, H)
            x, y = float(xy[0]), float(xy[1])
            if not (0 <= x < W and 0 <= y < H):
                continue
            r = max(8, int(0.014 * min(W, H)))
            draw.ellipse((x - r, y - r, x + r, y + r), fill=colors_rgb[color], outline=colors_rgb[color], width=2)

        # BBoxes
        for i, color in enumerate(order):
            bb = extra_info.get(f"{color}_bbox")
            if bb is None:
                continue
            if bbox_img_idx is not None:
                if i >= len(bbox_img_idx) or bbox_img_idx[i] is None or bbox_img_idx[i] != current_img_idx:
                    continue
            x1, y1, x2, y2 = self._scale_bbox_old(bb, W, H)
            if x1 >= W or y1 >= H or x2 <= 0 or y2 <= 0:
                continue
            draw.rectangle((x1, y1, x2, y2), outline=colors_rgb[color], width=max(2, int(0.004 * min(W, H))))

        # Mask
        red_mask_info = extra_info.get("red_mask_info", None)
        if red_mask_info is not None and mask_img_idx is not None:
            allowed = False
            if isinstance(mask_img_idx, list):
                allowed = (current_img_idx in [m for m in mask_img_idx if m is not None])
            else:
                allowed = (current_img_idx == mask_img_idx)
            if allowed:
                mask_arr = red_mask_info.get("mask_array")
                if isinstance(mask_arr, np.ndarray):
                    img_with = self._overlay_mask_rgba(img, mask_arr, color=(255, 0, 0), alpha=0.35)
                    img.paste(img_with)

        return img

    def _draw_annotations_on_pil_new(self, img: Image.Image, extra_info: Dict[str, Any], current_img_idx: int) -> Image.Image:
        draw = ImageDraw.Draw(img)
        W, H = img.width, img.height
        colors_rgb = {"red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 128, 255)}
        order = ["red", "green", "blue"]

        def _normalize_indices(val):
            if val is None:
                return None
            if isinstance(val, list) and len(val) == 1 and isinstance(val[0], list):
                return val[0]
            return val

        point_img_idx = _normalize_indices(extra_info.get("point_img_idx"))
        bbox_img_idx = _normalize_indices(extra_info.get("bbox_img_idx"))

        # Points
        for i, color in enumerate(order):
            pt = extra_info.get(f"{color}_point")
            if pt is None:
                continue
            if point_img_idx is not None:
                if i >= len(point_img_idx) or point_img_idx[i] is None or point_img_idx[i] != current_img_idx:
                    continue
            xy = self._scale_point_new(pt, W, H)
            x, y = float(xy[0]), float(xy[1])
            r = max(4, int(0.01 * min(W, H)))
            draw.ellipse((x - r, y - r, x + r, y + r), fill=colors_rgb[color], outline=colors_rgb[color], width=2)
            try:
                draw.text((x + r + 2, y + r + 2), str(i + 1), fill=colors_rgb[color])
            except Exception:
                pass

        # BBoxes
        for i, color in enumerate(order):
            bb = extra_info.get(f"{color}_bbox")
            if bb is None:
                continue
            if bbox_img_idx is not None:
                if i >= len(bbox_img_idx) or bbox_img_idx[i] is None or bbox_img_idx[i] != current_img_idx:
                    continue
            x1, y1, x2, y2 = self._scale_bbox_new(bb, W, H)
            draw.rectangle((x1, y1, x2, y2), outline=colors_rgb[color], width=max(2, int(0.004 * min(W, H))))

        return img

    def _maybe_annotate_frames_old(
        self,
        pil_frames_at_original_size: List[Image.Image],
        sampled_indices: Optional[List[int]],
        extra_info: Optional[Dict[str, Any]],
        debug: bool = False,
        debug_dir: str = "debug_video_anns",
        video_tag: str = "vid",
    ) -> List[Image.Image]:
        if not extra_info or not pil_frames_at_original_size:
            return pil_frames_at_original_size

        os.makedirs(debug_dir, exist_ok=True)
        annotated_frames: List[Image.Image] = []

        for j, pil in enumerate(pil_frames_at_original_size):
            current_idx = int(sampled_indices[j]) if sampled_indices is not None and j < len(sampled_indices) else j
            annotated = self._draw_annotations_on_pil_old(pil.copy(), extra_info, current_img_idx=current_idx)
            annotated_frames.append(annotated)

            if debug:
                now = datetime.now()
                timestamp = now.strftime("%Y%m%d_%H%M%S_") + f"{now.microsecond // 1000:03d}"
                debug_name = f"{video_tag}_frame{j:04d}_{current_idx:04d}"+f"_{timestamp}.jpg"
                try:
                    annotated.save(os.path.join(debug_dir, debug_name))
                except Exception:
                    pass

        return annotated_frames

    def _maybe_annotate_frames_new(
        self,
        processed_frames: List[Any],
        sampled_indices: List[int],
        extra_info: Optional[Dict[str, Any]],
        debug: bool = False,
        debug_dir: str = "debug_video_anns",
        video_tag: str = "vid",
    ) -> List[Any]:
        if not extra_info:
            return processed_frames

        os.makedirs(debug_dir, exist_ok=True)
        annotated_frames = []
        for j, frame in enumerate(processed_frames):
            pil = self._safe_to_pil(frame)
            current_idx = int(sampled_indices[j]) if sampled_indices is not None and j < len(sampled_indices) else j
            annotated = self._draw_annotations_on_pil_new(pil.copy(), extra_info, current_img_idx=current_idx)

            if isinstance(frame, Image.Image):
                out = annotated
            else:
                out = self._pil_to_numpy_bgr(annotated)
            annotated_frames.append(out)

            if debug:
                debug_name = f"{video_tag}_frame{j:04d}_orig{current_idx:04d}.jpg"
                try:
                    annotated.save(os.path.join(debug_dir, debug_name))
                except Exception:
                    pass

        return annotated_frames

    def process_and_pad_images(self, geometry_encoder_inputs):
        if not geometry_encoder_inputs or not geometry_encoder_inputs[0]:
            raise ValueError("Input list cannot be empty")

        processed_batches_364 = []
        processed_batches_420 = []

        to_tensor = TF.ToTensor()
        extra_type = "dinov3"
        target_sizes = [364, 420] if extra_type != "vggt" else [364, 364]

        def preprocess_for_target(img: Image.Image, target_size: int):
            if not isinstance(img, Image.Image):
                raise TypeError(f"Expected a PIL Image, but got {type(img)}")

            if img.mode == "RGBA":
                background = Image.new("RGBA", img.size, (255, 255, 255, 255))
                img = Image.alpha_composite(background, img)

            img = img.convert("RGB")
            width, height = img.size

            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14

            new_width = max(14, new_width)
            new_height = max(14, new_height)

            img_resized = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
            img_tensor = to_tensor(img_resized)

            h_padding = target_size - img_tensor.shape[1]
            w_padding = target_size - img_tensor.shape[2]

            if h_padding < 0 or w_padding < 0:
                top = max(0, (img_tensor.shape[1] - target_size) // 2)
                left = max(0, (img_tensor.shape[2] - target_size) // 2)
                img_tensor = img_tensor[:, top : top + target_size, left : left + target_size]
                h_padding = 0
                w_padding = 0

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img_tensor = torch.nn.functional.pad(
                    img_tensor, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
            return img_tensor

        for image_list in geometry_encoder_inputs:
            processed_images_364 = []
            processed_images_420 = []
            for img in image_list:
                pil_img = self._safe_to_pil(img)
                img_364 = preprocess_for_target(pil_img, target_sizes[0])
                img_420 = preprocess_for_target(pil_img, target_sizes[1])
                processed_images_364.append(img_364)
                processed_images_420.append(img_420)

            images_tensor_364 = torch.stack(processed_images_364)
            images_tensor_420 = torch.stack(processed_images_420)
            processed_batches_364.append(images_tensor_364)
            processed_batches_420.append(images_tensor_420)

        return processed_batches_364, processed_batches_420

    @override
    def _regularize_videos(
        self,
        videos: list["VideoInput"],
        **kwargs,
    ) -> dict[str, Union[list[list["ImageObject"]], list[float]]]:
        results_per_sample: List[List[Image.Image]] = []
        fps_per_sample: List[float] = []

        video_fps_default = kwargs.get("video_fps", 1.0)
        video_maxlen = kwargs.get("video_maxlen", 128)
        debug = bool(kwargs.get("debug", False))
        debug_dir = kwargs.get("debug_dir", "debug_video_anns")

        extra_info_in = kwargs.get("extra_info", None)
        extra_info_list = self._normalize_extra_info(extra_info_in, batch_size=len(videos))

        for b_idx, video in enumerate(videos):
            sample_extra = extra_info_list[b_idx] if extra_info_list else None
            mode = self.dataset_mode
            if mode == "auto":
                mode = self._detect_mode_from_extra(sample_extra)

            processed_frames: List[Image.Image] = []
            fps_used = video_fps_default
            sampled_indices: Optional[List[int]] = None

            if mode == "old":
                if isinstance(video, list) and all(isinstance(p, str) for p in video):
                    all_paths = video
                    total_frames = len(all_paths)
                    if total_frames == 0:
                        results_per_sample.append([])
                        fps_per_sample.append(fps_used)
                        continue

                    num_samples = min(total_frames, video_maxlen)
                    sample_indices = np.linspace(0, total_frames - 1, num_samples, dtype=np.int32)
                    sampled_indices = np.unique(sample_indices).tolist()
                    sampled_paths = [all_paths[i] for i in sampled_indices]

                    loaded_pils: List[Image.Image] = []
                    for p in sampled_paths:
                        try:
                            im = Image.open(p).convert("RGB")
                        except Exception:
                            im = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))
                        loaded_pils.append(im)

                    built = self._build_scene_extra_for_video(sample_extra, all_paths) if isinstance(sample_extra, dict) else {}
                    annotated_pils = self._maybe_annotate_frames_old(
                        pil_frames_at_original_size=loaded_pils,
                        sampled_indices=sampled_indices,
                        extra_info=built,
                        debug=debug,
                        debug_dir=debug_dir,
                        video_tag=f"vid{b_idx:03d}",
                    )
                    processed_frames = self._regularize_images(annotated_pils, **kwargs)["images"]

                elif isinstance(video, (str, BytesIO)) or hasattr(video, "read"):
                    container = av.open(video, "r")
                    video_stream = next(stream for stream in container.streams if stream.type == "video")
                    sample_indices = self._get_video_sample_indices(video_stream, **kwargs)
                    sampled_indices = (sample_indices.tolist() if hasattr(sample_indices, "tolist") else list(sample_indices))
                    target_set = set(sampled_indices)

                    decoded_frames_pil: List[Image.Image] = []
                    container.seek(0)
                    for frame_idx, frame in enumerate(container.decode(video_stream)):
                        if frame_idx in target_set:
                            decoded_frames_pil.append(frame.to_image())

                    built = {}
                    if isinstance(sample_extra, dict):
                        frames_list = sample_extra.get("frames", None)
                        if isinstance(frames_list, list) and all(isinstance(p, str) for p in frames_list):
                            built = self._build_scene_extra_for_video(sample_extra, frames_list)

                    annotated_pils = self._maybe_annotate_frames_old(
                        pil_frames_at_original_size=decoded_frames_pil,
                        sampled_indices=sampled_indices,
                        extra_info=built,
                        debug=debug,
                        debug_dir=debug_dir,
                        video_tag=f"vid{b_idx:03d}",
                    )

                    processed_frames = self._regularize_images(annotated_pils, **kwargs)["images"]

                    duration_in_seconds = 0.0
                    if video_stream.duration is not None:
                        duration_in_seconds = float(video_stream.duration * video_stream.time_base)
                    fps_used = (len(sampled_indices) / duration_in_seconds) if duration_in_seconds > 0 else video_fps_default

                else:
                    raise TypeError(f"Unsupported video input type at batch index {b_idx}: {type(video)}")

            else:
                processed_frames_list: list = []
                sampled_indices = None
                if isinstance(video, list):
                    video_frames = video
                    total_frames = len(video_frames)
                    if total_frames == 0:
                        results_per_sample.append([])
                        fps_per_sample.append(video_fps_default)
                        continue
                    num_samples = min(total_frames, video_maxlen)
                    sample_indices = np.linspace(0, total_frames - 1, num_samples, dtype=np.int32)
                    unique_indices = np.unique(sample_indices)
                    sampled_frame_paths = [video_frames[i] for i in unique_indices]
                    sampled_indices = list(map(int, unique_indices.tolist()))
                    processed_frames_list = self._regularize_images(sampled_frame_paths, **kwargs)["images"]
                    fps_used = video_fps_default

                elif isinstance(video, (str, BytesIO, BufferedReader)) or hasattr(video, "read"):
                    container = av.open(video, "r")
                    video_stream = next(stream for stream in container.streams if stream.type == "video")
                    sample_indices = self._get_video_sample_indices(video_stream, **kwargs)
                    sampled_indices = list(map(int, sample_indices.tolist())) if hasattr(sample_indices, "tolist") else list(sample_indices)

                    decoded_frames: list = []
                    container.seek(0)
                    for frame_idx, frame in enumerate(container.decode(video_stream)):
                        if frame_idx in sample_indices:
                            decoded_frames.append(frame.to_image())

                    processed_frames_list = self._regularize_images(decoded_frames, **kwargs)["images"]
                    duration_in_seconds = 0.0
                    if video_stream.duration is not None:
                        duration_in_seconds = float(video_stream.duration * video_stream.time_base)
                    fps_used = (len(sample_indices) / duration_in_seconds) if duration_in_seconds > 0 else video_fps_default
                else:
                    raise TypeError(f"Unsupported video input type at batch index {b_idx}: {type(video)}")

                if processed_frames_list:
                    processed_frames_list = self._maybe_annotate_frames_new(
                        processed_frames=processed_frames_list,
                        sampled_indices=sampled_indices,
                        extra_info=sample_extra,
                        debug=debug,
                        debug_dir=debug_dir,
                        video_tag=f"vid{b_idx:03d}",
                    )

                processed_frames = processed_frames_list

            if len(processed_frames) % 2 != 0 and len(processed_frames) > 0:
                processed_frames.append(processed_frames[-1])

            results_per_sample.append(processed_frames)
            fps_per_sample.append(fps_used)

        if getattr(self, "use_geometry_encoder", True):
            results_364, results_420 = self.process_and_pad_images(results_per_sample)
            return {"videos": results_364, "geometry_encoder_inputs": results_420, "fps_per_video": fps_per_sample}

        return {"videos": results_per_sample, "fps_per_video": fps_per_sample}

    # ========== Parse extra_info (string/JSON5/ast) ==========
    def parse_extra(self, i):
        if isinstance(i, bytes):
            i = i.decode('utf-8', errors='strict')
        i = i.lstrip('\ufeff').strip()
        try:
            return json.loads(i)
        except json.JSONDecodeError:
            pass
        try:
            import json5
            return json5.loads(i)
        except Exception:
            pass
        try:
            import ast
            return ast.literal_eval(i)
        except Exception as e:
            raise ValueError(f"Failed to parse extra_info: {e}\nRaw content: {repr(i)}")

    # ========== Build multimodal inputs ==========
    @override
    def _get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        extra_info: Optional[List[Optional[Dict[str, Any]]]],
        processor: "MMProcessor",
    ) -> dict[str, "torch.Tensor"]:
        image_processor: BaseImageProcessor = getattr(processor, "image_processor", None)
        mm_inputs: Dict[str, Any] = {}

        if isinstance(extra_info, str):
            extra_info = [self.parse_extra(extra_info)]
        elif isinstance(extra_info, list):
            parsed = []
            for e in extra_info:
                parsed.append(self.parse_extra(e) if isinstance(e, (str, bytes)) else e)
            extra_info = parsed

        if len(images) != 0:
            images = self._regularize_images(
                images,
                image_max_pixels=getattr(processor, "image_max_pixels", 768 * 768),
                image_min_pixels=getattr(processor, "image_min_pixels", 32 * 32),
            )["images"]
            mm_inputs.update(image_processor(images, return_tensors="pt"))

        if len(videos) != 0:
            self.use_geometry_encoder = True

            video_dict = self._regularize_videos(
                videos,
                image_max_pixels=getattr(processor, "video_max_pixels", 256 * 256),
                image_min_pixels=getattr(processor, "video_min_pixels", 16 * 16),
                video_fps=getattr(processor, "video_fps", 1.0),
                video_maxlen=getattr(processor, "video_maxlen", 128),
                extra_info=extra_info,
            )
            if self.use_geometry_encoder and "geometry_encoder_inputs" in video_dict:
                mm_inputs["geometry_encoder_inputs"] = video_dict["geometry_encoder_inputs"]

            mm_inputs.update(image_processor(images=None, videos=video_dict["videos"], return_tensors="pt"))
            temporal_patch_size: int = getattr(image_processor, "temporal_patch_size", 2)
            if "second_per_grid_ts" in getattr(processor, "model_input_names", []):
                mm_inputs["second_per_grid_ts"] = [
                    temporal_patch_size / fps if fps > 0 else 0.0 for fps in video_dict["fps_per_video"]
                ]

        return mm_inputs

    # ========== Expand multimodal tokens in text messages ==========
    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        extra_info: list[Any],
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        num_image_tokens, num_video_tokens = 0, 0
        messages = deepcopy(messages)
        image_processor: BaseImageProcessor = getattr(processor, "image_processor", None)

        merge_length = getattr(image_processor, "merge_size", 2) ** 2
        if self.expand_mm_tokens:
            mm_inputs = self._get_mm_inputs(images, videos, audios, extra_info, processor)
            image_grid_thw = mm_inputs.get("image_grid_thw", [])
            video_grid_thw = mm_inputs.get("video_grid_thw", [])
        else:
            image_grid_thw = [None] * len(images)
            video_grid_thw = [None] * len(videos)

        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                image_seqlen = image_grid_thw[num_image_tokens].prod() // merge_length if self.expand_mm_tokens else 1
                content = content.replace(
                    IMAGE_PLACEHOLDER, f"<|vision_start|>{self.image_token * image_seqlen}<|vision_end|>", 1
                )
                num_image_tokens += 1

            while VIDEO_PLACEHOLDER in content:
                video_seqlen = video_grid_thw[num_video_tokens].prod() // merge_length if self.expand_mm_tokens else 1
                content = content.replace(
                    VIDEO_PLACEHOLDER, f"<|vision_start|>{self.video_token * video_seqlen}<|vision_end|>", 1
                )
                num_video_tokens += 1

            message["content"] = content

        return messages

@dataclass
class Qwen2OmniPlugin(Qwen2VLPlugin):
    @override
    def _get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: "MMProcessor",
    ) -> dict[str, "torch.Tensor"]: 
        image_processor: BaseImageProcessor = getattr(processor, "image_processor", None)
        feature_extractor: SequenceFeatureExtractor = getattr(processor, "feature_extractor", None)
        mm_inputs = {}
        if len(images) != 0:
            images = self._regularize_images(
                images,
                image_max_pixels=getattr(processor, "image_max_pixels", 768 * 768),
                image_min_pixels=getattr(processor, "image_min_pixels", 32 * 32),
            )["images"]
            mm_inputs.update(image_processor(images, return_tensors="pt"))

        if len(videos) != 0:
            video_dict = self._regularize_videos(
                videos,
                image_max_pixels=getattr(processor, "video_max_pixels", 256 * 256),
                image_min_pixels=getattr(processor, "video_min_pixels", 16 * 16),
                video_fps=getattr(processor, "video_fps", 1.0),
                video_maxlen=getattr(processor, "video_maxlen", 128),
            )
            mm_inputs.update(image_processor(images=None, videos=video_dict["videos"], return_tensors="pt"))
            temporal_patch_size: int = getattr(image_processor, "temporal_patch_size", 2)
            mm_inputs["video_second_per_grid"] = torch.tensor(
                [temporal_patch_size / fps for fps in video_dict["fps_per_video"]]
            )

        if len(audios) != 0:
            audios = self._regularize_audios(
                audios,
                sampling_rate=getattr(processor, "audio_sampling_rate", 16000),
            )["audios"]
            mm_inputs.update(
                feature_extractor(
                    audios,
                    sampling_rate=getattr(processor, "audio_sampling_rate", 16000),
                    return_attention_mask=True,
                    padding="max_length",
                    return_tensors="pt",
                )
            )
            mm_inputs["feature_attention_mask"] = mm_inputs.pop("attention_mask")  # prevent conflicts

        return mm_inputs

    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        num_image_tokens, num_video_tokens, num_audio_tokens = 0, 0, 0
        messages = deepcopy(messages)
        image_processor: BaseImageProcessor = getattr(processor, "image_processor", None)

        merge_length = processor.image_processor.merge_size**2
        use_audio_in_video = getattr(processor, "use_audio_in_video", False)
        if self.expand_mm_tokens:
            mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
            image_grid_thw = mm_inputs.get("image_grid_thw", [])
            video_grid_thw = mm_inputs.get("video_grid_thw", [])
            if "feature_attention_mask" in mm_inputs:
                input_lengths = (mm_inputs["feature_attention_mask"].sum(-1).numpy() - 1) // 2 + 1
                audio_lengths = (input_lengths - 2) // 2 + 1
        else:
            mm_inputs = {}
            image_grid_thw = [None] * len(images)
            video_grid_thw = [None] * len(videos)
            audio_lengths = [None] * len(audios)

        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                image_seqlen = image_grid_thw[num_image_tokens].prod() // merge_length if self.expand_mm_tokens else 1
                content = content.replace(
                    IMAGE_PLACEHOLDER, f"<|vision_bos|>{self.image_token * image_seqlen}<|vision_eos|>", 1
                )
                num_image_tokens += 1

            if (
                use_audio_in_video and len(audios) and len(videos)
            ):
                if len(videos) != len(audios):
                    raise ValueError(
                        f"Number of videos ({len(videos)}) must match number of audios ({len(audios)}) when using audio in video."
                    )

                while VIDEO_PLACEHOLDER in content:
                    video_pos = content.find(VIDEO_PLACEHOLDER)
                    audio_pos = content.find(AUDIO_PLACEHOLDER, video_pos)
                    if audio_pos == -1 or audio_pos < video_pos:
                        raise ValueError(
                            f"Each {VIDEO_PLACEHOLDER} must be followed by an {AUDIO_PLACEHOLDER} when using audio in video."
                        )

                    audio_t_index = torch.arange(audio_lengths[num_audio_tokens])
                    video_t_index = (
                        torch.arange(video_grid_thw[num_video_tokens][0])
                        .view(-1, 1, 1)
                        .expand(
                            -1,
                            video_grid_thw[num_video_tokens][1] // image_processor.merge_size,
                            video_grid_thw[num_video_tokens][2] // image_processor.merge_size,
                        )
                        .flatten()
                        * mm_inputs["video_second_per_grid"][num_video_tokens]
                        * 25
                    ).long()
                    t_ntoken_per_chunk = 50
                    video_chunk_indices = processor.get_chunked_index(video_t_index, t_ntoken_per_chunk)
                    audio_chunk_indices = processor.get_chunked_index(audio_t_index, t_ntoken_per_chunk)
                    placeholder_string = ""
                    placeholder_string += "<|vision_bos|>" + "<|audio_bos|>"
                    for j in range(max(len(video_chunk_indices), len(audio_chunk_indices))):
                        video_chunk_index = video_chunk_indices[j] if j < len(video_chunk_indices) else None
                        audio_chunk_index = audio_chunk_indices[j] if j < len(audio_chunk_indices) else None
                        if video_chunk_index is not None:
                            placeholder_string += self.video_token * (video_chunk_index[1] - video_chunk_index[0])

                        if audio_chunk_index is not None:
                            placeholder_string += self.audio_token * (audio_chunk_index[1] - audio_chunk_index[0])

                    placeholder_string += "<|audio_eos|>" + "<|vision_eos|>"
                    content = content.replace(VIDEO_PLACEHOLDER, placeholder_string, 1)
                    content = content.replace(AUDIO_PLACEHOLDER, "", 1)
                    num_audio_tokens += 1
                    num_video_tokens += 1
            else:
                while AUDIO_PLACEHOLDER in content:
                    audio_seqlen = audio_lengths[num_audio_tokens] if self.expand_mm_tokens else 1
                    content = content.replace(
                        AUDIO_PLACEHOLDER, f"<|audio_bos|>{self.audio_token * audio_seqlen}<|audio_eos|>", 1
                    )
                    num_audio_tokens += 1

                while VIDEO_PLACEHOLDER in content:
                    video_seqlen = (
                        video_grid_thw[num_video_tokens].prod() // merge_length if self.expand_mm_tokens else 1
                    )
                    content = content.replace(
                        VIDEO_PLACEHOLDER, f"<|vision_bos|>{self.video_token * video_seqlen}<|vision_eos|>", 1
                    )
                    num_video_tokens += 1

            message["content"] = content

        return messages


@dataclass
class VideoLlavaPlugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        num_image_tokens, num_video_tokens = 0, 0
        messages = deepcopy(messages)
        num_frames = 0
        if self.expand_mm_tokens:
            mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
            if "pixel_values_images" in mm_inputs:
                height, width = get_image_size(to_numpy_array(mm_inputs["pixel_values_images"][0]))
                num_frames = 1

            if "pixel_values_videos" in mm_inputs:
                one_video = to_numpy_array(mm_inputs["pixel_values_videos"][0])
                height, width = get_image_size(one_video[0])
                num_frames = one_video.shape[0]

            if "pixel_values_images" in mm_inputs or "pixel_values_videos" in mm_inputs:
                image_seqlen = (height // processor.patch_size) * (
                    width // processor.patch_size
                ) + processor.num_additional_image_tokens
                video_seqlen = image_seqlen * num_frames
                if processor.vision_feature_select_strategy == "default":
                    image_seqlen -= 1
        else:
            image_seqlen, video_seqlen = 1, 1

        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                content = content.replace(IMAGE_PLACEHOLDER, "{{image}}" * image_seqlen, 1)
                num_image_tokens += 1

            while VIDEO_PLACEHOLDER in content:
                content = content.replace(VIDEO_PLACEHOLDER, "{{video}}" * video_seqlen, 1)
                num_video_tokens += 1

            content = content.replace("{{image}}", self.image_token)
            message["content"] = content.replace("{{video}}", self.video_token)

        return messages


PLUGINS = {
    "base": BasePlugin,
    "gemma3": Gemma3Plugin,
    "intern_vl": InternVLPlugin,
    "kimi_vl": KimiVLPlugin,
    "llama4": Llama4Plugin,
    "llava": LlavaPlugin,
    "llava_next": LlavaNextPlugin,
    "llava_next_video": LlavaNextVideoPlugin,
    "minicpm_v": MiniCPMVPlugin,
    "mllama": MllamaPlugin,
    "paligemma": PaliGemmaPlugin,
    "pixtral": PixtralPlugin,
    "qwen2_audio": Qwen2AudioPlugin,
    "qwen2_omni": Qwen2OmniPlugin,
    "qwen2_vl": Qwen2VLPlugin,
    "video_llava": VideoLlavaPlugin,
}


def register_mm_plugin(name: str, plugin_class: type["BasePlugin"]) -> None:
    r"""Register a multimodal plugin."""
    if name in PLUGINS:
        raise ValueError(f"Multimodal plugin {name} already exists.")

    PLUGINS[name] = plugin_class


def get_mm_plugin(
    name: str,
    image_token: Optional[str] = None,
    video_token: Optional[str] = None,
    audio_token: Optional[str] = None,
) -> "BasePlugin":
    r"""Get plugin for multimodal inputs."""
    if name not in PLUGINS:
        raise ValueError(f"Multimodal plugin `{name}` not found.")

    return PLUGINS[name](image_token, video_token, audio_token)