"""Stable Diffusion feature extractor via ODISE.

Extracts multi-scale UNet decoder activations (s3, s4, s5) from SD v1-5.
Uses the ODISE fork's modified backbone that returns raw features.
"""

import itertools
from contextlib import ExitStack

import numpy as np
import torch
from detectron2.config import instantiate, LazyCall as L
from detectron2.data import MetadataCatalog, transforms as T
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from detectron2.evaluation import inference_context
from detectron2.utils.env import seed_all_rng
from detectron2.utils.visualizer import random_color
from mask2former.data.datasets.register_ade20k_panoptic import ADE20K_150_CATEGORIES

from odise import model_zoo
from odise.checkpoint import ODISECheckpointer
from odise.config import instantiate_odise
from odise.data import get_openseg_labels
from odise.modeling.wrapper import OpenPanopticInference


COCO_THING_CLASSES = [
    label for idx, label in enumerate(get_openseg_labels("coco_panoptic", True))
    if COCO_CATEGORIES[idx]["isthing"] == 1
]
COCO_THING_COLORS = [c["color"] for c in COCO_CATEGORIES if c["isthing"] == 1]
COCO_STUFF_CLASSES = [
    label for idx, label in enumerate(get_openseg_labels("coco_panoptic", True))
    if COCO_CATEGORIES[idx]["isthing"] == 0
]
COCO_STUFF_COLORS = [c["color"] for c in COCO_CATEGORIES if c["isthing"] == 0]

ADE_THING_CLASSES = [
    label for idx, label in enumerate(get_openseg_labels("ade20k_150", True))
    if ADE20K_150_CATEGORIES[idx]["isthing"] == 1
]
ADE_THING_COLORS = [c["color"] for c in ADE20K_150_CATEGORIES if c["isthing"] == 1]
ADE_STUFF_CLASSES = [
    label for idx, label in enumerate(get_openseg_labels("ade20k_150", True))
    if ADE20K_150_CATEGORIES[idx]["isthing"] == 0
]
ADE_STUFF_COLORS = [c["color"] for c in ADE20K_150_CATEGORIES if c["isthing"] == 0]

LVIS_CLASSES = get_openseg_labels("lvis_1203", True)
LVIS_COLORS = list(itertools.islice(
    itertools.cycle([c["color"] for c in COCO_CATEGORIES]), len(LVIS_CLASSES)
))


class StableDiffusionSeg(object):
    def __init__(self, model, metadata, aug):
        self.model = model
        self.metadata = metadata
        self.aug = aug
        self.cpu_device = torch.device("cpu")

    def get_features(self, original_image, caption=None, pca=None):
        height, width = original_image.shape[:2]
        aug_input = T.AugInput(original_image, sem_seg=None)
        self.aug(aug_input)
        image = torch.as_tensor(aug_input.image.astype("float32").transpose(2, 0, 1))
        inputs = {"image": image, "height": height, "width": width}
        if caption is not None:
            features = self.model.get_features([inputs], caption, pca=pca)
        else:
            features = self.model.get_features([inputs], pca=pca)
        return features


def _build_classes_and_metadata(vocab, label_list):
    extra_classes = []
    if vocab:
        for words in vocab.split(";"):
            extra_classes.append([word.strip() for word in words.split(",")])
    extra_colors = [random_color(rgb=True, maximum=1) for _ in range(len(extra_classes))]

    thing_classes = extra_classes
    stuff_classes = []
    thing_colors = extra_colors
    stuff_colors = []

    if "COCO" in label_list:
        thing_classes += COCO_THING_CLASSES
        stuff_classes += COCO_STUFF_CLASSES
        thing_colors += COCO_THING_COLORS
        stuff_colors += COCO_STUFF_COLORS
    if "ADE" in label_list:
        thing_classes += ADE_THING_CLASSES
        stuff_classes += ADE_STUFF_CLASSES
        thing_colors += ADE_THING_COLORS
        stuff_colors += ADE_STUFF_COLORS
    if "LVIS" in label_list:
        thing_classes += LVIS_CLASSES
        thing_colors += LVIS_COLORS

    MetadataCatalog.pop("odise_demo_metadata", None)
    meta = MetadataCatalog.get("odise_demo_metadata")
    meta.thing_classes = [c[0] for c in thing_classes]
    meta.stuff_classes = [*meta.thing_classes, *[c[0] for c in stuff_classes]]
    meta.thing_colors = thing_colors
    meta.stuff_colors = thing_colors + stuff_colors
    meta.stuff_dataset_id_to_contiguous_id = {i: i for i in range(len(meta.stuff_classes))}
    meta.thing_dataset_id_to_contiguous_id = {i: i for i in range(len(meta.thing_classes))}

    return thing_classes + stuff_classes, meta


def load_model(diffusion_ver="v1-5", image_size=960, num_timesteps=50,
               block_indices=(2, 5, 8, 11), seed=42):
    cfg = model_zoo.get_config("Panoptic/odise_label_coco_50e.py", trained=True)
    cfg.model.backbone.feature_extractor.init_checkpoint = "sd://" + diffusion_ver
    cfg.model.backbone.feature_extractor.steps = (num_timesteps,)
    cfg.model.backbone.feature_extractor.unet_block_indices = block_indices
    cfg.model.backbone.feature_extractor.encoder_only = False
    cfg.model.backbone.feature_extractor.decoder_only = True
    cfg.model.backbone.feature_extractor.resblock_only = False
    cfg.model.overlap_threshold = 0
    seed_all_rng(seed)

    cfg.dataloader.test.mapper.augmentations = [
        L(T.ResizeShortestEdge)(short_edge_length=image_size, sample_style="choice", max_size=2560),
    ]
    aug = instantiate(cfg.dataloader.test.mapper).augmentations
    model = instantiate_odise(cfg.model)
    model.to(cfg.train.device)
    ODISECheckpointer(model).load(cfg.train.init_checkpoint)
    return model, aug


def process_features_and_mask(model, aug, image, category=None, input_text=None, mask=False, raw=True):
    caption = input_text
    vocab = ""
    label_list = ["COCO"]
    classes, meta = _build_classes_and_metadata(vocab, label_list)

    with ExitStack() as stack:
        inference_model = OpenPanopticInference(
            model=model, labels=classes, metadata=meta,
            semantic_on=False, instance_on=False, panoptic_on=True,
        )
        stack.enter_context(inference_context(inference_model))
        stack.enter_context(torch.no_grad())

        demo = StableDiffusionSeg(inference_model, meta, aug)
        if caption is not None:
            features = demo.get_features(np.array(image), caption, pca=raw)
        else:
            features = demo.get_features(np.array(image), pca=raw)
    return features
