import gc
import json
import os
import pathlib
import time

import torch
from torchvision import models, datasets, tv_tensors
from torchvision.transforms import v2
import torch.utils.data
from tqdm import tqdm

from engine import train_one_epoch, evaluate


HEIGHT = 3984
WIDTH = 5312
SCALE = 0.75

NUM_EPOCHS = 20
BATCH_SIZE = 2


ROOT = pathlib.Path("../data") / "flight263_COCO"
IMAGES_PATH = str(ROOT / "img")

RAW_ANNOTATIONS_PATH = str(ROOT / "annotations" / "instances_default.json")
ANNOTATIONS_PATH = ROOT / "annotations/instances_annotated.json"

def create_annotated_subset(coco_dataset):
    idx_has_ann = []
    for i, entry in tqdm(enumerate(coco_dataset)):
        if len(entry[1]) > 0:
            idx_has_ann += [i]

    print(len(idx_has_ann))


    with open(RAW_ANNOTATIONS_PATH, "r") as f:
        instances = json.load(f)

    idxs = [x + 1 for x in idx_has_ann]
    instances["images"] = [x for x in instances["images"] if x["id"] in idxs]

    with open(ROOT / "annotations/instances_annotated.json", "w") as f:
        json.dump(instances, f)

if __name__ == "__main__":
    print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
    print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
    print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
    print(torch.cuda.memory_summary())
    # torch.manual_seed(0)

    gc.collect()
    torch.cuda.empty_cache()

    # Define training augmentation
    transforms = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(int(HEIGHT * SCALE)),
            # v2.RandomPhotometricDistort(p=1),
            v2.RandomPerspective(distortion_scale=0.6, p=1.0),
            v2.RandomRotation(degrees=(0, 180)),
            v2.RandomZoomOut(fill={tv_tensors.Image: (123, 117, 104), "others": 0}),
            v2.RandomIoUCrop(),
            v2.RandomHorizontalFlip(p=1),
            v2.SanitizeBoundingBoxes(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )


    # Check if annotated subset exists
    if not os.path.isfile(ROOT / "annotations/instances_annotated.json"):
        # Raw dataset
        coco_dataset = datasets.CocoDetection(IMAGES_PATH, RAW_ANNOTATIONS_PATH)
        create_annotated_subset(coco_dataset)

    # Import data subset
    dataset = datasets.CocoDetection(IMAGES_PATH, ANNOTATIONS_PATH, transforms=transforms)
    dataset = datasets.wrap_dataset_for_transforms_v2(
        dataset, target_keys=("boxes", "labels", "image_id")
    )

    train_dataset, test_dataset = tuple(torch.utils.data.random_split(dataset, [0.8, 0.2]))

    data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        drop_last=True,  # Drop remainder
        # We need a custom collation function here, since the object detection
        # models expect a sequence of images and target dictionaries. The default
        # collation function tries to torch.stack() the individual elements,
        # which fails in general for object detection, because the number of bouding
        # boxes varies between the images of a same batch.
        collate_fn=lambda batch: tuple(zip(*batch)),
    )

    data_loader_test = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        drop_last=True,  # Drop remainder
        # We need a custom collation function here, since the object detection
        # models expect a sequence of images and target dictionaries. The default
        # collation function tries to torch.stack() the individual elements,
        # which fails in general for object detection, because the number of bouding
        # boxes varies between the images of a same batch.
        collate_fn=lambda batch: tuple(zip(*batch)),
    )

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = models.get_model(
        "ssdlite320_mobilenet_v3_large", weights=None, weights_backbone=None
    )


    # move model to the right device
    model.to(device)


    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(
        params,
        lr=0.005,
        # momentum=0.9,
        weight_decay=0.0005,
    )

    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Update memory management
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    for epoch in range(NUM_EPOCHS):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

        # print(f"{[img.shape for img in imgs] = }")
        # print(f"{[type(target) for target in targets] = }")
        # for name, loss_val in loss_dict.items():
        #     print(f"{name:<20}{loss_val:.3f}")

    torch.save(model.state_dict(), f"model{time.time()}")