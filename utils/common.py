import torch
from PIL import Image
from torchvision.transforms import v2

to_tensor = v2.Compose(
    [
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def pil2torch(image: Image.Image, resize):
    return to_tensor(image)
