import io
import json
import logging
import os
import sys

import requests
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))
JSON_CONTENT_TYPE = "application/json"
JPEG_CONTENT_TYPE = "image/jpeg"
NUM_CLASSES = 133

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def net():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_features, NUM_CLASSES))
    return model


def model_fn(model_dir):
    logger.debug("In model_fn. Model directory is -")
    logger.debug(model_dir)

    model = net().to(device)

    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        logger.info("Loading the dog-classifier model")
        checkpoint = torch.load(f, map_location=device)
        model.load_state_dict(checkpoint)
        logger.info("MODEL-LOADED")
        logger.info("model loaded successfully")
    model.eval()
    return model


def input_fn(request_body, content_type=JPEG_CONTENT_TYPE):
    logger.info("Deserializing the input data.")

    logger.debug(f"Request body CONTENT-TYPE is: {content_type}")
    logger.debug(f"Request body TYPE is: {type(request_body)}")
    if content_type == JPEG_CONTENT_TYPE:
        return Image.open(io.BytesIO(request_body))

    if content_type == JSON_CONTENT_TYPE:
        logger.debug(f"Request body is: {request_body}")
        request = json.loads(request_body)
        logger.debug(f"Loaded JSON object: {request}")
        url = request["url"]
        img_content = requests.get(url).content
        return Image.open(io.BytesIO(img_content))

    raise Exception(
        "Requested unsupported ContentType in content_type: {}".format(content_type)
    )


def predict_fn(input_object, model):
    logger.info("In predict fn")
    test_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    logger.info("transforming input")
    processed_image = test_transform(input_object).unsqueeze(0).to(device)

    with torch.no_grad():
        logger.info("Calling model")
        prediction = model(processed_image)
    return prediction
