import argparse
import os

import smdebug.pytorch as smd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import ImageFile
from torch.utils.data import DataLoader

# I had a problem with 'truncated image', setting this flag fixes the issue
# See: https://discuss.pytorch.org/t/oserror-image-file-is-truncated-150-bytes-not-processed/64445
ImageFile.LOAD_TRUNCATED_IMAGES = True

NUM_CLASSES = 133


def test(model, test_loader, criterion, device, hook=None):
    model.eval()
    if hook:
        hook.set_mode(smd.modes.EVAL)

    test_loss = 0
    correct = 0
    total = len(test_loader.dataset)

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[
                1
            ]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= total
    print(
        f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{total} ({correct/total:.2%})"
    )


def train(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    epoch,
    hook=None,
    log_interval: int = 10,
    dry_run: bool = False,
):
    model.train()
    if hook:
        hook.set_mode(smd.modes.TRAIN)

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            pred = output.max(1, keepdim=True)[
                1
            ]  # get the index of the max log-probability
            correct = pred.eq(target.view_as(pred)).sum().item()
            accuracy = correct / pred.shape[0]
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                f"({batch_idx / len(train_loader):.0%})]\tLoss: {loss.item():.6f}"
                f"\t Accuracy: {accuracy:.2%}"
            )
            if dry_run:
                break


def net(model_name):
    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    elif model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_features, NUM_CLASSES))
    return model


def create_data_loaders(train_dir, test_dir, batch_size):
    """
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    """

    training_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # These transforms are taken from
    # https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html#torchvision.models.resnet50
    testing_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_data = torchvision.datasets.ImageFolder(
        root=train_dir, transform=training_transform
    )
    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    test_data = torchvision.datasets.ImageFolder(
        root=test_dir, transform=testing_transform
    )
    test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_data_loader, test_data_loader


def create_hook():
    # Check if running inside SageMaker training job
    if os.getenv("SM_TRAINING_ENV"):
        return smd.Hook.create_from_json_file()
    else:
        print("Running locally, using manual hook configuration.")
        return smd.Hook(
            out_dir="./debug_output", include_collections=["gradients", "biases"]
        )


def main(args):

    model = net(args.model_name)

    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), args.learning_rate, weight_decay=args.weight_decay
    )

    hook = None
    try:
        hook = create_hook()
        hook.register_module(model)
        hook.register_loss(loss_criterion)
        print(f"My hook is {hook}")
    except AttributeError:
        print("CANNOT CREATE THE SM DEBUG HOOK!")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device {device}")

    train_loader, test_loader = create_data_loaders(
        args.train, args.test, args.batch_size
    )

    for epoch in range(args.epochs):
        train(
            model,
            train_loader,
            loss_criterion,
            optimizer,
            device,
            epoch,
            hook,
            args.log_interval,
            args.dry_run,
        )

        test(model, test_loader, loss_criterion, device, hook)
        if args.dry_run:
            return

    with open(os.path.join(args.model_dir, "model.pth"), "wb") as f:
        torch.save(model.state_dict(), f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for testing (default: 64)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=14,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        metavar="LR",
        help="learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        metavar="L2",
        help="weight decay (L2 penalty) (default: 0)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="log training loss each N batches (default: 10)",
    )
    parser.add_argument(
        "--train",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/training"),
        metavar="TEXT",
        help="folder where the training data is saved",
    )
    parser.add_argument(
        "--test",
        type=str,
        default=os.environ.get("SM_CHANNEL_TEST", "/opt/ml/input/data/test"),
        metavar="TEXT",
        help="folder where the test data is saved",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="resnet50",
        metavar="TEXT",
        help="which pretrained model to choose. Possible options: resnet18, resnet50 (default: resnet50)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"),
        metavar="TEXT",
        help="folder where the trained model will be saved",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="quickly check a single pass",
    )

    args = parser.parse_args()

    main(args)
