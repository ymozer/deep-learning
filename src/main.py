from test_model import test_model
from models.models import CNN
from utils.utils import print_left_and_right_aligned, convert_seconds, TimerThread, plot_loss_and_accuracy, fitness_func
import datasets.BDMediLeaves.BDML_dataset as BDML
import os
import re
import sys
import time
import click
import argparse
import threading
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet50, densenet201, efficientnet_v2_m, EfficientNet_V2_M_Weights, DenseNet201_Weights, ResNet50_Weights
from torchvision import transforms
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck


sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(".."))

# import datasets.kvasir.kvasir as Kvasir


class TrainLoop(threading.Thread):
    def __init__(self, total_epoch, loss, device=None):
        threading.Thread.__init__(self, daemon=True)
        self.daemon = True
        self.device = device
        self.total_epoch = total_epoch
        self.loss = loss
        self.step_count = 0
        self.training_loss_list = []
        self.training_acc_per_epoch = []
        self.validation_loss_list = []
        self.validation_acc_per_epoch = []
        self.print_info = ""
        self.epoch_broadcast = 0
        self.loss_broadcast = 0
        self.timer = TimerThread()
        click.echo(click.style("Training started", fg="green"))
        click.echo(click.style(f"Device:{device}", fg="green"))

    def __del__(self):
        print("Train loop thread deleted")

    def plot_loss_and_accuracy(self, training_loss_list, training_acc_per_epoch, validation_loss_list, validation_acc_per_epoch):
        pil_image= plot_loss_and_accuracy(training_loss_list, training_acc_per_epoch, validation_loss_list, validation_acc_per_epoch)
        filesplitted=re.split("[']", str(type(model)))
        pil_image.save(f"model_{filesplitted[1]}_e{self.epoch_broadcast+1}_augment{args.augment}.png")
        
    def accuracy(self, network, dataloader):
        network.eval()
        total_correct = 0
        total_instances = 0
        for images, labels in dataloader:
            labels = [labels_dict[label] for label in list(labels)]
            labels = torch.tensor(labels, dtype=torch.long, device=self.device)
            images, labels = images.to(device), labels.to(device)
            predictions = torch.argmax(network(images), dim=1)
            correct_predictions = sum(predictions == labels).item()
            total_correct += correct_predictions
            total_instances += len(images)
        return round(total_correct/total_instances, 3)

    def train_loop(self):
        try:
            for epoch in range(self.total_epoch):
                self.epoch_broadcast = epoch
                train_loss = 0
                train_acc = 0
                validation_loss = 0
                validation_acc = 0
                epoch_losses = []

                model.train()
                for i, (images, labels) in enumerate(train_loader):
                    self.step_count = i + 1

                    # convert all labels to numbers using labels_dict
                    labels = [labels_dict[label] for label in list(labels)]
                    labels = torch.tensor(
                        labels, dtype=torch.long, device=self.device)

                    # Move tensors to the configured device
                    images = images.to(device)
                    labels = labels.to(device)

                    # Forward pass
                    outputs = model(images)
                    self.loss = criterion(outputs, labels)
                    train_loss += self.loss.item() * images.size(0)
                    self.loss_broadcast = self.loss 
                    epoch_losses.append(self.loss.item() * images.size(0))

                    # Backward and optimize
                    optimizer.zero_grad()
                    self.loss.backward()
                    optimizer.step()

                    # Track the accuracy
                    total = labels.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    correct = (predicted == labels).sum().item()
                    train_acc += correct / total

                    if os.name == 'nt':
                        self.print_info = "Epoch [{}/{}], Step [{}/{}] Loss: {:.4f} Acc: {:.2f}".format(
                            epoch + 1, num_epochs, i + 1, total_step, train_loss / len(train_loader), train_acc / len(train_loader))
                
                self.training_loss_list.append(sum(epoch_losses) / len(train_loader))
                with torch.no_grad():
                    train_acc_epoch = self.accuracy(model, train_loader)
                    click.echo(click.style(f"\nTrain accuracy for epoch {epoch+1}: {train_acc_epoch}", fg="green"))
                    self.training_acc_per_epoch.append(train_acc_epoch)

                # Validation
                val_losses = []
                model.eval()
                with torch.no_grad():
                    for images, labels in validation_loader:
                        # convert all labels to numbers using labels_dict
                        labels = [labels_dict[label]
                                  for label in list(labels)]
                        labels = torch.tensor(
                            labels, dtype=torch.long, device=self.device)

                        # Move tensors to the configured device
                        images = images.to(device)
                        labels = labels.to(device)

                        # Forward pass
                        outputs = model(images)
                        valloss = criterion(outputs, labels)
                        validation_loss += valloss.item() * images.size(0)
                        val_losses.append(valloss.item() * images.size(0))

                        # Calculate accuracy
                        total = labels.size(0)
                        _, predicted = torch.max(outputs.data, 1)
                        correct = (predicted == labels).sum().item()
                        validation_acc += correct / total
                    val_accuracy = self.accuracy(model, validation_loader)
                    click.echo(click.style(f"\nValidation accuracy for epoch {epoch+1}: {val_accuracy}", fg="green"))
                    self.validation_acc_per_epoch.append(val_accuracy)
                    self.validation_loss_list.append(sum(val_losses) / len(validation_loader))
        except Exception as e:
            click.echo(click.style(f"Exception: {e}", fg="red"))
            sys.exit(1)
        finally:
            model_str = str(type(model)).split("'")[1].replace(".", "-")
            click.echo(click.style("Training finished", fg="magenta"))
            click.echo(click.style(f"Saving model to model_{model_str}.ckpt", fg="green"))
            torch.save({
                'type': type(model),
                'epoch': self.epoch_broadcast,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_list': self.training_loss_list,
                'criterion': criterion,
            }, f"model_{model_str}_augment{args.augment}_epoch{self.epoch_broadcast+1}.ckpt")
            click.echo(click.style(f"Model saved to model_{model_str}.ckpt", fg="green"))
            return self.training_loss_list, self.training_acc_per_epoch, self.validation_loss_list, self.validation_acc_per_epoch

    def run(self):
        training_loss_list, training_acc_per_epoch, validation_loss_list, validation_acc_per_epoch = self.train_loop()
        self.plot_loss_and_accuracy(training_loss_list, training_acc_per_epoch, validation_loss_list, validation_acc_per_epoch)



def transform_resize_and_to_tensor(height: int = 229, width: int = 299):
    return transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
    ])


def set_device(device):
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    click.echo(click.style(f"Device:\t\t\t\t{device}", fg="green"))
    return device


def test_func(model):
    test = True
    criterion = None
    test_loss_list = []
    test_acc_list = []
    # load model
    checkpoint = torch.load(args.model_path, map_location=device)

    # if key 'model_state_dict' exists, load it
    if not 'model_state_dict' in checkpoint.keys():
        click.echo(click.style("Model state dict not found", fg="red"))
        model.load_state_dict(checkpoint)
    else:
        click.echo(click.style(f"Model state dict found. ", fg="green"))
        model.load_state_dict(checkpoint['model_state_dict'])

        if 'optimizer_state_dict' in checkpoint.keys():
            click.echo(click.style(f"Optimizer found", fg="green"))
            # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'loss_list' in checkpoint.keys():
            click.echo(click.style(f"Loss found", fg="green"))
            train_loss_list = checkpoint['loss_list']
        else:
            click.echo(click.style("Loss not found", fg="red"))

        if 'criterion' in checkpoint.keys():
            click.echo(click.style(f"Criterion found", fg="green"))
            criterion = checkpoint['criterion']

        if 'epoch' in checkpoint.keys():
            click.echo(click.style(f"Epoch found", fg="green"))
            epoch = checkpoint['epoch']

    model = model.to(device)

    click.echo(click.style("Model loaded", fg="green"))
    click.echo(click.style(
        f"Model path:\t\t\t{args.model_path}", fg="green"))

    test_transform = transform_resize_and_to_tensor()
    test = BDML.BDML(split="Test", transform=test_transform)
    test_loader = DataLoader(
        dataset=test, batch_size=batch_size, shuffle=False)
    click.echo(click.style(
        f"test dataset length:\t\t{len(test)}", fg="green"))

    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    test_loss_list, test_acc_list = test_model(
        model, test_loader, device, criterion, labels_dict)
    # format 2 decimal places
    test_loss_list = [round(x, 2) for x in test_loss_list]
    test_acc_list = [round(x, 2) for x in test_acc_list]

    click.echo(click.style(
        f"Test loss:\t\t\t{np.min(test_loss_list)}", fg="green"))
    click.echo(click.style(
        f"Test accuracy:\t\t\t{np.max(test_acc_list)}", fg="green"))
    click.echo(click.style(
        f"Test loss list:\t\t\t{test_loss_list}", fg="green"))
    click.echo(click.style(
        f"Test accuracy list:\t\t{test_acc_list}", fg="green"))
    
def model_selection(model, device):
    if model == "CNN":
        model = CNN(height=height, width=width, channels=3, classes=10)
    elif model == "CNN2":
        from models.models import CNN2
        model = CNN2(channels=3, classes=10)
    elif model == "ResNet50":
        model = resnet50()
    elif model == "DenseNet201":
        model = densenet201()
    elif model == "EfficientNetV2M":
        model = efficientnet_v2_m()
    elif model == "EfficientNetV2MWeights":
        model = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights)
    elif model == "DenseNet201Weights":
        model = DenseNet201_Weights()
    elif model == "ResNet50Weights":
        model = resnet50(weights=ResNet50_Weights)
    else:
        click.echo(click.style(f"Model {type(model)} not found", fg="red"))
        sys.exit(1)
    click.echo(click.style(f"Model type:\t\t\t{type(model)}", fg="green"))
    click.echo(click.style(f"Device:\t\t\t\t{device}", fg="green"))
    return model

if __name__ == "__main__":
    # Parse arguments
    device = None
    train = False
    test = False

    # labels dictionary
    labels_dict = {
        "Azadirachta indica":     0,
        "Calotropis gigantea":    1,
        "Centella asiatica":      2,
        "Hibiscus rosa-sinensis": 3,
        "Justicia adhatoda":      4,
        "Kalanchoe pinnata":      5,
        "Mikania micrantha":      6,
        "Ocimum tenuiflorum":     7,
        "Phyllanthus emblica":    8,
        "Terminalia arjuna":      9,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None,
                        help="Model to use for training")
    parser.add_argument("--train", action='store_true',
                        default=False, help="Train model")
    parser.add_argument("--augment", action='store_true',
                        default=False, help="Augment dataset")
    parser.add_argument("--test", action='store_true',
                        default=False, help="Test model")
    parser.add_argument("--validation", action='store_true',
                        default=False, help="Validate model")
    parser.add_argument("--device", type=str, default='cuda:0',
                        help="Device to use for training")
    parser.add_argument("--model_path", type=str,
                        default="model.ckpt", help="Path to model")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Number of epochs")
    parser.add_argument("--batch_size", type=int,
                        default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float,
                        default=0.001, help="Learning rate")
    parser.add_argument("--image_size", type=int,
                        default=32, help="Image size")

    args = parser.parse_args()

    batch_size = args.batch_size
    learning_rate = args.learning_rate
    width, height = args.image_size, args.image_size

    device = args.device
    if device is None:
        device = set_device(device)

    model=model_selection(args.model, device)

    if args.train:
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        train_bool = True
        num_epochs = args.epochs
        train_dataset = None

        train_transform = transform_resize_and_to_tensor(height=height, width=width)
        if args.augment:
            train_dataset = BDML.BDML(split="Train", transform=train_transform,augment=True, download=True)
        else:
            train_dataset = BDML.BDML(split="Train", transform=train_transform,augment=False, download=True)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        validation_transform = transform_resize_and_to_tensor()
        validation_dataset = BDML.BDML(split="Validation", transform=validation_transform)
        validation_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False)
        click.echo(click.style(f"Validation dataset length:\t{len(validation_dataset)}", fg="green"))
        click.echo(click.style(f"Train dataset length:\t\t{len(train_dataset)}", fg="green"))

        total_step = len(train_loader)
        click.echo(click.style(f"Total Step:\t\t\t{total_step}", fg="green"))
        click.echo(click.style(f"Epochs:\t\t\t\t{num_epochs}", fg="green"))

        # initialize train loop and timer thread
        train_loop = TrainLoop(total_epoch=num_epochs, loss=criterion, device=device)
        timer = TimerThread()

        if os.path.exists(args.model_path):
            # if model exists, load it and test it
            click.echo(click.style("Model already exists. loading..,", fg="green"))
            checkpoint = torch.load(args.model_path)
            model.load_state_dict(checkpoint)
        else:
            # if model doesn't exist, train it
            try:
                train_loop.start()
                timer.start()
                while True:
                    print_left_and_right_aligned(
                        train_loop.print_info, f"Time: {convert_seconds(timer.seconds)}")
                    time.sleep(0.1)
                    if not train_loop.is_alive():
                        click.echo(click.style("Joining threads.", fg="green"))
                        try:
                            train_loop.join(timeout=2)
                            timer.join(timeout=2)
                        except RuntimeError:
                            click.echo(click.style(
                                "Threads already joined.", fg="red"))
                        except Exception as e:
                            click.echo(click.style(
                                f"Exception: {e}", fg="red"))

                        click.echo(click.style("Threads joined.", fg="green"))
                        break
            except KeyboardInterrupt:
                click.echo(click.style("Training interrupted", fg="red"))
                click.echo(click.style("Saving interrupted model", fg="green"))
                model_str = str(type(model)).split("'")[1].replace(".", "-")
                torch.save({
                    'type': type(model),
                    'epoch': train_loop.epoch_broadcast,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss_list': train_loop.training_loss_list,
                    'criterion': criterion,
                }, f"model_{model_str}_augment{args.augment}_epoch{train_loop.epoch_broadcast}.ckpt")
                print(train_loop.epoch_broadcast)
                print(train_loop.loss_broadcast)
                sys.exit(0)
            finally:
                if args.test:
                    click.echo(click.style("Testing model", fg="green"))
                    test_transform = transform_resize_and_to_tensor()
                    test = BDML.BDML(split="Test", transform=test_transform)
                    test_loader = DataLoader(dataset=test, batch_size=batch_size, shuffle=False)
                    click.echo(click.style(f"Test dataset length:\t\t{len(test)}", fg="green"))
                    test_model(model, test_loader, device, criterion, labels_dict)
                    click.echo(click.style("Testing finished", fg="magenta"))

    elif args.test:
        click.echo(click.style("Testing model", fg="magenta"))
        test_func(model)

