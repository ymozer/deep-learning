
import os
import sys
import click
import torch
sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(".."))

from models.models import CNN
from torchvision import transforms
from torch.utils.data import DataLoader
import datasets.BDMediLeaves.BDML_dataset as BDML

import torchmetrics
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification.confusion_matrix import MulticlassConfusionMatrix, ConfusionMatrix
from torchmetrics import F1Score, Precision, Recall
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision, MulticlassRecall

from matplotlib import pyplot as plt
from PIL import Image
import pandas as pd

# supreess future warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# surpress warnings from torcheval
import logging
logging.getLogger("torcheval").setLevel(logging.ERROR)


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

def model_selection(model, device):
    if model == "CNN":
        model = CNN(height=height, width=width, channels=3, classes=10)
    elif model == "CNN2":
        from models.models import CNN2
        model = CNN2(channels=3, classes=10)
    elif model == "ResNet50":
        from torchvision.models import resnet50
        model = resnet50()
    elif model == "DenseNet201":
        from torchvision.models import densenet201
        model = densenet201()
    elif model == "EfficientNetV2M":
        from torchvision.models import efficientnet_v2_m
        model = efficientnet_v2_m()
    elif model == "EfficientNetV2MWeights":
        from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights
        model = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights)
    elif model == "DenseNet201Weights":
        from torchvision.models import densenet201, DenseNet201_Weights
        model = DenseNet201_Weights()
    elif model == "ResNet50Weights":
        from torchvision.models import resnet50, ResNet50_Weights
        model = resnet50(weights=ResNet50_Weights)
    elif model == "InceptionV3":
        from torchvision.models import inception_v3
        model = inception_v3()
    elif model == "InceptionV3Weights":
        from torchvision.models import inception_v3, Inception_V3_Weights
        model = inception_v3(weights=Inception_V3_Weights.DEFAULT)  
    elif model == "MobileNetV3Small":
        from torchvision.models import mobilenet_v3_small
        model = mobilenet_v3_small()  
    elif model == "MobileNetV3SmallWeights":
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights 
        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    elif model == "MobileNetV3Large":
        from torchvision.models import mobilenet_v3_large
        model = mobilenet_v3_large()
    elif model == "MobileNetV3LargeWeights":
        from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
        model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
    else:
        click.echo(click.style(f"Model {type(model)} not found", fg="red"))
        sys.exit(1)
    click.echo(click.style(f"Model type:\t\t\t{type(model)}", fg="green"))
    click.echo(click.style(f"Device:\t\t\t\t{device}", fg="green"))
    return model

if __name__ == '__main__':
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  criterion = torch.nn.CrossEntropyLoss()

  if not os.path.exists("outputs"):
    os.makedirs("outputs")
  
  batch_size = 32
  
  test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
  ])

  models_path = {
    "CNN2": {
      "path":"models/model_models-models-CNN2_augmentFalse_epoch10.ckpt",
    },
    "CNN2_augmented": {
      "path":"models/model_models-models-CNN2_augmentTrue_epoch10.ckpt",
    },
    "DenseNet201": {
      "path":"models/model_torchvision-models-densenet-DenseNet_augmentFalse_epoch10_lr1e-05.ckpt",
    },
    "DenseNet201_augment": {
      "path":"models/model_torchvision-models-densenet-DenseNet_augmentTrue_epoch10_lr1e-05.ckpt",
    },
    "MobileNetV3Large": { 
      "path":"models/model_torchvision-models-mobilenetv3-MobileNetV3_augmentFalse_epoch10_lr1e-05.ckpt",
    },
    "MobileNetV3Large_augment": {
      "path":"models/model_torchvision-models-mobilenetv3-MobileNetV3_augmentTrue_epoch10_lr1e-05.ckpt",
    },
    "EfficientNetV2M": {
      "path":"models/model_torchvision-models-efficientnet-EfficientNet_augmentTrue_epoch10_lr1e-05.ckpt",
    },
    "EfficientNetV2M_augment": {
      "path":"models/model_torchvision-models-efficientnet-EfficientNet_augmentFalse_epoch10_lr1e-05.ckpt",
    },
    "ResNet50": {
      "path":"models/model_torchvision-models-resnet-ResNet_augmentFalse_epoch10_lr1e-05.ckpt",
    },
    "ResNet50_augment": {
      "path":"models/model_torchvision-models-resnet-ResNet_augmentTrue_epoch10_lr1e-05.ckpt",
    },
  }

  accuracy  = Accuracy(task="multiclass", num_classes=10)
  f1        = MulticlassF1Score  (num_classes=10,average='macro')
  precision = MulticlassPrecision(num_classes=10,average='macro')
  recall    = MulticlassRecall   (num_classes=10,average='macro')
  conf_mat  = ConfusionMatrix(task="multiclass", num_classes=10)

  accuracy.to(device)
  f1.to(device)
  precision.to(device)
  recall.to(device)
  conf_mat.to(device)

  df_class = pd.DataFrame(columns=['algorithm', 'plant_class', 'f1_score', 'precision_score', 'recall_score', 'accuracy_score'])
  for model_path in models_path.items():
    checkpoint = torch.load(model_path[1]["path"], map_location=device)
    if "_" in model_path[0]:
      if model_path[0].split("_")[1] == "augment":
        model_type = model_path[0].split("_")[0]
        model=model_selection(model_type, device)
    else:
      model_type = model_path[0]
      model=model_selection(model_type, device)

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
    # ----------------------------Model Testing------------------------------------------
    print(f"Testing {model_path[0]} model generally")
    model.eval()
    model = model.to(device)
    with torch.no_grad():
      test_dataset = BDML.BDML(split="Test", transform=test_transform)
      test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
      for images, labels in test_loader:
        # convert all labels to numbers using labels_dict
        labels = [labels_dict[label] for label in list(labels)]
        labels = torch.tensor(labels, dtype=torch.long, device=device)

        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        test_acc        = correct / len(labels)

        accuracy_score  = accuracy(predicted, labels)
        f1_score        = f1(predicted, labels)
        precision_score = precision(predicted, labels)
        recall_score    = recall(predicted, labels)
        conf_mat.update(predicted, labels)
        fig, ax = conf_mat.plot()
        fig.savefig(f"outputs/confusion_matrix_{model_path[0]}.png")
        plt.close(fig)
        conf_mat.reset()

        df_class = df_class._append(
        {
          'algorithm': model_path[0], # 'DenseNet201', 'MobileNetV3Large', 'EfficientNetV2M', 'ResNet50
          'plant_class': 'all', 
          'f1_score': f1_score.tolist(), 
          'precision_score': precision_score.tolist(), 
          'recall_score': recall_score.tolist(), 
          'accuracy_score': accuracy_score.tolist(),
        }, ignore_index=True)

    # ----------------------------Model Testing Per Plant------------------------------------------
    for plant_class in labels_dict.keys():
      print(f"Testing {model_path[0]} model on {plant_class} class")
      test_dataset = BDML.BDML(split="Test", selected_plant_class=plant_class, transform=test_transform)
      test_loader = DataLoader(test_dataset, batch_size=50, shuffle=False)
      df_per_plant = pd.DataFrame(columns=['algorithm','plant_class', 'f1_score', 'precision_score', 'recall_score', 'accuracy_score'])
      for images, labels in test_loader:
        # convert all labels to numbers using labels_dict
        labels = [labels_dict[plant_class] for _ in range(len(images))]
        labels = torch.tensor(labels, dtype=torch.long, device=device)

        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        test_acc        = correct / len(labels)

        accuracy_score  = accuracy(predicted, labels)
        f1_score        = f1(predicted, labels)
        precision_score = precision(predicted, labels)
        recall_score    = recall(predicted, labels)

        df_per_plant = df_per_plant._append(
        {
          'algorithm': model_path[0], # 'DenseNet201', 'MobileNetV3Large', 'EfficientNetV2M', 'ResNet50
          'plant_class': plant_class, 
          'f1_score': f1_score.tolist(), 
          'precision_score': precision_score.tolist(), 
          'recall_score': recall_score.tolist(), 
          'accuracy_score': accuracy_score.tolist(),
        }, ignore_index=True)
        df_class = df_class._append(df_per_plant, ignore_index=True)
  df_class.to_csv("outputs/test_results_per_plant_class.csv", index=False)

  df_class_all = pd.DataFrame(columns=['algorithm','plant_class', 'f1_score', 'precision_score', 'recall_score', 'accuracy_score', 'confusion_matrix'])
  for index, row in df_class.items():
    if row['plant_class'] == 'all':
      df_class_all = df_class_all._append(row, ignore_index=True)
    # plot all metrics
    fig, ax = plt.subplots()
    ax.plot(row['f1_score'], label='f1_score')
    ax.plot(row['precision_score'], label='precision_score')
    ax.plot(row['recall_score'], label='recall_score')
    ax.plot(row['accuracy_score'], label='accuracy_score')
    ax.legend()
    ax.set_title(f"{row['algorithm']}_{row['plant_class']}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    fig.savefig(f"outputs/{row['algorithm']}_{row['plant_class']}.png")
    plt.close(fig)
  df_class_all.to_csv("outputs/test_results_all.csv", index=False)
  print("Testing finished")
