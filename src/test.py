
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
from torcheval.metrics.classification.f1_score import MulticlassF1Score

from torcheval.metrics.classification.precision import MulticlassPrecision
from torcheval.metrics.classification.recall import MulticlassRecall
from torcheval.metrics.classification.accuracy import MulticlassAccuracy


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

  batch_size = 32
  
  test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
  ])

  models_path = {
    "DenseNet201": {
      "path":"model_torchvision-models-densenet-DenseNet_augmentFalse_epoch10_lr1e-05.ckpt",
    },
    "DenseNet201_augment": {
      "path":"model_torchvision-models-densenet-DenseNet_augmentTrue_epoch10_lr1e-05.ckpt",
    },
    "MobileNetV3Large": { 
      "path":"model_torchvision-models-mobilenetv3-MobileNetV3_augmentFalse_epoch10_lr1e-05.ckpt",
    },
    "MobileNetV3Large_augment": {
      "path":"model_torchvision-models-mobilenetv3-MobileNetV3_augmentTrue_epoch10_lr1e-05.ckpt",
    },
    "EfficientNetV2M": {
      "path":"model_torchvision-models-efficientnet-EfficientNet_augmentTrue_epoch10_lr1e-05.ckpt",
    },
    "EfficientNetV2M_augment": {
      "path":"model_torchvision-models-efficientnet-EfficientNet_augmentFalse_epoch10_lr1e-05.ckpt",
    },
    "ResNet50": {
      "path":"model_torchvision-models-resnet-ResNet_augmentFalse_epoch10_lr1e-05.ckpt",
    },
    "ResNet50_augment": {
      "path":"model_torchvision-models-resnet-ResNet_augmentTrue_epoch10_lr1e-05.ckpt",
    },
  }

  f1        = MulticlassF1Score  (num_classes=10,device=device,average='macro')
  precision = MulticlassPrecision(num_classes=10,device=device,average='macro')
  recall    = MulticlassRecall   (num_classes=10,device=device,average='macro')
  accuracy  = MulticlassAccuracy (num_classes=10,device=device,average='macro')

  #model=None

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
      

    for plant_class in labels_dict.keys():
      print(f"Testing {model_path[0]} model on {plant_class} class")
      model.eval()
      model = model.to(device)
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
        f1        .update(predicted, labels)
        precision .update(predicted, labels)
        recall    .update(predicted, labels)
        accuracy  .update(predicted, labels)
        print(predicted)
        print(labels)

        f1_score        = f1        .compute()
        precision_score = precision .compute()
        recall_score    = recall    .compute()
        accuracy_score  = accuracy  .compute()
        
        print(f1_score.tolist(), f1_score.tolist(), precision_score.tolist(),  accuracy_score.tolist())
        df_per_plant = df_per_plant._append(
          {
            'algorithm': model_path[0], # 'DenseNet201', 'MobileNetV3Large', 'EfficientNetV2M', 'ResNet50
            'plant_class': plant_class, 
            'f1_score': f1_score.tolist(), 
            'precision_score': precision_score.tolist(), 
            'recall_score': 0, 
            'accuracy_score': accuracy_score.tolist()
          }, ignore_index=True)
        

        f1.reset()
        precision.reset()
        recall.reset()
        accuracy.reset()
        df_class = df_class._append(df_per_plant, ignore_index=True)

        print(f"Testing {model_path[0]} model on {plant_class} class finished")
  df_class.to_csv("test_results.csv", index=False)





