
import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(".."))

import torch


#test model 
def test_model(model, test_loader, device, criterion, labels_dict):
  test_loss = 0
  test_acc = 0
  test_loss_list = []
  test_acc_list = []
  
  model.eval()
  with torch.no_grad():
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
      test_loss += loss.item()

      # Calculate accuracy
      _, predicted = torch.max(outputs.data, 1)
      correct = (predicted == labels).sum().item()
      test_acc += correct / len(labels)

      test_loss_list.append(loss.item())
      test_acc_list.append(correct / len(labels))

  return  test_loss_list, test_acc_list



