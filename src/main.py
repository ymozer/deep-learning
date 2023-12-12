import io
import os
import sys
import threading
from tabnanny import check
sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(".."))

import time
import click
import shutil  

import torch
import nvidia_smi
import datasets.BDMediLeaves.BDML_dataset as BDML
import datasets.kvasir.kvasir as Kvasir
from torchvision import transforms
from torchvision.models import resnet50, densenet201, efficientnet_v2_m, EfficientNet_V2_M_Weights
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch.nn as nn




def print_left_and_right_aligned(left_text, right_text):
    terminal_width = shutil.get_terminal_size().columns
    left_text_width = len(left_text)
    available_width_for_right_text = terminal_width - left_text_width 
    truncated_right_text = right_text[:available_width_for_right_text]
    click.echo(left_text, nl=False)
    remaining_space = terminal_width - len(left_text) - len(truncated_right_text)
    click.echo(" " * remaining_space, nl=False)
    click.echo(truncated_right_text, nl=False)
    click.echo("\r", nl=False)
    time.sleep(0.1)


# Define the CNN model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(32 * 57 * 75, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 32 * (height//4) * (width//4))
        x = nn.functional.relu(x)
        x = nn.functional.softmax(x, dim=1)
        return x
    
class CNN(nn.Module):
    def __init__(self, num_classes=10, device=None):
        super(CNN, self).__init__()
        self.num_classes = num_classes
        # Define the feature extraction part
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ).to(device)
        
        # The classifier will be defined later in the forward method
        self.classifier = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass through the feature extraction part
        x = self.features(x)
        
        # Dynamically obtain the input features for the classifier
        x_flatten_size = x.view(x.size(0), -1).size(1)
        
        # Define the classifier if it hasn't been defined yet
        if self.classifier is None:
            self.classifier = nn.Sequential(
                nn.Linear(x_flatten_size, 1024),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(1024, self.num_classes),
            ).to(device)
        
        # Flatten the output for the classification part
        x = x.view(x.size(0), -1).to(device)
        
        # Forward pass through the classification part
        x = self.classifier(x)
        return x


class TimerThread(threading.Thread):

  def __init__(self):
    threading.Thread.__init__(self)
    self.daemon = True
    self.seconds = 0

  def __del__(self):
    print("Timer thread deleted")

  def timer_thread(self):
    while True:
      self.seconds += 1
      time.sleep(1)

  def run(self):
    self.timer_thread()

class TrainLoop(threading.Thread):
    def __init__(self, total_epoch, loss, device=None):
        threading.Thread.__init__(self, daemon=True)
        self.daemon = True
        self.device = device
        self.total_epoch = total_epoch
        self.loss = loss
        self.step_count = 0
        self.training_loss_list = []
        self.training_acc_list = []
        self.print_info = ""
        self.timer = TimerThread()
        click.echo(click.style("Training started", fg="green"))
        click.echo(click.style(f"Device:{device}", fg="green"))

    def __del__(self):
        print("Train loop thread deleted")
        
    def train_loop(self):
        try:
            for epoch in range(self.total_epoch):
                train_loss = 0
                train_acc = 0

                # Training
                model.train()

                for i, (images, labels) in enumerate(train_loader):
                    self.step_count = i + 1

                    # convert all labels to numbers using labels_dict
                    labels = [labels_dict[label] for label in list(labels)]
                    labels = torch.tensor(labels, dtype=torch.long, device=self.device)

                    # Move tensors to the configured device
                    images = images.to(device)
                    labels = labels.to(device)                    

                    # Forward pass
                    outputs = model(images)
                    self.loss = criterion(outputs, labels)
                    train_loss += self.loss.item()

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
                        self.print_info = "Epoch [{}/{}], Step [{}/{}] Loss: {:.4f} Acc: {:.2f} Batch size: {}".format(
                            epoch + 1, num_epochs, i + 1, total_step, self.loss.item(), train_acc / (i + 1) * 100, batch_size)
                        
                self.training_loss_list.append(train_loss / len(train_loader))
                self.training_acc_list.append(train_acc / len(train_loader))
            # save model
            torch.save(model.state_dict(), 'model.ckpt')
        except KeyboardInterrupt:
            click.echo(click.style(f"Training interrupted at {self.step_count}/{total_step}", fg="red"))
            click.echo(click.style("Saving model", fg="green"))
        #torch.save({
        #    'epoch': epoch,
        #    'model_state_dict': model.state_dict(),
        #    'optimizer_state_dict': optimizer.state_dict(),
        #    'loss': loss,
        #}, f"model.ckpt")
        #click.echo(click.style("Model saved", fg="green"))
        #sys.exit(0)

    def run(self):
        self.train_loop()


if __name__ == "__main__":
    train_transform = transforms.Compose([
        transforms.Resize((229, 299)),
        '''
        # Augmentation
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        ''',
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((229, 299)),
        transforms.ToTensor(),
    ])

    # Set device MPS
    # Check that MPS is available
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')    
    # print device
    click.echo(click.style(f"Device:\t\t\t\t{device}", fg="green"))

    # Load dataset and apply transformations
    train = BDML.BDML(split="Train", transform=train_transform, augment=False, download=True)
    test = BDML.BDML(split="Test", transform=test_transform)
    validation = BDML.BDML(split="Validation", transform=test_transform)
    click.echo(click.style(f"train dataset length:\t\t{len(train)}", fg="green"))
    click.echo(click.style(f"test dataset length:\t\t{len(test)}", fg="green"))
    click.echo(click.style(f"validation dataset length:\t{len(validation)}", fg="green"))

    device = None
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')


    # Hyperparameters
    num_epochs = 3
    batch_size = 32
    learning_rate = 0.001

    # Data loaders
    train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(dataset=test, batch_size=batch_size, shuffle=False)
    validation_loader = DataLoader(dataset=validation, batch_size=batch_size, shuffle=False)

    
    # Model
    #model = SimpleCNN(num_classes=10)
    #model = resnet50(pretrained=True)
    #model = densenet201(pretrained=True)
    model = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.DEFAULT, progress=True)
    #model = CNN(num_classes=10, device=device)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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

    # Train the model
    total_step = len(train_loader)
    click.echo(click.style(f"Total Step:\t\t\t{total_step}", fg="green"))

    since = time.time()
    train_loop = TrainLoop(total_epoch=num_epochs, loss=criterion, device=device)
    timer = TimerThread()
    if os.path.exists("model.ckpt"):
        checkpoint = torch.load("model.ckpt")
        # Update model and optimizer with the loaded state_dicts
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        click.echo(click.style("Model loaded", fg="green"))
        # TODO: test model

    else:
        click.echo(click.style("Model not found", fg="red"))
        train_loop.start()
        timer.start()
        while True:
            print_left_and_right_aligned(train_loop.print_info, f"Time: {timer.seconds}")
            time.sleep(0.1)



    
    
    test_loss = 0
    test_acc = 0
    validation_loss = 0
    validation_acc = 0

    test_loss_list = []
    test_acc_list = []
    validation_loss_list = []
    validation_acc_list = []
    # Testing
    model.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            height, width = images.shape[-2], images.shape[-1]
            # Forward pass
            outputs = model(images, height, width)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            # Track the accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            test_acc += correct / total

    # Validation
    model.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(validation_loader):
            images = images.to(device)
            labels = labels.to(device)
            height, width = images.shape[-2], images.shape[-1]
            # Forward pass
            outputs = model(images, height, width)
            loss = criterion(outputs, labels)
            validation_loss += loss.item()
            # Track the accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            validation_acc += correct / total

    # Print the results
    test_loss_list.append(test_loss / len(test_loader))