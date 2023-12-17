import PIL
from PIL import Image
import io
import shutil
import time
import click
import threading
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pygad
import torch


def fitness_func(solution, solution_idx):
    global model
    model_weights = pygad.torchga.model_weights_as_dict(
        model=model, weights_vector=solution)
    model.load_state_dict(model_weights)
    predictions = model(torch.tensor(
        [[1, 1, 1], [2, 2, 2], [3, 3, 3]]).float())
    error = torch.nn.functional.mse_loss(
        predictions, torch.tensor([[1], [2], [3]]).float())
    fitness = 1.0 / error
    return fitness


def print_left_and_right_aligned(left_text, right_text):
  terminal_width = shutil.get_terminal_size().columns
  left_text_width = len(left_text)
  available_width_for_right_text = terminal_width - left_text_width
  truncated_right_text = right_text[:available_width_for_right_text]
  click.echo(left_text, nl=False)
  remaining_space = terminal_width - \
      len(left_text) - len(truncated_right_text)
  click.echo(" " * remaining_space, nl=False)
  click.echo(truncated_right_text, nl=False)
  click.echo("\r", nl=False)
  time.sleep(0.1)


def convert_seconds(seconds):
  hours, remainder = divmod(seconds, 3600)
  minutes, seconds = divmod(remainder, 60)
  return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def plot_loss_and_accuracy(train_loss_list, train_acc_list, validation_loss_list, validation_acc_list):
  """Plot loss and accuracy of training and validation per epoch"""
  plt.figure(figsize=(20, 10))
  plt.subplot(1, 2, 1)
  plt.plot(train_loss_list, label="Training loss")
  plt.plot(validation_loss_list, label="Validation loss")
  plt.legend(frameon=False)
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.title("Loss per epoch")
  plt.subplot(1, 2, 2)
  plt.plot(train_acc_list, label="Training accuracy")
  plt.plot(validation_acc_list, label="Validation accuracy")
  plt.legend(frameon=False)
  plt.xlabel("Epoch")
  plt.ylabel("Accuracy")
  plt.title("Accuracy per epoch")
  plt.tight_layout()
  # return plt as PIL image
  img_buf = io.BytesIO()
  plt.savefig(img_buf, format='png')
  img_buf.seek(0)
  return Image.open(img_buf)



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
