import shutil
import time
import click
import threading
import matplotlib.pyplot as plt


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
  plt.figure(figsize=(10, 10))
  plt.subplot(2, 1, 1)
  plt.plot(train_loss_list, label="Train loss")
  plt.plot(validation_loss_list, label="Validation loss")
  plt.legend()
  plt.subplot(2, 1, 2)
  plt.plot(train_acc_list, label="Train accuracy")
  plt.plot(validation_acc_list, label="Validation accuracy")
  plt.legend()
  plt.show()


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
