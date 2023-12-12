import time
import threading
import click
import shutil


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

def print_right_aligned(text):
  terminal_width = shutil.get_terminal_size().columns
  text_width = len(text)
  remaining_space = terminal_width - text_width
  click.echo(" " * remaining_space, nl=False)
  click.echo(text, nl=False)
  click.echo("\r", nl=False)

def print_left_aligned(text):
  'wıth flush'
  click.echo(text, nl=False)
  click.echo("\r", nl=False)


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


class MainLoop(threading.Thread):
  def __init__(self):
    threading.Thread.__init__(self)
    self.daemon = True

  def main(self):
    while True:
      print_left_aligned("Hello")
      time.sleep(2)

  def run(self):
    self.main()

    
if __name__ == "__main__":
  main = MainLoop()
  t = TimerThread()
  main.start()
  t.start()


  while True:
    print_right_aligned(str(t.seconds))


  t.join()
  main.join()
  
