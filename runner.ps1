$epoch = 10
$batch_size = 32
$lr = 0.001


# Set the path to your virtual environment
$venvPath = "venv\Scripts\Activate.ps1"

# Set the path to your Python script
$scriptPath = "src\main.py"

# Activate the virtual environment
& $venvPath

# Example ForEach loop with an array
$model_names = @("EfficientNetV2M", "ResNet50", "DenseNet201", "MobileNetV3Large")

# train loop 
foreach ($model_name in $model_names) {
  for ($i = 1; $i -le 2; $i++) {

    if ($i -eq 1) {
      Write-Host "Training $model_name with augmentation"
      python $scriptPath --train --device cuda:0 --augment --model $model_name --epoch $epoch --batch_size $batch_size --learning_rate 0.00001
    }
    else {
      Write-Host "Training $model_name without augmentation"
      python $scriptPath --train --device cuda:0 --model $model_name --epoch $epoch --batch_size $batch_size --learning_rate 0.00001
    }
  }
  
}


# Deactivate the virtual environment (optional, comment out if not needed)
deactivate
