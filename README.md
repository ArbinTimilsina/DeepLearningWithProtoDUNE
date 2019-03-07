#### Semantic segmentation (pixel-wise classification) network to perform cosmic ray and beam particle separation in prototype DUNE detector.


![alt text](plots/History.gif "Training")


## Instructions

### Git clone the code
```
git clone https://github.com/ArbinTimilsina/DeepLearningWithProtoDUNE.git
cd DeepLearningWithProtoDUNE
```

### Download the dataset
```
# 320x320 inputs; min 2000 beam hits 
wget -O input_files.zip https://www.dropbox.com/sh/a00fbuye3i1c0sj/AACeI2l-iEpIoeDbDBtogjJKa?dl=1
unzip input_files.zip -d input_files
rm -rf input_files.zip
```

### Train the model
```
python train_model.py --help

# Example
python train_model.py -o Development -e 5
```
Details can be found in the configuration file.


### Analyze the model
```
python analyze_model.py --help

# Example
python analyze_model.py -p 5 -s Development
```


### Additional information
### To create a conda environment (Python 3)
```
conda create --name envDeepLearningWithProtoDUNE python=3.5
conda activate envDeepLearningWithProtoDUNE
pip install --upgrade pip
pip install -r requirements/cpu_requirements.txt
conda install pydot graphviz
```

### To run with singularity container
```
singularity pull --name DeepLearningWithProtoDUNE.img shub://ArbinTimilsina/Base-Singularity:deeplearningwithprotodune

# If using GPUs, don't forget --nv option
singularity exec --nv DeepLearningWithProtoDUNE.img python train_model.py -o Development -e 5
```

### To switch Keras backend to TensorFlow
```
KERAS_BACKEND=tensorflow python -c "from keras import backend"
```

### To calculate the weights
```
python calculate_weights.py
```
It will run over the default traning files in the configuration. Median for each class will be displayed in plots/weights_median.pdf.

### To make plots of events
```
# For 10 events
python plot_events.py --events 10
```

### To open jupyter notebook
#### Create an IPython kernel for the environment
```
# Create an IPython kernel for the environment
python -m ipykernel install --user --name envDeepLearningWithProtoDUNE --display-name "envDeepLearningWithProtoDUNE"
```

```
# Open the notebook
jupyter notebook miscellaneous/model_creation_playground.ipynb

# Note: Make sure to change the kernel to envDeepLearningWithProtoDUNE using the drop-down menu (Kernel > Change kernel > envDeepLearningWithProtoDUNE)
```
