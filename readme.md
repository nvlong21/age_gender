# Age Gender Prediction				

This repository is used for finding and predicting from an image one or multiple human's gender and age(confidence scores provided for both age and gender). **97% acc** for gender and **MAE of 4.7** for age.


## Requirements

- python3, **pytorch**
- `pip3 install --upgrade opencv-python, imutils, skimage`
- using retinaface 
## Usage

#### **Training**

1. Put your image in pics/ (see config.ini for naming details)
2. Run preprocessing steps in `preprocess.py`
3. Run `train.py`


## Train/Test Pipeline

![Example](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/img/pipeline.png)

#### **Train**

1. Using **cleaned**(https://drive.google.com/file/d/1Bd7o1_rlpG-NTk9ny4jkxilDJitq1Rl9/view?usp=sharing)[1] for training .
2. Using [FG-NET dataset](http://www-prima.inrialpes.fr/FGnet/html/benchmarks.html)[3] for testing.
3. Train a, 
   - the output is 2 neuron represents probs of male&female plus 100 neurons represents probs of being age 0-99.
   - auto detect if use GPU or even multiple GPUs for training.
   - auto reduce learning rate when we have no loss reduce on val dataset for >N epochs.
   - auto freeze CNN layers and train only last FCN layers when first epoch.
   - auto load and save weights, log training loss and metadatas after each epoch.
   - more detains can be found on src file `train.py` and configuration file `config.ini`

#### **Test**

1. detect and align faces using `retinaface`.
2. predict age, gender and confidence scores(probability of each gender and variance of age).





















