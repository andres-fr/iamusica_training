# iamusica_training

The present repository hosts the software needed to train and evaluate a Deep Learning piano onset+velocity detection model, developed in the context of the [IAMúsica](https://joantrave.net/en/iamusica/) project. Specifically, it provides the means to:
* Install the required software dependencies
* Download and preprocess the required dataset
* Run and evaluate (pre)trained models
* Train models from scratch

See [this companion repository](https://github.com/andres-fr/iamusica_demo) for a real-time, graphical software demonstration.

<img src="assets/iamusica_logo.jpg" alt="IAMúsica logo" width="41.5%"/> <img src="assets/ieb_logo.jpg" alt="IEB logo" width="54%"/>

*IAMúsica was supported by research grant [389062, INV-23/2021](http://www.iebalearics.org/media/files/2022/02/10/resolucio-definitiva-inv-boib-2021-cat.pdf) from the [Institut d'Estudis Baleàrics](http://www.iebalearics.org/ca/), and is composed by:*
* [Eulàlia Febrer Coll](https://www.researchgate.net/profile/Eulalia-Febrer-Coll)
* [Joan Lluís Travé Pla](https://joantrave.net/en)
* [Andrés Fernández Rodríguez](https://aferro.dynu.net)

This is [Free/Libre and Open Source Software](https://www.gnu.org/philosophy/floss-and-foss.en.html), see the [LICENSE](LICENSE) for more details.





---

# Software dependencies

We use `PyTorch`. The following instructions should allow to create a working environment from scratch, with all required dependencies (tested on `Ubuntu 20.04` with `conda 4.13.0`):

```
# create and activate conda venv
conda create -n iamusica_ml python==3.9
conda activate iamusica_ml

# conda dependencies
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 -c pytorch
conda install pandas==1.4.2 -c anaconda
conda install omegaconf==2.1.2 -c conda-forge
conda install matplotlib==3.4.3 -c conda-forge
conda install h5py==3.6.0 -c anaconda

# pip dependencies
pip install coloredlogs==15.0.1
pip install mido==1.2.10
pip install mir-eval==0.7
pip install parse==1.19.0
```







---

# Data downloading

For this project, training and evaluation is done using the [MAESTRO](https://magenta.tensorflow.org/datasets/maestro) dataset. Specifically, we focus on the latest version, `MAESTROv3`. The full dataset can be readily downloaded at the provided link, and the file structure is expected to end up looking like this:

```
MAESTROv3 ROOT PATH
├── LICENSE
├── maestro-v3.0.0.csv
├── maestro-v3.0.0.json
├── README
├── 2004
├── 2006
├── 2008
├── 2009
├── 2011
├── 2013
├── 2014
├── 2015
├── 2017
└── 2018
```

Where each of the `20xx` directories contains `wav` files with their corresponding `midi` annotations, making a total of 2552 files.

### Downloading other supported datasets:

To ensure compatibility with prior literature, this repository also provides functionality for `MAESTROv1` and `MAESTROv2` (the procedure for those is analogous to v3).

Furthermore, it also provides all functionality needed to use the [MAPS](https://hal.inria.fr/inria-00544155/document) dataset. To download it,

1. Request user and password here: https://adasp.telecom-paris.fr/resources/2010-07-08-maps-database/
2. Download e.g. via: `wget -r --ask-password --user="<YOUR EMAIL>" ftp://ftps.tsi.telecom-paristech.fr/share/maps/`
3. Merge partial zips into folders containing wavs, midis and txt files

For MAPS, the result should end up looking like this (9 folders with 11445 files each):

```
MAPS ROOT PATH
├── license.txt
├── MAPS_doc.pdf
├── MD5SUM
├── readme.txt
├── AkPnBcht
|   ├── ISOL
|   ├── MUS
│   ├── RAND
│   └── UCHO
├── AkPnBsdf
│   ├── ISOL ...
│   ├── MUS  ...
│   ├── RAND ...
│   └── UCHO ...
...
```







---

# Data preprocessing

To train the model, we represent the audio as log-mel spectrograms and the annotations as piano rolls. To speed up training and avoid redundant computations, we preprocess the full datasets ahead of time into [HDF5](https://www.h5py.org/) files.

If we store all datasets inside of a `datasets` folder in this repo, preprocessing `MAESTROv3` with the default parameters can be done by simply calling the following script:

```
python 0a_maestro_to_hdf5mel.py
```

Processing `MAESTROv3` with the default settings takes about 30min on a 16-core CPU. The piano roll HDF5 file takes about 0.5GB of space, and the log-mel file about 22.5GB.

> :warning: **onset/offset collision**:
> Note that creating piano rolls from MIDI requires to time-quantize the events. If the time resolution is too low, it could happen that two events for the same note end up in the same "bin", and therefore ignored. Another possible explanation is that the MIDI file includes redundant/inconsistent messages, which are also ignored.
> During the preprocessing of MAESTRO/MAPS we can expect quite a few of those to happen, most likely due to the latter reason. We can ignore them, since we don't use piano rolls for evaluation.


### Preprocessing other supported datasets:

To precompute former maestro versions (assuming they are inside `datasets`):

```
python 0a_maestro_to_hdf5mel.py MAESTRO_VERSION=1 MAESTRO_INPATH=datasets/maestro/maestro-v1.0.0
python 0a_maestro_to_hdf5mel.py MAESTRO_VERSION=2 MAESTRO_INPATH=datasets/maestro/maestro-v2.0.0
```

To precompute MAPS with default parameters (assuming it is inside `datasets`):

```
python 0b_maps_to_hdf5mel.py
```

Processing `MAPS` with the default settings takes about 20min on a 16-core CPU. The piano roll HDF5 file takes about 100MB of space, and the log-mel file about 4GB.








---

# Running/evaluating the model

This repository also hosts an instance of a [pretrained model](assets/OnsetVelocityNet_2022_09_08_01_27_40.139step=95000_f1=0.9640.torch) (25.5MB), trained on MAESTROv3 for 95000 steps with the default settings and no augmentation. The evaluation script can be run on the pretrained model with default parameters as follows:



```
python 2_eval_onsets_velocities.py SNAPSHOT_INPATH=assets/OnsetVelocityNet_2022_09_08_01_27_40.139step=95000_f1=0.9640.torch
```

Yielding the following results (96.43% for onset detection, 92.83% for onset+velocity):


```
ONSETS:
                                              Filename         P         R        F1
0    MIDI-Unprocessed_11_R1_2009_06-09_ORIG_MID--AU...  0.992867  0.965326  0.978903
1    MIDI-Unprocessed_02_R1_2009_03-06_ORIG_MID--AU...  0.992147  0.968072  0.979961
2    MIDI-Unprocessed_01_R1_2006_01-09_ORIG_MID--AU...  0.980645  0.916667  0.947577
3    MIDI-Unprocessed_24_R1_2006_01-05_ORIG_MID--AU...  0.972730  0.892212  0.930733
4    MIDI-Unprocessed_11_R1_2009_06-09_ORIG_MID--AU...  0.976380  0.906158  0.939959
..                                                 ...       ...       ...       ...
173  MIDI-Unprocessed_052_PIANO052_MID--AUDIO-split...  0.999290  0.995757  0.997520
174  ORIG-MIDI_03_7_6_13_Group__MID--AUDIO_09_R1_20...  0.994460  0.979803  0.987077
175  MIDI-Unprocessed_XP_14_R1_2004_04_ORIG_MID--AU...  0.994482  0.981330  0.987862
176  MIDI-Unprocessed_XP_14_R1_2004_04_ORIG_MID--AU...  0.977350  0.986286  0.981797
177                         AVERAGES (t=0.77, s=-0.01)  0.983384  0.946456  0.964281

[178 rows x 4 columns]


ONSETS+VELOCITIES:
                                              Filename         P         R        F1
0    MIDI-Unprocessed_11_R1_2009_06-09_ORIG_MID--AU...  0.984308  0.957004  0.970464
1    MIDI-Unprocessed_02_R1_2009_03-06_ORIG_MID--AU...  0.942408  0.919540  0.930834
2    MIDI-Unprocessed_01_R1_2006_01-09_ORIG_MID--AU...  0.941056  0.879660  0.909323
3    MIDI-Unprocessed_24_R1_2006_01-05_ORIG_MID--AU...  0.920157  0.843992  0.880430
4    MIDI-Unprocessed_11_R1_2009_06-09_ORIG_MID--AU...  0.924098  0.857635  0.889627
..                                                 ...       ...       ...       ...
173  MIDI-Unprocessed_052_PIANO052_MID--AUDIO-split...  0.992193  0.988685  0.990436
174  ORIG-MIDI_03_7_6_13_Group__MID--AUDIO_09_R1_20...  0.968975  0.954694  0.961782
175  MIDI-Unprocessed_XP_14_R1_2004_04_ORIG_MID--AU...  0.957824  0.945158  0.951449
176  MIDI-Unprocessed_XP_14_R1_2004_04_ORIG_MID--AU...  0.916195  0.924571  0.920364
177                         AVERAGES (t=0.77, s=-0.01)  0.946503  0.911335  0.928316

[178 rows x 4 columns]
```


### Fast learning and inference

In contrast with other state-of-the-art models from the literature that surpass an F1-score of 96% for note onset detection (e.g. [Kong et al.](https://arxiv.org/abs/2010.01815), [Hawthorne et al.](https://arxiv.org/abs/2107.09142)), this model is **fully convolutional, has less parameters and learns faster**). Furthermore, our time resolution for the spectrograms is 24ms (in contrast with the 10ms from the above cited sources), and we make use of a very simple but effective multi-task supervision.

To illustrate learning speed, when trained with batches of 30 5-second chunks on MAESTROv3 (19119 batches per epoch), the model processes 1000 batches every 75 minutes on a RTX3070-8GB-laptop GPU. The training progress is illustrated in the table below:


| Training step | Onset F1 (MAESTROv3) | Onset+Velocity F1 (MAESTROv3) |
|:-------------:|:--------------------:|:-----------------------------:|
|      500      |        89.37%        |             79.63%            |
|     1000      |        91.90%        |             83.61%            |
|     2500      |        93.63%        |             86.67%            |
|     6000      |        94.71%        |             88.80%            |
|    95000      |        96.43%        |             92.83%            |


Among other advantages, this allows for precise real-time detection on commodity hardware. Check the companion repository provided above for a real-time, graphical demonstration.




---

# Training the model

For adequate training, a GPU with at least 8GB of memory is sufficient. The following command trains a model from scratch on `MAESTROv3`:

```
python 1_train_onsets_velocities.py
```

Running the script without parameters on a CUDA-enabled system results in the following configuration:

```
CONFIGURATION:
DEVICE: cuda
MAESTRO_PATH: datasets/maestro/maestro-v3.0.0
MAESTRO_VERSION: 3
HDF5_MEL_PATH: datasets/MAESTROv3_logmel_sr=16000_stft=2048w384h_mel=250(50-8000).h5
HDF5_ROLL_PATH: datasets/MAESTROv3_roll_quant=0.024_midivals=128_extendsus=True.h5
SNAPSHOT_INPATH: null
OUTPUT_DIR: out
TRAIN_BS: 30
TRAIN_BATCH_SECS: 5.0
DATALOADER_WORKERS: 8
LR_MAX: 6.0
LR_MIN: 1.0e-05
LR_PERIOD: 1500
LR_DECAY: 0.95
LR_SLOWDOWN: 1.0
MOMENTUM: 0.85
WEIGHT_DECAY: 0.0
BATCH_NORM: 0.85
DROPOUT: 0.15
LEAKY_RELU_SLOPE: 0.1
ONSET_POSITIVES_WEIGHT: 8.0
VEL_LOSS_LAMBDA: 10.0
TRAINABLE_ONSETS: true
NUM_EPOCHS: 30
TRAIN_LOG_EVERY: 10
XV_EVERY: 500
XV_CHUNK_SIZE: 600.0
XV_CHUNK_OVERLAP: 2.5
XV_THRESHOLDS:
- 0.6
- 0.675
- 0.7
- 0.725
- 0.75
- 0.775
- 0.8
```

The script preiodically cross-validates the model every `XV_EVERY` batches, and saves the crossvalidated model snapshots under `OUTPUT_DIR`, for further usage and evaluation.



### Inspecting during training

This repo also provides the possibility to pause the training script at arbitrary points, articulated through the [breakpoint.json](breakpoint.json) file, expected to be in the following JSON format:

```
{"inconditional": false,
 "step_gt": null,
 "step_every": null}
```

At every training step, after the loss is computed and before the backward pass and optimization step, the training script checks the contents of the JSON file:

* If `inconditional` is set to `true`, a `breakpoint()` will be called (otherwise ignore)
* If `step_gt` is an integer, `breakpoint()` if the current step is greater than the given integer (otherwise ignore).
* If the contents can't be understood, the file is ignored and training progresses

Note that the default is simply to ignore this file, and to stop the training, the user can e.g. open the file, set `inconditional` to `true`, and save. Then, the training script pauses and the state can be inspected. To resume training, set the value to `false`, save, and press `c` to continue with the process, as explained [here](https://docs.python.org/3/library/pdb.html).
