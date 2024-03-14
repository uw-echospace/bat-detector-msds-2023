
# MSDS Capstone Project: Bats!!!

## Overview
This repository offers the following features:
* Detection of bat search, social, and feedbuzz calls
* A fast, customizable pipeline for automating application of the aforementioned detectors

Bats are crucial in gauging ecosystem health due to their sensitivity to habitat alterations and climate change. The Echospace@UW team has gathered bat echolocation recordings from the Union Bay Conservatory area, and for monitoring bat activity, the team needs to generate statistics on bat call frequencies and feeding buzz occurrences. These would enable researchers to look into the overall health of the surrounding ecosystem. Since the process of bat call detection is labor-intensive, inefficient, and prone to errors, the Echospace team has tried to adopt a more streamlined cloud-based pipeline. This pipeline incorporates a deep convolutional network, empowering researchers to identify bat calls and feeding buzzes with enhanced efficiency accurately.

This repository offers the following features:
- A fine-tuned bat call detection deep learning model.
- A fast customizable visualization tool pipeline that can run the model on 5-10s audio to detect bat calls and compare them to the ground truth.

## Usage

### Setup

We recommend using Python 3.9.x. Other versions may work, but they're not tested. We also recommend creating a Python virtual environment.

From repository root, run the following commands:
```bash
pip install -r requirements.txt
```

### Fine-tuning

To fine-tune the model, first enter the folder containing the fine-tune module:
```bash
cd src/models/bat_call_detector/batdetect2/bat_detect/finetune
```

#### Split train and test sets
```bash
DATA_DIR='../../../../../../../data/'
python prep_data_finetune.py dataset_name $DATA_DIR/audio/  $DATA_DIR/anns/ $DATA_DIR/output_anns/ --rand_seed 123
```

#### Fine-tune
```bash

MODEL_PATH='../../models/Net2DFast_UK_same.pth.tar'
python finetune_model.py $DATA_DIR/audio/ \
                         $DATA_DIR/output_anns/dataset_name_TRAIN.json \
                         $DATA_DIR/output_anns/dataset_name_TEST.json \
                         $MODEL_PATH
```

### Visualization tool

To run the visualization tool, use the following commands:
```bash
sudo su
source venv/bin/activate
cd /home/exouser/finetuning_hm/bat-detector-msds-2023/src/models/bat_call_detector/batdetect2/
```

Change the MODEL_PATH in the code (demo.py)

```bash
streamlit run demo.py
```

- Use the “External URL” to view the webapp. 
- On changing the model you can use the rerun on the app to reload it. 
- Select Anns, Audio File, Start Time, Duration, and Detection Threshold. 
- Use “Run Model” to run the model on the given audio file for the given duration from the start. 
- Visualize the bat calls and detections made.

## Running the Pipeline

1. Obtain annotated audio files in .txt format
   - Echospace @ UW provided some manually annotated bat call detection files in .txt format corresponding to their audio files. Each audio file was about 30 minutes long.

2. Clean and preprocess annotated files to JSON
   - The original annotated files include many invalid and duplicated detections, so it is necessary to clean them before using them; they’re also converted into JSON format to be consistent with the batdetect2 preprocessing steps.

3. Migrate repository to JetStream2 for GPU computation
   - Due to the high computation resources needed to retrain the model, we migrated the working repository to the GPU instances on JetStream2, a cloud environment for ACCESS.

4. Split train and test sets
   - batdetect2 provides the functions to split train and test sets for the fine-tuning process. We chose to split the data to 80% for training and 20% for testing purposes. Between a total of 11 audio files, we used 8 for training and 3 for testing.

5. Fine-tune the model
   - We adjusted the hyperparameters for our input data for each run, and have displayed the results of three successful ones.

6. Visualization of the model performance
   - We developed a web app using Streamlit, we can run the model on 5-10s audio chunks to see how the model performs when detecting bat calls compared to the ground truth labels.

## Evaluation

The test dataset comprised 30-minute-long audio files, which were broken down into smaller files for faster processing. The evaluation assessed enhancements in detection accuracy to those of the previous version. The metrics used were precision and recall around the bounding boxes of the spectrographs.

Feeding Buzz detections as observed in the web app:

<img width="1470" alt="Feeding Buzz detections as observed in the web app" src="https://github.com/uw-echospace/bat-detector-msds-2023/assets/72214535/02d2f33f-1d61-4a6b-9cb2-b097b84bb057">


Since the original batdetect2 model only used bat call data from Eastern Europe, Brazil, and Australia, it misses bat calls especially for low-frequency (25-35 kHz) calls common to the North American bat population. However, based on Figure 1 (a screenshot from the visualization webapp), we can see that our model was able to pick up on many lower frequency calls as well.

Test loss (originally it was around 5274k) throughout different models:

<img width="490" alt="Test loss throughout different models" src="https://github.com/uw-echospace/bat-detector-msds-2023/assets/72214535/55f9cc06-e124-455c-8b41-fb4b381f174c">

The original model demonstrated a very high loss on the test files and the three models we made after finetuning reduced it (Figure 2). The original model also scored very low on the precision and recall scores and after finetuning the metrics had improved by over 20% (Figure 3) for each of them.

Performance improvements in precision and recall throughout different models:

<img width="492" alt="Performance improvements in precision and recall throughout different models" src="https://github.com/uw-echospace/bat-detector-msds-2023/assets/72214535/9fcad851-f1dd-4829-94e9-e063119266ad">


## Model-by-model Performance

### Model #1

- Layers Trained: Final
- JSON annotations used: Halved
- Loss used: Focal
- Overlap: 0.75
- Epochs: 60
- Steps: 100
- Learning rate: 0.01
- Test Loss: 0.1527
- Train Loss: 0.2505
- Rec at 0.95 (det): 0.0242
- Avg prec (cls): 0.2405
- Tar file: 2024_02_28__01_04_10.pth.tar

### Model #2

- Layers Trained: All
- JSON annotations used: 3 files (no split)
- Loss used: default
- Overlap: 0.5
- Epochs: 100
- Steps: 1000
- 'target_samp_rate' = 124000
- 'fft_win_length' = 512 / 124000.0
- Learning rate: default
- Test Loss: 3.407
- Train Loss: 0.2183
- Rec at 0.95 (det): 0.1364
- Avg prec (cls): 0.2946
- Tar file: 2024_02_12__08_00_59.pth.tar

### Model #3

- Layers Trained: All
- JSON annotations used: 3 files (not halved)
- Loss used: default
- Overlap: 0.5
- Epochs: 200
- Steps: 100
- 'target_samp_rate' = 124000
- 'fft_win_length' = 512 / 124000.0
- Learning rate: default
- Test Loss: 1.5796
- Train Loss: 0.4763
- Rec at 0.95 (det): 0.1136

## Future Work

One of the major differences in the original paper and the EchoSpace data is completely different sampling rates. Since the audio segmentation to image generation is an error prone process, having this difference meant that the model essentially saw a lower quality image than what would have been possible with a custom sampling rate. This resulted in the model creating a good range of predictions which have a low detection probability. One improvement would be to make this customizable for each file and allow the model to work with better quality images.


## Acknowledgments
We would like to thank the UW EchoSpace team who were great to work with and helped us make this project successful: Dr Wu-Jung Lee, Aditya Krishna, Caesar Tuguinay, and Liuyixin Shao. We would also like to thank the batdetect2 team (Mac Aodha, Oisin and  Martinez Balvanera, Santiago and  Damstra, Elise and  Cooke, Martyn and  Eichinski, Philip and  Browning, Ella and  Barataudm, Michel and  Boughey, Katherine and  Coles, Roger and  Giacomini, Giada and MacSwiney G., M. Cristina and  K. Obrist, Martin and Parsons, Stuart and  Sattler, Thomas and  Jones, Kate E.), whose work was instrumental in this field and the base model for our research.

