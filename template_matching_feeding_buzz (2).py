
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Template matching experiments

"""

#%% Import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from maad import sound, util, rois
from maad.rois import template_matching
from pathlib import Path
import pickle

#%% Set constants
TEMPLATE_AUDIO_1 = Path('/Users/ernestocediel/OneDrive - Universidad de los Andes/MSDS/DATA 590 Capstone I & II/ravenpro_test/buzz/20210910_030000_time2303_LFbuzz.wav')
TEMPLATE_AUDIO_2 = Path('/Users/ernestocediel/OneDrive - Universidad de los Andes/MSDS/DATA 590 Capstone I & II/ravenpro_test/20210910_033000.WAV')
TEMPLATE_AUDIO_3 = Path('/Users/ernestocediel/OneDrive - Universidad de los Andes/MSDS/DATA 590 Capstone I & II/ravenpro_test/20210910_030000.WAV')
TEMPLATE_PATH = Path('/Users/ernestocediel/OneDrive - Universidad de los Andes/MSDS/DATA 590 Capstone I & II/ravenpro_test/templates')
#PATH_AUDIO = Path('/Users/kirsteenng/Desktop/UW/DATA 590/sample_wav/20210910_030000_time2303_LFbuzz.wav')
#PATH_AUDIO = '/Users/kirsteenng/Desktop/UW/DATA 590/sample_wav/20210921_030000_time1054_LFbuzz_varyFreq.wav'
#PATH_AUDIO = '/Users/kirsteenng/Desktop/UW/DATA 590/sample_wav/20210910_030000_time2303_LFbuzz.wav'
#PATH_AUDIO = Path('/Users/kirsteenng/Desktop/UW/DATA 590/workflow/2_clipped_audio_wav/20210910_033000/20210910_033000__0.00_1500.00.wav')
PATH_AUDIO = Path('/Users/ernestocediel/OneDrive - Universidad de los Andes/MSDS/DATA 590 Capstone I & II/ravenpro_test/20221012_030000_1min.WAV')
RESULTS_DIR = Path('/Users/ernestocediel/OneDrive - Universidad de los Andes/MSDS/DATA 590 Capstone I & II/ravenpro_test/results')
# Set spectrogram parameters
tlims = { 'template_1':(9.762, 10.059),
          'template_2_1':(70.637, 71.328),
          'template_2_2':(620.663, 620.854),
          'template_2_3':(898.079, 898.368),
          'template_2_4':(1189.957, 1190.130),
          'template_3_1':(608.139,	608.452),
          'template_3_2':(744.961, 745.0877),
          'template_3_3':(1065.034, 1065.228),
          'template_3_4':(1098.204, 1098.519)

        }
flims = { 'template_1':(14532.7, 29760.3),
          'template_2_1':(19745, 28638.2),
          'template_2_2':(12434.9,29910.9),
          'template_2_3':(11426.6, 25205.9),
          'template_2_4':(12434.9, 23021.3),
          'template_3_1':(14328.0,30138.3),
          'template_3_2':(10375.5, 47430.83),
          'template_3_3':(14328, 25691.7),
          'template_3_4':(17272, 34565.5)

        }
nperseg = 1024
noverlap = 512
window = 'hann'
db_range = 80

# %%
def generate_template(template_audio_path:Path, template_path:Path, type:str, flims:tuple, tlims:tuple):
    # we want to create template, save it and update the template dictionary
    template_name = 'template_{}_{}_{}_{}'.format(type,template_audio_path.stem, tlims[0], tlims[1])
    s_template, fs_template = sound.load(template_audio_path)
    Sxx_template, _, _, _ = sound.spectrogram(s_template, fs_template, window, nperseg, noverlap, flims, tlims)
    if template_name not in template_dict:
        # we update the dictionary
        template_dict[template_name] = Sxx_template    
    # we save the template
    with open(template_path / template_name, 'wb') as handle:
        pickle.dump(Sxx_template, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return 

# %%
def load_templates(template_path:Path):
    #Ideally we should be able to choose which templates to load, for now we will load all
    try:
        with open(template_path / 'template_dict.pickle', 'wb') as handle:
            template_dict = pickle.load(handle)
    except:
        # if it's the first time creating template dict
        template_dict = dict()
# %%
def save_template_dict(template_path: Path):
        # we save the updated dict
    with open(template_path / 'template_dict.pickle', 'wb') as handle:
        pickle.dump(template_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
#%%
def template_matching_loop(PATH_AUDIO:Path, result_dir:Path, template:np.ndarray,flims:tuple, peak_th:float, db_range:int = 80,):
    suffix = PATH_AUDIO.stem
    s, fs = sound.load(PATH_AUDIO)
    #tlims_target = (5.413,6.083)
    flims_target = (16940,49000)

    #peak_th = 0.2
    peak_distance = 0.05

    # Compute spectrogram for target audio
    Sxx_audio, tn, fn, ext = sound.spectrogram(s, fs, window, nperseg, noverlap, flims)
    Sxx_audio = util.power2dB(Sxx_audio, db_range)
    xcorrcoef, rois = template_matching(Sxx_audio, template, tn, ext, peak_th, peak_distance)
    rois['min_f'] = flims[0]
    rois['max_f'] = flims[1]
    print(rois)
    rois.to_csv(Path('/Users/kirsteenng/Desktop/UW/DATA 590/workflow/4_feeding_buzz_detector/feeding_buzz_csv')/"{}.csv".format(suffix))

    # plot
    Sxx, tn, fn, ext = sound.spectrogram(s, fs, window, nperseg, noverlap)
    fig, ax = plt.subplots(2,1, figsize=(8, 5), sharex=True)
    util.plot_spectrogram(Sxx, ext, db_range=80, ax=ax[0], colorbar=False)
    
    util.overlay_rois(Sxx, util.format_features(rois, tn, fn), fig=fig, ax=ax[0])
    ax[1].plot(tn[0: xcorrcoef.shape[0]], xcorrcoef)
    ax[1].hlines(peak_th, 0, tn[-1], linestyle='dotted', color='0.75')
    ax[1].plot(rois.peak_time, rois.xcorrcoef, 'x')
    ax[1].set_xlabel('Time [s]')
    ax[1].set_ylabel('Correlation coeficient')
    fig.savefig(result_dir/"bat_calls_detected_{}.png".format(suffix),format="png")
    plt.show()
    plt.close()
    return

# %% Loop for creating feeding buzz detection
audio_dir = Path('/Users/kirsteenng/Desktop/UW/DATA 590/workflow/2_clipped_audio_wav')
audio_list = [f for f in audio_dir.glob('*.wav')]
for audio in audio_list:
    template_matching_loop(audio, Path('/Users/kirsteenng/Desktop/UW/DATA 590/workflow/4_feeding_buzz_detector/correlation_image'))


# %% 
# TODO: run each template against 20210921_033000 with 0.2 threshold and check coefficient results


# %% ################# Multiple template matching experiment begins #############################
global template_dict
load_templates(TEMPLATE_PATH)

generate_template(TEMPLATE_AUDIO_1, TEMPLATE_PATH, 'hf', tlims = tlims['template_1'], flims = flims['template_1'])
generate_template(TEMPLATE_AUDIO_2,TEMPLATE_PATH, 'hf',tlims = tlims['template_2_1'], flims = flims['template_2_1'])
#generate_template(TEMPLATE_AUDIO_2, 'hf',tlims = tlims['template_2_2'], flims = flims['template_2_2'])
#generate_template(TEMPLATE_AUDIO_2, 'hf',tlims = tlims['template_2_3'], flims = flims['template_2_3'])
#generate_template(TEMPLATE_AUDIO_2, 'hf',tlims = tlims['template_2_4'], flims = flims['template_2_4'])

#generate_template(TEMPLATE_AUDIO_3, 'hf',tlims = tlims['template_3_1'], flims = flims['template_3_1'])
#generate_template(TEMPLATE_AUDIO_3, 'hf',tlims = tlims['template_3_2'], flims = flims['template_3_2'])
#generate_template(TEMPLATE_AUDIO_3, 'hf',tlims = tlims['template_3_3'], flims = flims['template_3_3'])
#generate_template(TEMPLATE_AUDIO_3, 'hf',tlims = tlims['template_3_4'], flims = flims['template_3_4'])
save_template_dict(TEMPLATE_PATH)

# 1-minute file to detect


# %% Run over all templates
def run_template_matching(PATH_AUDIO: Path, RESULTS_DIR: Path, Sxx_audio: np.ndarray, template: np.ndarray, tn: any, ext: any, peak_th: float, peak_distance: float):
    suffix = PATH_AUDIO.stem
    xcorrcoef, rois = template_matching(Sxx_audio, template, tn, ext, peak_th, peak_distance)
    if not rois.empty:
        rois['min_f'] = flims[0]
        rois['max_f'] = flims[1]
        print(rois)
        rois.to_csv(Path(RESULTS_DIR,'feeding_buzz_csv',"{}.csv".format(suffix)))

    # plot
    Sxx, tn, fn, ext = sound.spectrogram(s, fs, window, nperseg, noverlap)
    fig, ax = plt.subplots(2,1, figsize=(8, 5), sharex=True)
    util.plot_spectrogram(Sxx, ext, db_range=80, ax=ax[0], colorbar=False)
    
    util.overlay_rois(Sxx, util.format_features(rois, tn, fn), fig=fig, ax=ax[0])
    ax[1].plot(tn[0: xcorrcoef.shape[0]], xcorrcoef)
    ax[1].hlines(peak_th, 0, tn[-1], linestyle='dotted', color='0.75')
    ax[1].plot(rois.peak_time, rois.xcorrcoef, 'x')
    ax[1].set_xlabel('Time [s]')
    ax[1].set_ylabel('Correlation coeficient')
    fig.savefig(RESULTS_DIR/"bat_calls_detected_{}.png".format(suffix),format="png")
    plt.show()
    plt.close()

def run_multiple_template_matching(PATH_AUDIO: Path, peak_th: float, peak_distance: float):
    s, fs = sound.load(PATH_AUDIO)
    # Compute spectrogram for target audio
    Sxx_audio, tn, fn, ext = sound.spectrogram(s, fs, window, nperseg, noverlap)
    Sxx_audio = util.power2dB(Sxx_audio, db_range)
    for template in template_dict.values():
        run_template_matching(PATH_AUDIO,RESULTS_DIR,Sxx_audio,template,tn,ext,peak_th,peak_distance)



#%% Test template
s, fs = sound.load(PATH_AUDIO)
#tlims_target = (5.413,6.083)
flims_target = (16940,49000)

peak_th = 0.25
peak_distance = 0.05

# Compute spectrogram for target audio
Sxx_audio, tn, fn, ext = sound.spectrogram(s, fs, window, nperseg, noverlap)
Sxx_audio = util.power2dB(Sxx_audio, db_range)



#%% Compute template matching
xcorrcoef, aois = rois.template_matching(Sxx_audio, template_2, tn, ext, peak_th, peak_distance)
aois['min_f'] = flims[1][0]
aois['max_f'] = flims[1][1]
print(aois)

#%% Load templates  
# load data
s_template_1, fs_template_1 = sound.load(TEMPLATE_AUDIO_2)
# Compute spectrogram for template signal
Sxx_template, tn_template, fn_template, ext_template = sound.spectrogram(s_template_1, fs_template_1, window, nperseg, noverlap, flims, tlims)

#Sxx_template = util.power2dB(Sxx_template, db_range)
fig, ax = plt.subplots()
util.plot_spectrogram(Sxx_template, ext_template, db_range=80, ax=ax, colorbar=False)
fig.savefig(Path('/Users/kirsteenng/Desktop/UW/DATA 590/workflow')/"template.png")
plt.show()

#%%
# plot
#Sxx, tn, fn, ext = sound.spectrogram(s, fs, window, nperseg, noverlap, flims = (20000,60000))
fig, ax = plt.subplots()
util.plot_spectrogram(Sxx_template, ext, db_range=80, ax=ax, colorbar=False)
# util.overlay_rois(Sxx, util.format_features(rois, tn, fn), fig=fig, ax=ax[0])
# ax[1].plot(tn[0: xcorrcoef.shape[0]], xcorrcoef)
# ax[1].hlines(peak_th, 0, tn[-1], linestyle='dotted', color='0.75')
# ax[1].plot(rois.peak_time, rois.xcorrcoef, 'x')
# ax[1].set_xlabel('Time [s]')
# ax[1].set_ylabel('Correlation coeficient')
result_dir = Path('/Users/kirsteenng/Desktop/UW/DATA 590/workflow')
suffix = PATH_AUDIO.stem
fig.savefig(result_dir/"template.png")

plt.show()
plt.close()

