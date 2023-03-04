#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Template matching experiments

"""

#%% Import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import sys
import math

package_path = '/Users/kirsteenng/Desktop/UW/DATA 590/bat-detector-msds/scikit-maad'

if package_path not in set(sys.path):
    sys.path.append(package_path)
    print('Adding path to sys path')

print(sys.path)


from maad import sound, util, rois
from maad.rois import template_matching
# %%
def generate_template(template_audio_path:Path, template_path:Path, template_dict:dict, freq_type:str, tlims:tuple, flims:tuple):
    # we want to create template, save it and update the template dictionary
    template_name = 'template_{}_{}_{}_{}'.format(freq_type,template_audio_path.stem, tlims[0], tlims[1])
    if template_name not in template_dict:
        s_template, fs_template = sound.load(template_audio_path)
        Sxx_template, _, _, _ = sound.spectrogram(s_template, fs_template, window, nperseg, noverlap, flims, tlims)
        # we update the dictionary
        template_dict[template_name] = (Sxx_template, freq_type, flims, tlims)    
        # we save the template
        with open(template_path / template_name, 'wb') as handle:
            pickle.dump(Sxx_template, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.close()

    return template_dict

def load_templates(template_path:Path):
    #Ideally we should be able to choose which templates to load, for now we will load all
    try:
        with open(template_path, 'rb') as handle:
            template_dict = pickle.load(handle)
            print('Template exists! ')
    except:
        # if it's the first time creating template dict
        template_dict = dict()
        print('Template doesnt exist! Returning empty dictionary ')

    return template_dict

def save_template_dict(template_dict:dict, template_path: Path):
        # we save the updated dict
    with open(template_path / 'template_dict.pickle', 'wb') as handle:
        pickle.dump(template_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()
    


# %% Run over all templates
def run_template_matching(PATH_AUDIO: Path, RESULT_DIR: Path, Sxx_audio: np.ndarray,  tn: any, ext: any, fn:any, flims:tuple, template: np.ndarray, template_name:str,peak_th: float, peak_distance: float):
    suffix = PATH_AUDIO.stem
    xcorrcoef, rois = template_matching(Sxx_audio, template, tn, ext, peak_th, peak_distance)
    
    if not rois.empty:
        rois['min_f'] = flims[0]
        rois['max_f'] = flims[1]
        rois['template_name'] = template_name
        print(rois)
        # rois.to_csv(Path(RESULT_DIR/'feeding_buzz_csv'/"{}.csv".format(suffix)))

        # plot
        # fig, ax = plt.subplots(2,1, figsize=(8, 5), sharex=True)
        # util.plot_spectrogram(Sxx_audio, ext, db_range=80, ax=ax[0], colorbar=False)
        # util.overlay_rois(Sxx_audio, util.format_features(rois, tn, fn), fig=fig, ax=ax[0])
        # ax[1].plot(tn[0: xcorrcoef.shape[0]], xcorrcoef)
        # ax[1].hlines(peak_th, 0, tn[-1], linestyle='dotted', color='0.75')
        # ax[1].plot(rois.peak_time, rois.xcorrcoef, 'x')
        # ax[1].set_xlabel('Time [s]')
        # ax[1].set_ylabel('Correlation coeficient')
        # fig.savefig(RESULT_DIR/"bat_calls_detected_{}_{}.png".format(suffix,template_name),format="png")
        # #plt.show()
        # plt.close()

    return rois

def run_multiple_template_matching(PATH_AUDIO: Path, RESULT_DIR:Path,peak_th: float, peak_distance: float, template_dict:dict, num_flim_cuts: int):

    # Load sound and initiate variables
    s, fs = sound.load(PATH_AUDIO)
    rois_df = pd.DataFrame() 

    for index, template in enumerate(template_dict.keys()): #change to enumerate(template_dict.keys) and change index to template_name whwere apropriate
        print('Running against template {}'.format(template))
        curr_template = template_dict[template]
        flims = curr_template[2]
        # We add more cuts in flims, we cut over a total shift of 3kHz (+-1.5kHz)
        for num_cut in range(num_flim_cuts+2):
            total_shift = 3000
            if num_cut == num_flim_cuts+1:
                new_flims = flims
            else:
                shift = -total_shift/2+num_cut*(total_shift/num_flim_cuts)
                new_flims = (math.floor(flims[0]+shift),math.ceil(flims[1]+shift))            
            print(num_cut)
            print("new flims:",new_flims)
            # Compute spectrogram for target audio of the same width as template
            Sxx_audio, tn, fn, ext = sound.spectrogram(s, fs, window, nperseg, noverlap,flims=new_flims)
            #util.plot_spectrogram(Sxx_audio, extent=ext, db_range=60, gain=20, colorbar=False, figsize=(2.5,10))
            #util.plot_spectrogram(curr_template[0], extent=ext, db_range=60, gain=20, colorbar=False, figsize=(2.5,10))
            print(Sxx_audio.shape)
            print(curr_template[0].shape)
            print("target",ext)
            if np.any(np.less(Sxx_audio.shape, curr_template[0].shape)):
                continue
            Sxx_audio = util.power2dB(Sxx_audio, db_range)
            
            
            curr_df = run_template_matching(PATH_AUDIO,RESULT_DIR,Sxx_audio, tn, ext, fn, 
                                            new_flims,
                                            template=curr_template[0], 
                                            template_name=template+str(new_flims), 
                                            peak_th=peak_th,
                                            peak_distance=peak_distance)

            rois_df = pd.concat([rois_df,curr_df], ignore_index=True)
    
    
    print('Matching template loop complete, saving combined dfs to {}.'.format(RESULT_DIR))
    rois_df.to_csv(RESULT_DIR/'feeding_buzz_all_template_{}.csv'.format(PATH_AUDIO.stem))

    return rois_df

#%%
def match_rois(rois: pd.DataFrame, RESULT_DIR: Path, threshold: float, num_matches_threshold: int, buzz_feed_range: float, alpha:float):
    match_dict = dict()
    match_range = alpha*buzz_feed_range/2

    print('Match range: {}'.format(match_range))
    # get a random rois from the df, find all matching rois
    rois_matching = rois.copy()
    while rois_matching.shape[0] > 0:
        # get a random row
        rnd_row = rois_matching.sample()
        # get rand row mid_point
        rnd_row_mid_point = float(rnd_row['peak_time'])
        # find all rows that match this row
        match_rows = rois_matching[rois_matching['peak_time'].between(rnd_row_mid_point-match_range,rnd_row_mid_point+match_range)]

        # TODO ADD THRESHOLD RESTRICTION IF DESIRED 
        # we store matched info in dictionary: (count, tlims, flims, avg.corrcoef)
        match_dict[rnd_row_mid_point] = (match_rows.shape[0], (match_rows.min_t.quantile(0.3), match_rows.max_t.quantile(0.7)), (match_rows.min_f.quantile(0.3), match_rows.max_f.quantile(0.7)), match_rows.xcorrcoef.mean())
        # we remove the matched rows from the DataFrame 
        rois_matching.drop(match_rows.index, inplace=True)

    match_dict_cut = {k: v for k, v in match_dict.items() if v[0] > num_matches_threshold}
    # we sort the dictionary by key (time)
    match_dict_cut = dict(sorted(match_dict_cut.items()))    

    # we convert dict 
    match_df = pd.DataFrame(columns = ['min_t','max_t','min_f','max_f','detection confidence'])
    for i, value in enumerate(match_dict_cut.values()):
        match_df.loc[i] = [value[1][0],value[1][1],value[2][0],value[2][1],value[3]]
    
    print('The followings are the filtered dfs.')
    print(match_df)
    
    return match_df
    
# %%
if __name__ == '__main__':
    #Set constants
    HOME_PATH = Path('/Users/ernestocediel/OneDrive - Universidad de los Andes/MSDS/DATA 590 Capstone I & II/ravenpro_test')
    TEMPLATE_AUDIO_1 = HOME_PATH / 'buzz/20210910_030000_time2303_LFbuzz.wav'
    TEMPLATE_AUDIO_2 = HOME_PATH / '20210910_033000.WAV'
    TEMPLATE_AUDIO_3 = HOME_PATH / '20210910_030000.WAV'
    TEMPLATE_AUDIO_4 = HOME_PATH / '20211016_030000.WAV'

    AUDIO_PATH = HOME_PATH / 'clipped/20210910_030000__0.00_1440.00.wav'
    RESULT_PATH= HOME_PATH / 'results' 
    TEMPLATE_PATH = HOME_PATH/ 'templates'
    TEMPLATE_PICKLE_PATH = HOME_PATH/'templates/template_dict.pickle'

    # Set spectrogram parameters
    tlims = { 'template_1':(9.762, 10.059),
            'template_2_1':(70.637, 71.328),
            'template_2_2':(620.663, 620.854),
            'template_2_3':(898.079, 898.368),
            'template_3_1':(608.139, 608.452),
            'template_3_2':(744.961, 745.0877),
            'template_3_3':(1065.034, 1065.228),
            'template_4_1':(1611.886, 1612.014),
            'template_4_2':(1717.383, 1717.518),
            'template_4_3':(1728.248, 1728.397)
            }
    flims = { 0:(14532.7, 29760.3), #lf
            1:(19745, 28638.2),   #lf
            2:(12434.9,29910.9),  #lf
            3:(11426.6, 25205.9), #lf
            4:(14328.0,30138.3),  #lf
            5:(10375.5, 47430.83),#hf
            6:(14328, 25691.7),   #lf
            7:(19214.9,53801.6),  #hf
            8:(19762.8, 46442.7), #hf
            9:(20751, 52865.6)    #hf
            }
    nperseg = 1024
    noverlap = 512
    window = 'hann'
    db_range = 80

    template_dict = load_templates(template_path=TEMPLATE_PICKLE_PATH)

    if len(template_dict) == 0:

        template_dict = generate_template(TEMPLATE_AUDIO_1,TEMPLATE_PATH, template_dict, 'lf',tlims = tlims['template_1'], flims = flims[0])
        template_dict = generate_template(TEMPLATE_AUDIO_2,TEMPLATE_PATH, template_dict, 'lf',tlims = tlims['template_2_1'], flims = flims[1])
        template_dict = generate_template(TEMPLATE_AUDIO_2,TEMPLATE_PATH, template_dict, 'lf',tlims = tlims['template_2_2'], flims = flims[2])
        template_dict = generate_template(TEMPLATE_AUDIO_2,TEMPLATE_PATH, template_dict, 'lf',tlims = tlims['template_2_3'], flims = flims[3])
        template_dict = generate_template(TEMPLATE_AUDIO_3,TEMPLATE_PATH, template_dict, 'lf',tlims = tlims['template_3_1'], flims = flims[4])
        template_dict = generate_template(TEMPLATE_AUDIO_3,TEMPLATE_PATH, template_dict, 'hf',tlims = tlims['template_3_2'], flims = flims[5])
        template_dict = generate_template(TEMPLATE_AUDIO_3,TEMPLATE_PATH, template_dict, 'lf',tlims = tlims['template_3_3'], flims = flims[6])
        template_dict = generate_template(TEMPLATE_AUDIO_4,TEMPLATE_PATH, template_dict, 'hf',tlims = tlims['template_4_1'], flims = flims[7])
        template_dict = generate_template(TEMPLATE_AUDIO_3,TEMPLATE_PATH, template_dict, 'hf',tlims = tlims['template_4_2'], flims = flims[8])
        template_dict = generate_template(TEMPLATE_AUDIO_3,TEMPLATE_PATH, template_dict, 'hf',tlims = tlims['template_4_3'], flims = flims[9])

        save_template_dict(template_dict, TEMPLATE_PATH)


    file_list = list(Path(HOME_PATH / '2_clipped_audio_wav').glob('*.wav'))
    total_df = pd.DataFrame()
    for indv_path in file_list:
        AUDIO_PATH = indv_path
        rois_df = run_multiple_template_matching(AUDIO_PATH,RESULT_PATH,
                                                peak_th=0.25,
                                                peak_distance=0.25,
                                                template_dict=template_dict,
                                                num_flim_cuts=5)
        match_df = match_rois(rois_df,RESULT_PATH, 0.25, 4, 0.15, 1)

        match_df['Starting point'] = indv_path.stem[22:]
        total_df = pd.concat([total_df,match_df],ignore_index=True)

    total_df.to_csv(RESULT_PATH/'feeding_buzz_csv/20210910_03000_match_all_template/matching_feeding_buzz_all_template_20210910_03000.csv')
# %%





# %%
