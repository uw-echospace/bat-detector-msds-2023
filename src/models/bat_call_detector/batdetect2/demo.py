import streamlit as st
import os
import glob
import matplotlib.pyplot as plt
import pandas as pd
import bat_detect.utils.detector_utils as du
import bat_detect.utils.audio_utils as au
import bat_detect.utils.plot_utils as viz


st.set_page_config(layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)
st.title('Bat Call Detector Dashboard')

#MODEL_PATH = '/home/exouser/finetuning_hm/bat-detector-msds-2023/src/models/bat_call_detector/batdetect2/experiments/2024_02_16__10_13_21/2024_02_16__10_13_21.pth.tar'
MODEL_PATH= '/home/exouser/finetuning_jw/bat-detector-msds-2023/src/models/bat_call_detector/batdetect2/experiments/2024_02_23__20_10_23/2024_02_23__20_10_23.pth.tar'
ANNS_DIR = '/home/exouser/data/output_anns/'
args = du.get_default_bd_args()
args['time_expansion_factor'] = 1
#args['model_path'] = 'models/Net2DFast_UK_same.pth.tar'
args['model_path'] = MODEL_PATH

# load the model
model, params = du.load_model(args['model_path'])
anns = ANNS_DIR + st.sidebar.selectbox(label='Anns',options=os.listdir(ANNS_DIR))
df = pd.read_json(anns)
audio_file = st.sidebar.selectbox(label='Audio File', options=df['file_path'])
annotations = df[df['file_path']==audio_file]['annotation'].values[0]
params['fft_window_length'] = st.sidebar.slider(label='FFT Window Length',
                                                min_value=0.004,
                                                max_value=0.413,
                                                step=0.001,
                                                value=0.413)
params['fft_overlap'] = st.sidebar.slider(label='FFT Overlap',
                                          min_value=0.25,
                                          max_value=0.75,
                                          value=0.75,
                                          step=0.1)
start_time_from_gt = st.sidebar.radio('Start Time from ground truth', options=['Yes', 'No'])
if start_time_from_gt == 'Yes':
    start_time = st.sidebar.selectbox('Start Time', options=[d['start_time']-1 for d in annotations])
else:
    start_time = st.sidebar.number_input(label='Start Time', min_value=0, max_value=1000000, value=33, step=1)
duration = st.sidebar.slider(label='Duration', min_value=5, max_value=20, value=10, step=1)
end_time = start_time + duration
st.write("Model Path")
st.code(MODEL_PATH, language='bash')
st.write('Audio File:')
st.code(audio_file, language='bash')

col1, col2 =st.columns(2, gap='medium')

with col1:
    st.markdown("**Ground Truth data**")
    table = '| start_time | end_time | lfreq | hfreq | species_name | \n |---|-----|-----|----|-------|\n'
    ground_truth = []
    # TODO - Take input file , currently taking first file
    for ann in annotations:
        if ann['start_time'] >= start_time and ann['end_time'] <= end_time:
            ground_truth.append(ann)
            ground_truth[-1]['start_time'] -= start_time
            ground_truth[-1]['end_time'] -= start_time
            table += '| {} | {} | {} | {} | {} |\n'.format(ann['start_time']+start_time, ann['end_time']+start_time, ann['low_freq'], ann['high_freq'], ann['class'])

    st.write('**{}** Ground truth calls present\n'.format(len(ground_truth)))
    table_container = col1.container(height=500)
    table_container.markdown(table)

w_scale = st.sidebar.number_input('Figure width scale', value=2000, min_value=1)
h_scale = st.sidebar.number_input('Figure height scale', value=100, min_value=1)
dpi = st.sidebar.number_input('Figure DPI ', value=200, min_value=100, step=50)


col2.markdown('**Model Predictions**')
args['detection_threshold'] = st.slider('Detection Threshold', min_value=0.01, max_value=0.99, value=0.3)
run_model = st.button('Run Model')
if run_model:
    with col2:
        with st.spinner(text='Running Model'):
            results = du.process_file(audio_file, model, params, args, max_duration=end_time, start_time=start_time)

        # print summary info for the individual detections 
        st.write('**{}** calls detected\n'.format(len(results['pred_dict']['annotation'])))

        table = '| start_time | end_time | prob | lfreq | hfreq | species_name | \n |---|---|---|---|----|---|\n'
        for ann in results['pred_dict']['annotation']:

            table += '| {} | {} | {} | {} | {} | {} |\n'.format(ann['start_time']+start_time, ann['end_time']+start_time, ann['class_prob'], ann['low_freq'], ann['high_freq'], ann['class'])
        table_container2 = col2.container(height=500)
        table_container2.markdown(table)
    # read the audio file 
    with st.spinner('Loading audio'):
        sampling_rate, audio = au.load_audio_file(audio_file, args['time_expansion_factor'], params['target_samp_rate'], params['scale_raw_audio'], max_duration=end_time, start_time=start_time)
        duration = audio.shape[0] / sampling_rate
        st.write('File duration: {} seconds'.format(duration))

    with st.spinner('Generating Spectrogram'):
        spec, spec_viz = au.generate_spectrogram(audio, sampling_rate, params, True, False)
        st.write(f'Spec shape: {spec.shape}')

    with st.spinner('Plotting results'):
        st.write("## Model Prediction")
        detections = [ann for ann in results['pred_dict']['annotation']]
        fig = plt.figure(1, figsize=((spec.shape[1]/w_scale, spec.shape[0]/h_scale)), dpi=dpi, frameon=False)
        spec_duration = au.x_coords_to_time(spec.shape[1], sampling_rate, params['fft_win_length'], params['fft_overlap'])
        print(spec_duration)
        viz.create_box_image(spec, fig, detections, 0, spec_duration, duration, params, spec.max()*0.4, False, False)
        plt.ylabel('Freq - kHz')
        plt.xlabel('Time - secs')
        plt.title(os.path.basename(audio_file))
        st.pyplot(plt.show())

    with st.spinner('Plotting results'):
        st.write("## Ground Truth")
        detections = ground_truth
        fig = plt.figure(1, figsize=((spec.shape[1]/w_scale, spec.shape[0]/h_scale)), dpi=dpi, frameon=False)
        spec_duration = au.x_coords_to_time(spec.shape[1], sampling_rate, params['fft_win_length'], params['fft_overlap'])
        print(spec_duration)
        viz.create_box_image(spec, fig, detections, 0, spec_duration, duration, params, spec.max()*0.4, False, False)
        plt.ylabel('Freq - kHz')
        plt.xlabel('Time - secs')
        plt.title(os.path.basename(audio_file))
        st.pyplot(plt.show())
else:
    with col2:
        st.write('#####  Click on Run Model')

st.write('--------------------- \n Capstone Project by Harshita Maddi, Rhea Sharma, Jenny Wong')