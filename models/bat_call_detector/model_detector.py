import os
import argparse
import pandas as pd
import numpy as np



from models.detection_interface import DetectionInterface
from utils.utils import gen_empty_df

import bat_detect.utils.detector_utils as du
import models.bat_call_detector.feed_buzz_helper as fbh



class BatCallDetector(DetectionInterface):

    # TODO: what constructor params are needed?
    def __init__(self, detection_threshold, spec_slices, chunk_size, model_path, time_expansion_factor, quiet, cnn_features,
                 peak_distance,peak_threshold,template_dict_path,num_matches_threshold,buzz_feed_range,alpha):
        self.detection_threshold = detection_threshold
        self.spec_slices = spec_slices
        self.chunk_size = chunk_size
        self.model_path = model_path
        self.time_expansion_factor = time_expansion_factor
        self.quiet = quiet
        self.cnn_features = cnn_features
        self.peak_distance = peak_distance,
        self.peak_th = peak_threshold,
        self.template_dict_path = template_dict_path,
        self.num_matches_threshold = num_matches_threshold, 
        self.buzz_feed_range = buzz_feed_range, 
        self.alpha = alpha

    def _get_name(self):
        return "batdetect2"

    def _run_batdetect(self, audio_file): #
        print('Loading model: ' + self.model_path)
        model, params = du.load_model(self.model_path)

        model_output = du.process_file(
            audio_file=audio_file, 
            model=model, 
            params=params, 
            args= {
                'detection_threshold': self.detection_threshold,
                'spec_slices': self.spec_slices,
                'chunk_size': self.chunk_size,
                'quiet': self.quiet,
                'spec_features' : False,
                'cnn_features': self.cnn_features,
            },
            time_exp=self.time_expansion_factor,
        )

        # TODO: what else needs to go in here?
        out_df = gen_empty_df()

        # TODO: move to base class
        annotations = model_output['pred_dict']['annotation']

        if annotations:
            out_df = pd.DataFrame.from_records(annotations) 
            out_df['detection_confidence'] = out_df['det_prob']
            out_df.drop(columns = ['class', 'class_prob', 'det_prob','individual'], inplace=True)
        return out_df
    
    def _run_feedbuzz(self, audio_file) -> pd.DataFrame: # TODO: type annotations
        print('Running feeding buzz template matching.')
        out_df = gen_empty_df()
        template_dict = fbh.load_templates(self.template_dict_path[0])
        out_df = fbh.run_multiple_template_matching(
                                            PATH_AUDIO=audio_file,
                                            out_df=out_df,
                                            peak_distance=self.peak_distance[0], #self.peak_distance is a tuple for some reason.
                                            peak_th=self.peak_th[0],
                                            template_dict=template_dict,
                                            num_matches_threshold=self.num_matches_threshold[0], 
                                            buzz_feed_range=self.buzz_feed_range[0], 
                                            alpha=self.alpha)
        
        # A flag for end user to differentiate between feeding buzz and bat calls.
        out_df['event'] = 'Feeding Buzz'
        return out_df
    
    """
    Remove collision between feeding buzz false positive and bat calls true positive values.

    Return: a boolean
    """
    def _removing_collision(self,curr_row:tuple, compare_df:pd.DataFrame):
        # TODO: Decide if bounding box interect is a good idea (might remove TP), maybe better to compare in center
        XB1 = curr_row.start_time
        XB2 = curr_row.end_time
        YB1 = curr_row.low_freq
        YB2 = curr_row.high_freq
        SB = (XB2 - XB1) * (YB2 - YB1)
    
        #print('Looping compare df to find collision')
        for i in compare_df.itertuples():
            XA1 = i.start_time #min_t
            XA2 = i.end_time #max_t
            YA1 = i.low_freq #min_f
            YA2 = i.high_freq #max_f

            if (XB2 >= XA2 and XA1 >= XB1 and YB2 >= YA2 and YA1 >= YB1 ):
                return 1
        return 0
    
        
    """
    Creates a loop for feeding buzz to remove false positive.

    Return: pd.DataFrame
    """
    def _buzzfeed_fp_removal(self,bd_output, fb_output):
        print('Entering FP removal.')

        collide = np.zeros(len(fb_output))
        for curr in fb_output.itertuples():
            collide[curr.Index] = self._removing_collision(curr,bd_output)
        
        fb_output['Collide'] = collide
        fb_df_filtered = fb_output[fb_output['Collide']== 0] 
        del fb_df_filtered['Collide']
        
        return fb_df_filtered
    
    def run(self, audio_file):
        bd_output = self._run_batdetect(audio_file)
        fb_output = self._run_feedbuzz(audio_file)
        print('FB output:')
        print(fb_output)
        fb_final_output = self._buzzfeed_fp_removal(bd_output, fb_output)

        return pd.concat([bd_output,fb_final_output])
    