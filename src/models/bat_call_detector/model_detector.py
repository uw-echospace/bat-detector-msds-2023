import os
import argparse
import pandas as pd
import numpy as np
import io
import sys


from models.detection_interface import DetectionInterface
from utils.utils import gen_empty_df

import bat_detect.utils.detector_utils as du
import models.bat_call_detector.feed_buzz_helper as fbh


class BatCallDetector(DetectionInterface):
    """
    A class containing the bat detect model and feeding buzz model. The parameters of this class are explained in cfg.py 
    """
    def __init__(self, detection_threshold, spec_slices, chunk_size, model_path, time_expansion_factor, quiet, cnn_features,
                 peak_distance,peak_threshold,template_dict_path,num_matches_threshold,buzz_feed_range,alpha):
        self.detection_threshold = detection_threshold
        self.spec_slices = spec_slices
        self.chunk_size = chunk_size
        self.model_path = model_path
        self.time_expansion_factor = time_expansion_factor
        self.quiet = quiet
        self.cnn_features = cnn_features
        self.peak_distance = peak_distance
        self.peak_th = peak_threshold
        self.template_dict_path = template_dict_path
        self.num_matches_threshold = num_matches_threshold
        self.buzz_feed_range = buzz_feed_range
        self.alpha = alpha
        

    def get_name(self):
        return "BatDetectorMSDS"

    def _run_batdetect(self, audio_file)-> pd.DataFrame: #
        """
        Parameters:: 
            audio_file: a path containing the post-processed wav file.

        Returns:: a pd.Dataframe containing the bat calls detections
        """
        model, params = du.load_model(self.model_path)

        # Suppress output from this call
        text_trap = io.StringIO()
        sys.stdout = text_trap

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
        # Restore stdout
        sys.stdout = sys.__stdout__

        annotations = model_output['pred_dict']['annotation']

        out_df = gen_empty_df()
        if annotations:
            out_df = pd.DataFrame.from_records(annotations) 
            out_df['detection_confidence'] = out_df['det_prob']
            out_df.drop(columns = ['class', 'class_prob', 'det_prob','individual'], inplace=True)
        return out_df
    