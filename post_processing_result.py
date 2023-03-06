import pandas as pd
from pathlib import Path
import argparse

def main(args):
        
    input_dir = Path(args['input_dir'])
    json_list = input_dir.glob(f"*.json")

    full_list = pd.DataFrame()

    for indv in json_list:
        curr = pd.json_normalize(pd.read_json(indv)['annotation'])

        curr_start_time = float(indv.stem[22:][:-4])
        curr['start_time'] =  curr['start_time'] + curr_start_time
        curr['end_time'] =  curr['end_time'] + curr_start_time
        
        full_list = pd.concat([full_list,curr],ignore_index = True)

    full_list.sort_values(by = ['start_time'],inplace = True, ascending = True)
    full_list.rename(columns={'det_prob':'detection_confidence',
                            'start_time':'Min_t',
                            'end_time':'Max_t',
                            'low_freq':'Min_f',
                            'high_freq':'Max_f'},inplace=True)

    full_list.to_csv(input_dir/'full_list_{}.csv'.format(input_dir.stem))


if __name__ == "__main__":
        info_str = '\nNon-feedingbuzz Workflow Part 3 - Post Processing Result\n'
        print(info_str)
        
        parser = argparse.ArgumentParser()
        parser.add_argument('input_dir', type=str, help='Input directory for audio')

        args = vars(parser.parse_args())
        main(args)

