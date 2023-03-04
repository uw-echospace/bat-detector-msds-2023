import pandas as pd
import numpy as np
from pathlib import Path

# %%
def manipulate_true_df(true_df:pd.DataFrame, choice:str = 'buzz'):

    # Manipulate true values
    true_df = true_df[true_df['View'] == 'Waveform 1' ].drop(['View','Channel'], axis=1).reset_index(drop=True)
    true_df['Label'] = 1
    if choice == 'call':
        true_df = true_df[true_df['Annotation'].apply(lambda x: 'call' in x)]

    elif choice == 'buzz':
       true_df = true_df[true_df['Annotation'].apply(lambda x: 'buzz' in x)]
    
    true_select = true_df[['Begin Time (s)','End Time (s)','Low Freq (Hz)','High Freq (Hz)','Label']]
    true_select.rename(columns={'Begin Time (s)':'Min_t','End Time (s)':'Max_t','Low Freq (Hz)':'min_f','High Freq (Hz)':'max_f'}, inplace=True)

    return true_select

def manipulate_detected_df(detected_df:pd.DataFrame):
    detected_df['Min_t'] = detected_df['min_t'] + detected_df['Starting point']
    detected_df['Max_t'] = detected_df['max_t'] + detected_df['Starting point']
    detected_df['Label'] = 0

    detected_df = detected_df[['Min_t','Max_t','min_f','max_f','Label']]
    return detected_df


# rename and join df
def combine_df(detected_select:pd.DataFrame, true_select:pd.DataFrame,choice:str):
    detected_select = manipulate_detected_df(detected_select)
    true_select = manipulate_true_df(true_select,choice)

    total_select = pd.concat([detected_select,true_select])
    total_select.sort_values(by=['Min_t'],ascending=True,inplace=True)

    total_select['Diff'] = total_select['Label'].diff()
    total_select['Diff'].replace(to_replace= np.nan, value = 0, inplace=True)

    return total_select

def check_TP(total_select:pd.DataFrame,threshold:float = 0.2):
    FP ,TP, FN = 0,0,0
    total_len = len(total_select)


    for curr in range(0, total_len):
        # Case 1,0 and 1,1
        if total_select.iloc[curr]['Label'] == 1:
            FN += 1 # this includes the duplicate of TP, will deduct later per each TP

        # Case 0,0
        elif total_select.iloc[curr]['Label'] == 0 and total_select.iloc[curr]['Diff'] == 0:
            FP += 1
        
        # Case 0,-1
        else:
            FP += 1
            # check with previous and after if their Label == 1
            time_diff = abs(total_select.iloc[curr]['Min_t'] - total_select.iloc[curr - 1]['Min_t'])
            if time_diff <= threshold: #comparing this and previous
                TP += 1
                FN -= 1 #deduct back the original assumption of FN
                FP -= 1 #deduct back the original assumption of FP
            else:
            # could not find TP with previous, check next one if it is label == 1, then check diff.
                if curr + 1 <= total_len and total_select.iloc[curr + 1]['Label'] == 1:
                    time_diff = abs(total_select.iloc[curr]['Min_t'] - total_select.iloc[curr + 1]['Min_t'])
                    if time_diff <= threshold:
                        TP += 1
                        FN -= 1 #deduct back the original assumption of FN
                        FP -= 1 #deduct back the original assumption of FP
    return TP, FP, FN

#%% Output cells with statistics
if __name__ == '__main__':
    HOME_PATH = Path('/Users/kirsteenng/Desktop/UW/DATA 590/')

    detected_df = pd.read_csv(HOME_PATH/'workflow/4_feeding_buzz_detector/feeding_buzz_csv/matching_feeding_buzz_all_template_20210910_033000__0.00_60.00.csv')
    
    true_df = pd.read_table(HOME_PATH/'wav_annotation/2021_09_10_txt/20210910_033000.selections.txt')

    total_select = combine_df(detected_select=detected_df,
                              true_select=true_df,
                              choice='call'
                              )


    TP,FP,FN = check_TP(total_select)

    ACTUAL_VALUE = len(true_df)
    TOTAL_DETECT = len(detected_df)

    print('Total feeding buzz detection: {}\nTrue Positive: {}\nFalse Positive: {}\nFalse Negative: {}'.format(TOTAL_DETECT,TP, FP, FN))

    print('Accuracy: {}'.format(round(TP/TOTAL_DETECT,2)))
    print('Recall: {}'.format(round(TP/(TP+FN),2)))


# %%
