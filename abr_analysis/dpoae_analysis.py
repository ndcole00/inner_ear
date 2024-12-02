import warnings
warnings.filterwarnings('ignore')
import fdasrsf as fs
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy import stats
import os
import plotly.graph_objects as go
import struct
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.ndimage import gaussian_filter1d
import colorcet as cc
from tkinter import filedialog as fd


# N.Cole ULiege 2024 - mostly adapted from Erra et al., 2024 (https://www.biorxiv.org/content/10.1101/2024.06.20.599815v1)
# For analysing ABR sessions, uses CNNs for finding hearing thresholds and wave 1 from click or pure tone sessions
# Plots single waves and collected waves at each frequency for all dBs, including stacked and 3D plots
# Extracts estimated thresholds and wave data and saves .csv file for each session, as well as concatenated data across all click and PT sessions
# Plots basic metrics for changes over time within these groups

# DEPENDENCIES:
# Before running, run Init file in console to install necessary modules.
# 'Models' folder containing 'abr_cnn_aug_norm_opt.keras' and 'waveI_cnn_model.pth' must be in same folder as this script
# Code relies on all files having naming structure 'date_clickOrPT_ear_mouseName_timepoint', in this order, to do between-session analysis


# For reading .arf files
def arfread(PATH, **kwargs):

    data = {'RecHead': {}, 'groups': []}

    # open file
    with open(PATH, 'rb') as fid:
        # open RecHead data
        data['RecHead']['ftype'] = struct.unpack('h', fid.read(2))[0]
        data['RecHead']['ngrps'] = struct.unpack('h', fid.read(2))[0]
        data['RecHead']['nrecs'] = struct.unpack('h', fid.read(2))[0]
        data['RecHead']['grpseek'] = struct.unpack('200i', fid.read(4 * 200))
        data['RecHead']['recseek'] = struct.unpack('2000i', fid.read(4 * 2000))
        data['RecHead']['file_ptr'] = struct.unpack('i', fid.read(4))[0]

        data['groups'] = []
        bFirstPass = True
        for x in range(data['RecHead']['ngrps']):
            # jump to the group location in the file
            fid.seek(data['RecHead']['grpseek'][x], 0)

            # open the group
            data['groups'].append({
                'grpn': struct.unpack('h', fid.read(2))[0],
                'frecn': struct.unpack('h', fid.read(2))[0],
                'nrecs': struct.unpack('h', fid.read(2))[0],
                'ID': get_str(fid.read(16)),
                'ref1': get_str(fid.read(16)),
                'ref2': get_str(fid.read(16)),
                'memo': get_str(fid.read(50)),
            })

            # read temporary timestamp
            if bFirstPass:
                ttt = struct.unpack('q', fid.read(8))[0]
                fid.seek(-8, 1)
                data['fileType'] = 'BioSigRZ'
                bFirstPass = False

            grp_t_format = 'q'
            beg_t_format = 'q'
            end_t_format = 'q'
            read_size = 8

            data['groups'][x]['beg_t'] = struct.unpack(beg_t_format, fid.read(read_size))[0]
            data['groups'][x]['end_t'] = struct.unpack(end_t_format, fid.read(read_size))[0]

            data['groups'][x].update({
                'sgfname1': get_str(fid.read(100)),
                'sgfname2': get_str(fid.read(100)),
                'VarName1': get_str(fid.read(15)),
                'VarName2': get_str(fid.read(15)),
                'VarName3': get_str(fid.read(15)),
                'VarName4': get_str(fid.read(15)),
                'VarName5': get_str(fid.read(15)),
                'VarName6': get_str(fid.read(15)),
                'VarName7': get_str(fid.read(15)),
                'VarName8': get_str(fid.read(15)),
                'VarName9': get_str(fid.read(15)),
                'VarName10': get_str(fid.read(15)),
                'VarUnit1': get_str(fid.read(5)),
                'VarUnit2': get_str(fid.read(5)),
                'VarUnit3': get_str(fid.read(5)),
                'VarUnit4': get_str(fid.read(5)),
                'VarUnit5': get_str(fid.read(5)),
                'VarUnit6': get_str(fid.read(5)),
                'VarUnit7': get_str(fid.read(5)),
                'VarUnit8': get_str(fid.read(5)),
                'VarUnit9': get_str(fid.read(5)),
                'VarUnit10': get_str(fid.read(5)),
                'SampPer_us': struct.unpack('f', fid.read(4))[0],
                'cc_t': struct.unpack('i', fid.read(4))[0],
                'version': struct.unpack('h', fid.read(2))[0],
                'postproc': struct.unpack('i', fid.read(4))[0],
                'dump': get_str(fid.read(92)),
                'recs': [],
            })

            for i in range(data['groups'][x]['nrecs']):
                record_data = {
                    'recn': struct.unpack('h', fid.read(2))[0],
                    'grpid': struct.unpack('h', fid.read(2))[0],
                    'grp_t': struct.unpack(grp_t_format, fid.read(read_size))[0],
                    'newgrp': struct.unpack('h', fid.read(2))[0],
                    'sgi': struct.unpack('h', fid.read(2))[0],
                    'chan': struct.unpack('B', fid.read(1))[0],
                    'rtype': get_str(fid.read(1)),
                    'npts': struct.unpack('H', fid.read(2))[0],
                    'osdel': struct.unpack('f', fid.read(4))[0],
                    'range_hz': struct.unpack('f', fid.read(4))[0],
                    'SampPer_us': struct.unpack('f', fid.read(4))[0],
                    'artthresh': struct.unpack('f', fid.read(4))[0],
                    'gain': struct.unpack('f', fid.read(4))[0],
                    'accouple': struct.unpack('h', fid.read(2))[0],
                    'navgs': struct.unpack('h', fid.read(2))[0],
                    'narts': struct.unpack('h', fid.read(2))[0],
                    'beg_t': struct.unpack(beg_t_format, fid.read(read_size))[0],
                    'end_t': struct.unpack(end_t_format, fid.read(read_size))[0],
                    'Var1': struct.unpack('f', fid.read(4))[0],
                    'Var2': struct.unpack('f', fid.read(4))[0],
                    'Var3': struct.unpack('f', fid.read(4))[0],
                    'Var4': struct.unpack('f', fid.read(4))[0],
                    'Var5': struct.unpack('f', fid.read(4))[0],
                    'Var6': struct.unpack('f', fid.read(4))[0],
                    'Var7': struct.unpack('f', fid.read(4))[0],
                    'Var8': struct.unpack('f', fid.read(4))[0],
                    'Var9': struct.unpack('f', fid.read(4))[0],
                    'Var10': struct.unpack('f', fid.read(4))[0],
                    'data': []
                    # list(struct.unpack(f'{data["groups"][x]["recs"][i]["npts"]}f', fid.read(4*data['groups'][x]['recs'][i]['npts'])))
                }

                # skip all 10 cursors placeholders
                fid.seek(36 * 10, 1)
                record_data['data'] = list(struct.unpack(f'{record_data["npts"]}f', fid.read(4 * record_data['npts'])))


                data['groups'][x]['recs'].append(record_data)

    return data


def plot_waves_stacked(df,freq):
    fig = go.Figure()
    # Get unique dB levels and color palette
    unique_dbs = sorted(df['Level(dB)'].unique())
    num_dbs = len(unique_dbs)
    vertical_spacing = 25 / num_dbs
    db_offsets = {db: i * vertical_spacing for i, db in enumerate(unique_dbs)}
    glasbey_colors = cc.glasbey_dark[:num_dbs]

    db_levels = sorted(unique_dbs, reverse=True)
    max_db = db_levels[0]
    db_thresh = max_db

    for i, db in enumerate(db_levels):
        try:
            khz = df[(df['Freq(Hz)'] == freq) & (df['Level(dB)'] == db)]

            if not khz.empty:
                index = khz.index.values[-1]
                final = df.loc[index, '0':].dropna()
                final = pd.to_numeric(final, errors='coerce')
                #final = interpolate_and_smooth(final)
                if db == max_db:
                    max_value = np.max(np.abs(final))
                final_normalized = final / max_value
                hz = np.linspace(0, range_hz, len(final_normalized))
                if db == max_db:  # first plot only
                    # Find peaks in highest dB trace
                    # Normalize the waveform
                    peaks,_ = find_peaks(final_normalized,height=final_normalized[0]+.2)
                    if len(peaks) < 2: # in case of very noisy start
                        peaks,_ = find_peaks(final_normalized,height=final_normalized[0]+.1)
                    peaks = peaks[peaks<(len(final_normalized)/2)-10] # = [] # remove any peaks over max frequency
                    peaks = peaks[peaks>10]
                    if len(peaks) > 3: # in case too many peaks still
                        temp_hz = np.asarray(hz)
                        freq_idx = (np.abs(temp_hz - (freq*1.1))).argmin()
                        peaks = peaks[peaks < freq_idx] # remove any values above F2 (freq * 1.09)
                        if len(peaks) > 3: # in case this didn't work
                            freq_idx = (np.abs(temp_hz - (freq * .909))).argmin() # manually find where F1 should be
                            F1 = np.abs(peaks-freq_idx).argmin() # find the peak that corresponds
                            freq_idx = (np.abs(temp_hz - (freq * 1.09))).argmin()  # manually find where F2 should be
                            F2 = np.abs(peaks - freq_idx).argmin() # find the peak that corresponds
                            peaks = peaks[[F1,F2]]
                    F1 = peaks[-2]
                    F2 = peaks[-1]
                    if len(peaks) == 3: # peak finding should hopefully find the DP peak as well as F1 and F2
                        DP = peaks[0]
                    else: # but if it doesn't, calculate
                        DP = 2*F1-F2
                    prev_thresh = False
                    for idx, db_idx in enumerate(
                            unique_dbs):  # now cycle through traces from lowest dB to find threshold
                        ff = df[(df['Freq(Hz)'] == freq) & (df['Level(dB)'] == db_idx)]
                        ix = ff.index.values[-1]
                        trace = df.loc[ix, '0':].dropna()
                        trace = pd.to_numeric(trace, errors='coerce')
                        trace = trace.to_numpy()
                        st_dev = np.std(trace[np.concatenate([np.arange(DP - 8, DP - 3), np.arange(DP + 3,
                                                                                                   DP + 8)])])  # find std of trace either side of expected peak
                        #st_dev = np.std(trace)
                        #mean_vals = np.mean(np.abs(trace))
                        mean_vals = np.mean(trace[np.concatenate([np.arange(DP - 8, DP - 3), np.arange(DP + 3,
                                                                                                       DP + 8)])])  # find mean of trace either side of expected peak
                        #max_val = np.max(trace[[DP-2,DP-1,DP,DP+1,DP+2]])  # find max value at expected frequency range
                        if trace[DP] > (mean_vals + st_dev * 2):  # if peak is greater than mean + [2 * sd]
                            if prev_thresh and idx>0: # check that one dB above also has acceptable peak
                                db_thresh = unique_dbs[idx - 1]
                                break
                            prev_thresh = True
                        else:
                            prev_thresh = False



                # Apply vertical offset
                y_values = final_normalized + db_offsets[db]

                # Plot the waveform
                color_scale = glasbey_colors[i]
                if (db == db_thresh) and (db_thresh != max_db): # highlight threshold trace, if there is one
                    fig.add_trace(go.Scatter(x=np.linspace(1, range_hz, len(y_values)),
                                     y=y_values,
                                     mode='lines',
                                     name=f'{int(db)} dB (threshold)',
                                     line=dict(color='black',width=5),
                                     showlegend=True))
                else:
                    fig.add_trace(go.Scatter(x=np.linspace(1, range_hz, len(y_values)),
                                             y=y_values,
                                             mode='lines',
                                             name=f'{int(db)} dB',
                                             line=dict(color=color_scale),
                                             showlegend=True))

                if db == max_db: # top plot only
                    fig.add_trace(go.Scatter(x=[hz[DP], hz[F1], hz[F2]], y=[y_values[DP]+.1,y_values[F1] + .1, y_values[F2] + .1],
                                             mode='markers', marker=dict(symbol='triangle-down', color='black'),showlegend=False))
                   # fig.add_trace(go.Scatter(x=[hz[DP]], y=[y_values[DP] + 0.5],mode='lines+text',text='dp',
                                     #        showlegend=False,orientation='v'))


                fig.add_annotation(
                        x=10,
                        y=y_values[-1] + 0.5,
                        xref="x",
                        yref="y",
                        text=f"{int(db)} dB",
                        showarrow=False,
                        font=dict(size=10, color=color_scale),
                        xanchor="right"
                            )
        except Exception as e:
            print('Error plotting stacked waves')
        fig.update_layout(title=dict(text=f'{df.name.split("/")[-1]} - Frequency: {freq} Hz', font = dict(family='Helvetica', size=15)),
                  xaxis_title='Frequency(Hz)',
                  yaxis_title='Voltage (μV)',
                  width=500,
                  height=700,
                  yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                  xaxis=dict(showgrid=False, zeroline=False))

    fig.write_image(os.path.join(saveDir,f'stacked_waves_{freq}Hz_{df.name}.pdf'))
    fig.write_image(os.path.join(saveDir, f'stacked_waves_{freq}Hz_{df.name}.jpg'))

    return db_thresh


def get_str(data):
    # return string up until null character only
    ind = data.find(b'\x00')
    if ind > 0:
        data = data[:ind]
    return data.decode('utf-8')


# Open UI for user to select files to be analysed
files = fd.askopenfilenames(title="Select .arf files | Sélectionnez les fichiers .arf")
saveRoot = '/home/nc/Documents/Analysis/Pilot_control/DPOAEs'#= fd.askdirectory(title='Select save directory | Sélectionnez le répertoire de sauvegarde')
counter = 1
metrics_data_all = {'File Name': [], 'Date': [], 'Session Type': [], 'Ear': [], 'Mouse Name': [], 'Timepoint': [],
                        'Frequency (Hz)': [], 'dB Level': [], 'DP Threshold': []}
for file in files:
    annotations = []
    dfs = []
    if file.endswith(".arf"):
        print(f'Analysing {file.split('/')[-1]}, file {counter} of {len(files)}...')
        # Read ARF file
        data = arfread(file)
        # Process ARF data
        rows = []
        freqs = []
        dbs = []
        calibration_levels = {}
        for group in data['groups']:
            for rec in group['recs']:
                # Extract data
                db = rec['Var1']
                freq = rec['Var2']
                F1 = rec['Var5']
                F2 = rec['Var6']
                range_hz = rec['range_hz']
                # Construct row
                wave_cols = list(enumerate(rec['data']))
                wave_data = {f'{i}': v * 1e6 for i, v in wave_cols}
                row = {'Freq(Hz)': freq, 'Level(dB)': db, 'F1_freq':F1,'F2_freq':F2, **wave_data}
                rows.append(row)

        metrics_data = {'File Name': [], 'Date': [], 'Session Type': [], 'Ear': [], 'Mouse Name': [], 'Timepoint': [],
                        'Frequency (Hz)': [], 'dB Level': [], 'DP Threshold': []}
        # Make dataframe with all data from session
        df = pd.DataFrame(rows)
        df.name = file.split('/')[-1][0:-4]
        db_column = 'Level(dB)'

        # Create separate folder for each session
        saveDir = os.path.join(saveRoot,df.name)
        if not os.path.exists(saveDir):
            os.mkdir(saveDir)
        df.to_csv(os.path.join(saveDir,'data.csv'))


        # Get distinct frequency and dB level values used in session
        if not any(df.columns=='Freq(Hz)'): # in case session does not have required data
            print('Wrong recording type, skipping...')
            continue
        distinct_freqs = sorted(pd.concat([df['Freq(Hz)']]).unique())
        distinct_freqs = [freq for freq in distinct_freqs if freq<=30000] # correct for strange bug where one trial of 32kHz is recorded
        distinct_dbs = sorted(pd.concat([df['Level(dB)']]).unique())

        # Cycle through frequencies
        for freq in distinct_freqs:
            db_thresh = plot_waves_stacked(df,freq)
            for db in distinct_dbs:
                # Update datatable for this session
                metrics_data['File Name'].append(df.name)
                metrics_data['Date'].append(df.name.split('_')[0])
                metrics_data['Session Type'].append(df.name.split('_')[1])
                metrics_data['Ear'].append(df.name.split('_')[2])
                metrics_data['Mouse Name'].append(df.name.split('_')[3])
                metrics_data['Timepoint'].append(df.name.split('_')[4])
                metrics_data['Frequency (Hz)'].append(freq)
                metrics_data['dB Level'].append(db)
                metrics_data['DP Threshold'].append(db_thresh)

        # Append to across sessions metrics data
        for key, val in dict.items(metrics_data):
            metrics_data_all[key] = metrics_data_all[key] + val
        metrics_data = pd.DataFrame(metrics_data)
        metrics_data = metrics_data.sort_values(by=['Mouse Name', 'Date'])
        metrics_data.to_csv(os.path.join(saveDir, 'dataTable.csv'))
        counter = counter + 1

# Save the across-sessions metrics table
metrics_data_all = pd.DataFrame(metrics_data_all)
metrics_data_all = metrics_data_all.sort_values(by=['Mouse Name', 'Date'])
metrics_data_all.to_csv(os.path.join(saveRoot, 'DPOAEs_dataTable.csv'))
