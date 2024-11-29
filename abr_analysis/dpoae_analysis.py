import warnings
warnings.filterwarnings('ignore')
import fdasrsf as fs
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import os
from scipy.interpolate import CubicSpline
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

# For plotting ABR curves
def interpolate_and_smooth(final, target_length=244):
    if len(final) > target_length:
        new_points = np.linspace(0, len(final), target_length + 2)
        interpolated_values = np.interp(new_points, np.arange(len(final)), final)
        final = np.array(interpolated_values[:target_length], dtype=float)
    elif len(final) < target_length:
        original_indices = np.arange(len(final))
        target_indices = np.linspace(0, len(final) - 1, target_length)
        cs = CubicSpline(original_indices, final)
        final = cs(target_indices)
    return final


def plot_wave(fig, x_values, y_values, color, name, marker_color=None):
    fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='lines', name=name, line=dict(color=color)))
    if marker_color:
        fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='markers', marker=dict(color=marker_color), name=name,
                                 showlegend=False))


def find_dp_threshold(df,freq):

    for idx, db in enumerate(df['Level(dB)'].unique()):
        # Extract df for this frequency
        df_filtered = df[df['Freq(Hz)']==freq & df['Level(dB)']==db]
        dp_freq = 2 * df['F1'] - df['F2']  # find frequency of distortion product


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
                    peaks,_ = find_peaks(final_normalized,height=final_normalized[0]+.1)
                    if len(peaks) < 2: # in case of very noisy start
                        peaks,_ = find_peaks(final_normalized,height=inal_normalized[0]+.01)
                    peaks = peaks[peaks<(len(final_normalized)/2)-10] # = [] # remove any peaks over max frequency
                    peaks = peaks[peaks>10]
                    F1 = peaks[-2]
                    F2 = peaks[-1]
                    if len(peaks) == 3: # peak finding should hopefully find the DP peak as well as F1 and F2
                        DP = peaks[0]
                    else: # but if it doesn't, calculate
                        DP = 2*F1-F2
                    prev_thresh = False
                    db_thresh = max_db
                    for idx, db_idx in enumerate(
                            unique_dbs):  # now cycle through traces from lowest dB to find threshold
                        ff = df[(df['Freq(Hz)'] == freq) & (df['Level(dB)'] == db_idx)]
                        index = ff.index.values[-1]
                        trace = df.loc[index, '0':].dropna()
                        trace = pd.to_numeric(final, errors='coerce')
                        #trace = trace/max_value
                        trace = trace.to_numpy()
                        st_dev = np.std(trace[np.concatenate([np.arange(DP - 13, DP - 3), np.arange(DP + 3,
                                                                                                   DP + 13)])])  # find std of trace either side of expected peak
                        mean_vals = np.max(np.abs(trace[np.concatenate([np.arange(DP - 13, DP - 3), np.arange(DP + 3,
                                                                                                       DP + 13)])]))  # find mean of trace either side of expected peak
                        #max_val = np.max(trace[[DP-2,DP-1,DP,DP+1,DP+2]])  # find max value at expected frequency range
                        if trace[DP] > mean_vals + st_dev * 2:  # if peak is greater than 2 * sd
                            if prev_thresh: # check that one dB above also has acceptable peak
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
                    #fig.add_annotation(dict(font=dict(color='black',size=10),
                     #                      x=hz[DP], y=[y_values[DP] + .1],
                      #                     showarrow=True,
                       #                    text='Distortion product'),
                        #                   showlegend=False)


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
                  width=400,
                  height=700,
                  yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                  xaxis=dict(showgrid=False, zeroline=False))

        khz = df['Freq(Hz)'] == freq
    fig.show()
        #fig.write_image(os.path.join(saveDir,f'stacked_waves_{freq}Hz_{df.name}.pdf'))
        #fig.write_image(os.path.join(saveDir, f'stacked_waves_{freq}Hz_{df.name}.jpg'))


def get_str(data):
    # return string up until null character only
    ind = data.find(b'\x00')
    if ind > 0:
        data = data[:ind]
    return data.decode('utf-8')


def peak_finding(wave):
    # Prepare waveform
    waveform = interpolate_and_smooth(wave)
    waveform_torch = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)

    # Get prediction from model
    outputs = peak_finding_model(waveform_torch)
    prediction = int(round(outputs.detach().numpy()[0][0], 0))

    # Apply Gaussian smoothing
    smoothed_waveform = gaussian_filter1d(wave, sigma=1)

    # Find peaks and troughs
    n = 18
    t = 14
    start_point = prediction - 9
    smoothed_peaks, _ = find_peaks(smoothed_waveform[start_point:], distance=n)
    smoothed_troughs, _ = find_peaks(-smoothed_waveform, distance=t)
    sorted_indices = np.argsort(smoothed_waveform[smoothed_peaks + start_point])
    highest_smoothed_peaks = np.sort(smoothed_peaks[sorted_indices[-5:]] + start_point)
    relevant_troughs = np.array([])
    for p in range(len(highest_smoothed_peaks)):
        c = 0
        for t in smoothed_troughs:
            if t > highest_smoothed_peaks[p]:
                if p != 4:
                    try:
                        if t < highest_smoothed_peaks[p + 1]:
                            relevant_troughs = np.append(relevant_troughs, int(t))
                            break
                    except IndexError:
                        pass
                else:
                    relevant_troughs = np.append(relevant_troughs, int(t))
                    break
    relevant_troughs = relevant_troughs.astype('i')
    return highest_smoothed_peaks, relevant_troughs

# Open UI for user to select files to be analysed
files = fd.askopenfilenames(title="Select .arf files | Sélectionnez les fichiers .arf")
saveRoot = '/home/nc/Documents/Analysis/Pilot_control/DPOAEs'#= fd.askdirectory(title='Select save directory | Sélectionnez le répertoire de sauvegarde')
counter = 1
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
        distinct_freqs = sorted(pd.concat([df['Freq(Hz)']]).unique())
        distinct_dbs = sorted(pd.concat([df['Level(dB)']]).unique())

        # Cycle through frequencies
        for freq in distinct_freqs:
            plot_waves_stacked(df,freq)

        # Extract important metrics
        metrics_data = make_metrics_table(df, distinct_freqs, distinct_dbs)
        # Append to metrics table for all click/PT sessions
        for key,val in dict.items(metrics_data):
            a = 1
            if click:
                click_metrics_data[key] = click_metrics_data[key] + val
            else:
                pt_metrics_data[key] = pt_metrics_data[key] + val
        metrics_data = pd.DataFrame(metrics_data)
        metrics_data.to_csv(os.path.join(saveDir,'dataTable.csv'))
    counter = counter + 1

# save all the click and PT datatables together
if len(click_metrics_data['File Name']) > 0:
    metrics_data = pd.DataFrame(click_metrics_data)
    metrics_data.sort_values(by=['Mouse Name','Date'])
    metrics_data.to_csv(os.path.join(saveRoot, 'click_dataTable.csv'))
    # Plot click thresholds over days

if len(pt_metrics_data['File Name']) > 0:
    metrics_data = pd.DataFrame(pt_metrics_data)
    metrics_data.sort_values(by=['Mouse Name','Date'])
    metrics_data.to_csv(os.path.join(saveRoot, 'pt_dataTable.csv'))
    # Plot pure tone thresholds over days








