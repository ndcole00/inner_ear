import warnings
import io
import plotly.io as pio
from PIL import Image
warnings.filterwarnings('ignore')
import fdasrsf as fs
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import os
from scipy.interpolate import CubicSpline
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import struct
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.ndimage import gaussian_filter1d
import torch
import torch.nn as nn
import torch.optim as optim
import keras
from tensorflow.keras.models import load_model
import colorcet as cc
import tkinter
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


# Define the CNN model
class CNN(nn.Module):
    def __init__(self, dropout_prob=0.1):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 61, 128)
        self.fc2 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(-1, 32 * 61)
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

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
                    'dur_ms': struct.unpack('f', fid.read(4))[0],
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

def calculate_and_plot_wave(df, freq, db, color, threshold=None):

    khz = df[(df['Freq(Hz)'] == freq) & (df['Level(dB)'] == db)]
    if not khz.empty:
        index = khz.index.values[0]
        final = df.loc[index, '0':].dropna()
        final = pd.to_numeric(final, errors='coerce').dropna()

        target = int(244 * (time_scale / 10))

        y_values = interpolate_and_smooth(final, target)  # Original y-values for plotting
        sampling_rate = len(y_values) / time_scale

        x_values = np.linspace(0, len(y_values) / sampling_rate, len(y_values))

        y_values_for_peak_finding = interpolate_and_smooth(final[:244])
        highest_peaks, relevant_troughs = peak_finding(y_values_for_peak_finding)

        return x_values, y_values, highest_peaks, relevant_troughs
    return None, None, None, None


def plot_waves_single_frequency(df, freq, y_min, y_max,threshold):

    # Make new figure
    fig = go.Figure()
    # Find different dBs, assign colours
    db_levels = sorted(df['Level(dB)'].unique())
    glasbey_colors = cc.glasbey[:len(db_levels)]
    original_waves = []


    for i, db in enumerate(sorted(db_levels)):
        x_values, y_values, highest_peaks, relevant_troughs = calculate_and_plot_wave(df, freq, db,
                                                                                              glasbey_colors[i])
        if y_values is not None:
            fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='lines', name=f'{int(db)} dB',
                                             line=dict(color=glasbey_colors[i])))


            if i == len(db_levels): # only add to legend once
                # Mark the highest peaks with black markers
                fig.add_trace(go.Scatter(x=x_values[highest_peaks], y=y_values[highest_peaks], mode='markers',
                                         marker=dict(color='black'), name='Peaks'))  # , showlegend=False))
                # Mark the relevant troughs with grey markers
                fig.add_trace(go.Scatter(x=x_values[relevant_troughs], y=y_values[relevant_troughs], mode='markers',
                                         marker=dict(color='grey'), name='Troughs'))  # , showlegend=False))
            else:
                fig.add_trace(go.Scatter(x=x_values[highest_peaks], y=y_values[highest_peaks], mode='markers',
                                         marker=dict(color='black'), name='Peaks', showlegend=False))

                fig.add_trace(go.Scatter(x=x_values[relevant_troughs], y=y_values[relevant_troughs], mode='markers',
                                         marker=dict(color='grey'), name='Troughs', showlegend=False))
    # Add thick black line for threshold (if it has been found)
    if threshold is not None:
        x_values, y_values, _, _ = calculate_and_plot_wave(df, freq, threshold, 'black')
        if y_values is not None:
            fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='lines', name=f'Threshold: {int(threshold)} dB',
                                     line=dict(color='black', width=5)))

    fig.update_layout(title=dict(text=f'{df.name} - Frequency: {freq}Hz', font=dict(family='Helvetica',size=20)),
                    xaxis_title='Time (ms)',
                    yaxis_title='Voltage (μV)')
    fig.update_layout(annotations=annotations)
    fig.update_layout(yaxis_range=[y_min, y_max])
    fig.update_layout(width=700, height=600)

    fig.write_image(os.path.join(saveDir, f'all_waves_{freq}_{df.name}.pdf'))
    fig.write_image(os.path.join(saveDir, f'all_waves_{freq}_{df.name}.jpg'))

def plot_waves_single_tuple(freq, db, y_min, y_max):
    fig = go.Figure()
    x_values, y_values, highest_peaks, relevant_troughs = calculate_and_plot_wave(df, freq, db, 'blue')

    if y_values is not None:
        fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='lines', name=df.name,
                                 showlegend=False))
        # Mark the highest peaks with red markers
        fig.add_trace(go.Scatter(x=x_values[highest_peaks], y=y_values[highest_peaks], mode='markers',
                                     marker=dict(color='black'), name='Peaks', showlegend=False))

        # Mark the relevant troughs with blue markers
        fig.add_trace(go.Scatter(x=x_values[relevant_troughs], y=y_values[relevant_troughs], mode='markers',
                                     marker=dict(color='grey'), name='Troughs', showlegend=False))


    y_units = 'Voltage (μV)'
    fig.update_layout(width=700, height=450)
    fig.update_layout(xaxis_title='Time (ms)', yaxis_title=y_units,
                      title=f'{df.name}, Freq = {freq}, {db}dB')
    fig.update_layout(annotations=annotations)
    fig.update_layout(yaxis_range=[y_min, y_max])
    fig.update_layout(font_family="Helvetica",
                      font_color="black",
                      title_font_family="Helvetica",
                      font=dict(size=17))

    fig.write_image(os.path.join(saveDir, f'single_wave_{freq}Hz_{db}dB_{df.name}.pdf'))
    fig.write_image(os.path.join(saveDir, f'single_wave_{freq}Hz_{db}dB_{df.name}.jpg'))

    return fig


def plot_3d_surface(df, freq, y_min, y_max,threshold):
    try:
        fig = go.Figure()
        db_levels = sorted(df['Level(dB)'].unique(), reverse=True)
        original_waves = []
        glasbey_colors = cc.glasbey[:len(db_levels)]
        for db in db_levels:
            x_values, y_values, _, _ = calculate_and_plot_wave(df, freq, db, 'blue')
            if y_values is not None:
                original_waves.append(y_values.tolist())

        original_waves_array = np.array([wave[:-1] for wave in original_waves])
        try:
            time = np.linspace(0, 10, original_waves_array.shape[1])
            obj = fs.fdawarp(original_waves_array.T, time)
            obj.srsf_align(parallel=True)
            warped_waves_array = obj.fn.T
        except IndexError:
            warped_waves_array = np.array([])

        for i, (db, warped_waves) in enumerate(zip(db_levels, warped_waves_array)):
            fig.add_trace(
                go.Scatter3d(x=[db] * len(warped_waves), y=x_values, z=warped_waves, mode='lines', name=f'{int(db)} dB',
                                 line=dict(color=glasbey_colors[i])))
            if db == threshold:
                fig.add_trace(go.Scatter3d(x=[db] * len(warped_waves), y=x_values, z=warped_waves, mode='lines',
                                           name=f'Thresh: {int(db)} dB', line=dict(color='black', width=5)))

        for i in range(len(time)):
            z_values_at_time = [warped_waves_array[j, i] for j in range(len(db_levels))]
            fig.add_trace(go.Scatter3d(x=db_levels, y=[time[i]] * len(db_levels), z=z_values_at_time, mode='lines',
                                       name=f'Time: {time[i]:.2f} ms', line=dict(color='rgba(0, 255, 0, 0.3)'),
                                       showlegend=False))

        fig.update_layout(width=700, height=450)
        fig.update_layout(title= dict(text=f'{df.name} - Frequency: {freq} Hz',font=dict(family='Helvetica', size=20)),
                          scene=dict(xaxis_title='dB', yaxis_title='Time (ms)', zaxis_title='Voltage (μV)'),
                          annotations=annotations)

        fig.write_html(os.path.join(saveDir, f'3d_waves_{freq}Hz_{df.name}.html'))
    except:
        print('Error plotting 3D waves')

# Put all important metrics data from this session into table
def make_metrics_table(df, freqs, db_levels):
    # Create empty dictionary for storing data
    metrics_data = {'File Name': [], 'Date': [], 'Session Type': [], 'Ear': [], 'Mouse Name': [], 'Timepoint': [],
                    'Frequency (Hz)': [], 'dB Level': [], 'Wave I amplitude (P1-T1) (μV)': [],
                    'Latency to First Peak (ms)': [], 'Amplitude Ratio (Peak1/Peak4)': [], 'Estimated Threshold': []}

    for idx,freq in enumerate(freqs):
        threshold = all_thresholds[idx]
        for db in db_levels:
            _, y_values, highest_peaks, relevant_troughs = calculate_and_plot_wave(df, freq, db, 'blue')
            if highest_peaks is not None:
                if highest_peaks.size > 0:  # Check if highest_peaks is not empty
                    first_peak_amplitude = y_values[highest_peaks[0]] - y_values[relevant_troughs[0]]
                    latency_to_first_peak = highest_peaks[0] * (
                                10 / len(y_values))  # Assuming 10 ms duration for waveform

                    if len(highest_peaks) >= 4 and len(relevant_troughs) >= 4:
                        amplitude_ratio = (y_values[highest_peaks[0]] - y_values[relevant_troughs[0]]) / (
                                y_values[highest_peaks[3]] - y_values[relevant_troughs[3]])
                    else:
                        amplitude_ratio = np.nan
                    metrics_data['File Name'].append(df.name)
                    metrics_data['Date'].append(df.name.split('_')[0])
                    metrics_data['Session Type'].append(df.name.split('_')[1])
                    metrics_data['Ear'].append(df.name.split('_')[2])
                    metrics_data['Mouse Name'].append(df.name.split('_')[3])
                    metrics_data['Timepoint'].append(df.name.split('_')[4])
                    metrics_data['Frequency (Hz)'].append(freq)
                    metrics_data['dB Level'].append(db)
                    metrics_data['Wave I amplitude (P1-T1) (μV)'].append(first_peak_amplitude)
                    metrics_data['Latency to First Peak (ms)'].append(latency_to_first_peak)
                    metrics_data['Amplitude Ratio (Peak1/Peak4)'].append(amplitude_ratio)
                    metrics_data['Estimated Threshold'].append(threshold)

    return metrics_data

def plot_waves_stacked(freq):
    fig = go.Figure()
    # Get unique dB levels and color palette
    unique_dbs = sorted(df['Level(dB)'].unique())
    num_dbs = len(unique_dbs)
    vertical_spacing = 25 / num_dbs
    db_offsets = {db: y_min + i * vertical_spacing for i, db in enumerate(unique_dbs)}
    glasbey_colors = cc.glasbey[:num_dbs]

    db_levels = sorted(unique_dbs, reverse=True)
    max_db = db_levels[0]

    # first plot the stacked waves for the user to view
    for i, db in enumerate(db_levels):
        khz = df[(df['Freq(Hz)'] == freq) & (df['Level(dB)'] == db)]

        if not khz.empty:
            index = khz.index.values[-1]
            final = df.loc[index, '0':].dropna()
            final = pd.to_numeric(final, errors='coerce')
            final = interpolate_and_smooth(final)

            # Normalize the waveform
            if db == max_db:
                max_value = np.max(np.abs(final))
            final_normalized = final / max_value
            # Apply vertical offset
            y_values = final_normalized + db_offsets[db]

            # Plot the waveform
            color_scale = glasbey_colors[i]
            fig.add_trace(go.Scatter(x=np.linspace(0, time_scale, len(y_values)),
                             y=y_values,
                             mode='lines',
                             name=f'{int(db)} dB',
                             line=dict(color=color_scale)))

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
    fig.update_layout(title=dict(text=f'{df.name.split("/")[-1]} - Frequency: {freq} Hz', font = dict(family='Helvetica', size=15)),
              xaxis_title='Time (ms)',
              yaxis_title='Voltage (μV)',
              width=400,
              height=700,
              yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
              xaxis=dict(showgrid=False, zeroline=False))

    buf = io.BytesIO()
    pio.write_image(fig, buf)
    img = Image.open(buf)
    img.show()
    #fig.show()

    # creating user-interactive buttons for selecting hearing threshold manually
    app = tkinter.Tk() # define this each time so that it opens a separate window
    # Create a set for all clicked buttons (set prevents duplication)
    clicked = set()
    def close_app():
        app.destroy()


    def create_buttons(db_levels, glasbey_colors):
        button = tkinter.Button(app,
                                    text='No threshold',
                                    command=lambda d=100: clicked.add(d),
                                    activebackground='white')
        button.pack()
        # Create a button for each db
        for i, db in enumerate(db_levels):
            button = tkinter.Button(app,
                                    text=db,
                                    command=lambda d=db: clicked.add(d),
                                    fg=glasbey_colors[i],
                                    activebackground='white')
            button.pack()
        button = tkinter.Button(app,
                                text='ACCEPT',
                                command=close_app,
                                activebackground='white')
        button.pack()
    create_buttons(db_levels,glasbey_colors)
    # Now we enter to event loop -> the program is running
    app.mainloop()

    if len(clicked) == 0:
        threshold = 100
    else:
        clicked = list(clicked)
        threshold = clicked[-1] # last value, in case user changed their mind

    img.close()

    for i, db in enumerate(db_levels):
        try:
            khz = df[(df['Freq(Hz)'] == freq) & (df['Level(dB)'] == db)]

            if not khz.empty:
                index = khz.index.values[-1]
                final = df.loc[index, '0':].dropna()
                final = pd.to_numeric(final, errors='coerce')
                final = interpolate_and_smooth(final)

                # Normalize the waveform
                if db == max_db:
                    max_value = np.max(np.abs(final))
                final_normalized = final / max_value
                # Apply vertical offset
                y_values = final_normalized + db_offsets[db]

                # Plot the waveform
                color_scale = glasbey_colors[i]
                fig.add_trace(go.Scatter(x=np.linspace(0, time_scale, len(y_values)),
                                 y=y_values,
                                 mode='lines',
                                 name=f'{int(db)} dB',
                                 line=dict(color=color_scale)))

                if db == threshold:
                    fig.add_trace(go.Scatter(x=np.linspace(0, time_scale, len(y_values)),
                                    y=y_values,
                                    mode='lines',
                                    name=f'Thresh: {int(db)} dB',
                                    line=dict(color='black', width=5),
                                    showlegend=True))

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
                  xaxis_title='Time (ms)',
                  yaxis_title='Voltage (μV)',
                  width=400,
                  height=700,
                  yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                  xaxis=dict(showgrid=False, zeroline=False))

        khz = df['Freq(Hz)'] == freq
        fig.write_image(os.path.join(saveDir,f'stacked_waves_{freq}Hz_{df.name}.pdf'))
        fig.write_image(os.path.join(saveDir, f'stacked_waves_{freq}Hz_{df.name}.jpg'))

    return threshold

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


peak_finding_model = CNN()
model_loader = torch.load('./models/waveI_cnn_model.pth')
peak_finding_model.load_state_dict(model_loader)
peak_finding_model.eval()
# Open UI for user to select files to be analysed
files = fd.askopenfilenames(title="Select .arf files | Sélectionnez les fichiers .arf",filetypes = (("ABR sessions","*PT* *CLICK*"),("Pure-tone sessions only","*PT*"),("Click sessions only","*CLICK*"),("All files","*")))
saveRoot = fd.askdirectory(title='Select save directory | Sélectionnez le répertoire de sauvegarde')
# create empty dictionaries for storing data across sessions
click_metrics_data = {'File Name': [], 'Date': [], 'Session Type': [], 'Ear': [], 'Mouse Name': [], 'Timepoint': [],
                      'Frequency (Hz)': [], 'dB Level': [], 'Wave I amplitude (P1-T1) (μV)': [],
                    'Latency to First Peak (ms)': [], 'Amplitude Ratio (Peak1/Peak4)': [], 'Estimated Threshold': []}
pt_metrics_data = {'File Name': [], 'Date': [], 'Session Type': [], 'Ear': [], 'Mouse Name': [], 'Timepoint': [],
                   'Frequency (Hz)': [], 'dB Level': [], 'Wave I amplitude (P1-T1) (μV)': [],
                    'Latency to First Peak (ms)': [], 'Amplitude Ratio (Peak1/Peak4)': [], 'Estimated Threshold': []}
counter = 1
for file in files:
    if  "CLICK" in (file.split('/')[-1]).upper():
        click = True
    else:
        click = False
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
                if not click:
                    freq = rec['Var1']
                    db = rec['Var2']
                else:
                    freq = 'Click'
                    db = rec['Var1']
                    # Construct row
                wave_cols = list(enumerate(rec['data']))
                wave_data = {f'{i}': v * 1e6 for i, v in wave_cols}
                row = {'Freq(Hz)': freq, 'Level(dB)': db, **wave_data}
                rows.append(row)

        # Make dataframe with all data from session
        df = pd.DataFrame(rows)
        df.name = file.split('/')[-1][0:-4]
        db_column = 'Level(dB)'

        # Create separate folder for each session
        saveDir = os.path.join(saveRoot,df.name)
        if not os.path.exists(saveDir):
            os.mkdir(saveDir)


        # Get distinct frequency and dB level values used in session
        distinct_freqs = sorted(pd.concat([df['Freq(Hz)']]).unique())
        distinct_dbs = sorted(pd.concat([df['Level(dB)']]).unique())

        # Sanity check that session hasn't been mis-labelled
        if click and len(distinct_freqs) > 1:
            print(f'{file.split('/')[-1]} mis-labelled as CLICK, skipping...')
            continue

        time_scale = 10.0 # 10ms recording by default
        y_min = -5.0 # default scale is -5 to +5 uV
        y_max = 5.0
        all_thresholds = []
        for idx,freq in enumerate(distinct_freqs):
            # Plot all waves at single frequency offset in y (to appear stacked)
            threshold = plot_waves_stacked(freq) # use this to calculate threshold as well
            all_thresholds.append(threshold)
            # Plot all waves at a single frequency
            plot_waves_single_frequency(df, freq, y_min, y_max,threshold)
            # Plot all waves at single frequency as 3D surface (saves as html to open in browser)
            plot_3d_surface(df, freq, y_min, y_max,threshold)
            #for db in distinct_dbs:
                ## Plot all waves at each frequency and dB
                #plot_waves_single_tuple(freq, db, y_min, y_max)
        # Extract important metrics
        metrics_data = make_metrics_table(df, distinct_freqs, distinct_dbs,all_thresholds)
        # Append to metrics table for all click/PT sessions
        for key,val in dict.items(metrics_data):
            if click:
                click_metrics_data[key] = click_metrics_data[key] + val
            else:
                pt_metrics_data[key] = pt_metrics_data[key] + val
        metrics_data = pd.DataFrame(metrics_data)
        metrics_data.to_csv(os.path.join(saveDir,'dataTable.csv'))
    counter = counter + 1


