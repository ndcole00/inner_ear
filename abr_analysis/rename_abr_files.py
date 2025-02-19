import os

fileDirectory = '/home/nc/Documents/Test Trauma sonore 98 dB'
files = [f for f in os.listdir(fileDirectory) if os.path.isfile(os.path.join(fileDirectory, f))]

os.chdir(fileDirectory)

for ff in files:
    filename = ff.split('_')
    date = filename[0].split('-')
    newdate = date[2][2:4] + date[1] + date[0]
    if filename[3] == 'J0':
        filename[3] = 'PE'
    else:
        filename[3] = filename[3].replace('J','D')
    if len(filename[1]) > 5: # sometimes he uses the wrong order
        new_filename = newdate + '_' + filename[2] + '_' + filename[5][0:2] + '_' + filename[4] + '_' + filename [3] + '.arf'
    else:
        new_filename = newdate + '_' + filename[1] + '_' + filename[5][0:2] + '_' + filename[4] + '_' + filename [3] + '.arf'
    os.rename(ff,new_filename)

