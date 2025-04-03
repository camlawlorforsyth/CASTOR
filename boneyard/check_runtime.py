
import os
import numpy as np

# from astropy.table import Table
# table = Table.read('tools/subIDs.fits')
# subIDs = table['subID'].data

inDir = 'SKIRT/SKIRT_output_quenched/'
subIDs = np.sort(np.array(os.listdir(inDir)).astype(int))
subIDs = subIDs[(subIDs < 14) | (subIDs > 14)] # subID 14's SKIRT run failed


time = np.full(subIDs.shape, -1)
memory = np.full(subIDs.shape, -1.)
for i, subID in enumerate(subIDs) :
    
    # read the file and populate values
    with open(inDir + '{}/log.log'.format(subID)) as file :
        contents = file.readlines()
    
    time[i] = contents[-2].split(' ')[-5] # in seconds, using only 2 cores
    memory[i] = contents[-1].split(' ')[-3] # in GB

# print(np.percentile(time, [2.5, 16, 50, 84, 97.5]))
# print(np.percentile(memory, [2.5, 16, 50, 84, 97.5]))

hours = time/3600/4 # hours, assuming a scaling if using 8 cores 
# print(hours)
total_hours = np.sum(hours)
total_days = total_hours/24
total_weeks = total_days/7
# print(total_weeks)

# if we have 6 computers running this full time, then we expect to take
CANFAR_expected_days = total_weeks/6*7
print(CANFAR_expected_days)
# assuming we only use 8 cores and 6 GB of memory

# what happens if we have double the number of cores, like 16, available?

# table = Table([subIDs, time, memory], names=('subID', 'time (s)', 'memory (GB)'))
# table.pprint(max_lines=-1)
