import numpy as np
import os
import subprocess

surface_index=10
num_polymers = 92
num_bins = 100
pmf_data = np.zeros((num_polymers, num_bins))
row_index=0
main_dir=os.getcwd()
#for surface_index in range(1,surface_num+1):
for polymer_index in range(101):
    if 91 <= polymer_index <= 99:
        continue
    working_dir = os.path.join(os.getcwd(), f'surface{surface_index}',f's{surface_index}p{polymer_index}')
    dir2 = os.path.join(working_dir, f'COLVAR-K500_s{surface_index}p{polymer_index}/metadata')
    wham_dir = os.path.join(os.getcwd(), 'wham', 'wham')
   # print(wham_dir)
    os.chdir(wham_dir)
#    ./wham 4.7 10.4 100 3e-08 120 0 {dir2} pmf_out.xvg
#    subprocess.run('rm pmf_out.xvg', shell=True)
    wham_command = f"./wham 4.7 10.4 100 3e-08 120 0 {dir2} pmf_out_s{surface_index}p{polymer_index}.xvg"
    subprocess.run(wham_command, shell=True)
#    if polymer_index == 1:
#    subprocess.run('cp pmf_out.xvg out.xvg', shell=True)
    pmf_output_file = f'pmf_out_s{surface_index}p{polymer_index}.xvg'
    pmf_values = np.full(num_bins, np.nan)
    with open(pmf_output_file, 'r') as pmf_file:
         bin_index = 0
         for line in pmf_file:
             if line.strip() and not line.startswith(('#', '@')):
                columns = line.split()
                pmf_values[bin_index] = float(columns[1])
                bin_index += 1
                if bin_index == num_bins:
                   break
#    pmf_values.reshape(1,-1)
    pmf_data[row_index,:] = pmf_values 
    row_index += 1
    os.chdir('../../')
np.save('pmf_surface10.npy', pmf_data)
#os.chdir('../../')


