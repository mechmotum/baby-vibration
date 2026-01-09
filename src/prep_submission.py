"""
Converts all png figures in the main.tex into jpg files and then drops all jpg
files into the submission directory.

Copies the main.tex and reference.bib to the submission folder and then makes
the edit to source the figures from top level and use the jpg files.

"""
import os
import subprocess
import shutil

image_filenames_in_main = [
    'Maxicosi_sensors_UA.jpg',
    'RW_UA.jpg',
    'SeatBotacc_ver-bandwidth-dist.png',
    'SeatBotacc_ver-bicycle-type-compare.png',
    'SeatBotacc_ver-peak-freq-dist.png',
    'SeatBotacc_ver-rms-bicycle-compare-all.png',
    'SeatBotacc_ver-rms-comfort-bicycle-compare-all.png',
    'SeatBotacc_ver-rms-comfort-stroller-compare-all.png',
    'SeatBotacc_ver-rms-stroller-compare-all.png',
    'SeatBotacc_ver-spectra-compare.png',
    'SeatBotacc_ver-spectra-compare.png',
    'SeatBotacc_ver-stroller-type-compare.png',
    'equipment.png',
    'session001-t2-aula-stroller-maxicosi-cot-0-SeatBotacc_ver-rep0.png',
    'session004-t0-pave-stroller-maxicosi-cot-0-SeatBotacc_ver-rep0.png',
    'session015.png',
    'stroller-dummy-camera.jpg',
    'surfaces.png',
]

if not os.path.exists('submission'):
    os.mkdir('submission')

for filename in image_filenames_in_main:
    if filename.endswith('.png'):
        name, ext = filename.split('.')
        subprocess.call(['convert',
                         'fig/' + filename,
                         '-quality', '90',
                         'submission/' + name + '.jpg'])
    else:
        shutil.copy('fig/' + filename, 'submission/')

shutil.copy('main.tex', 'submission/')
shutil.copy('reference.bib', 'submission/')

subprocess.call([
    'sed',
    '-i',
    r"s/fig\///g",
    r'submission/main.tex',
])

subprocess.call([
    'sed',
    '-i',
    r"s/png/jpg/g",
    r'submission/main.tex',
])
