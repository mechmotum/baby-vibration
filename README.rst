================================================
Vibration of Babies in Cargo Bikes and Strollers
================================================

This repository houses the analysis and paper that investigates the vibrations
experienced by babies (0-9 months old) when carried in cargo bicycles and
strollers.

License
=======

The paper content (text, figures, etc.) and the generated paper itself are
licensed under the Creative Commons BY-4.0 license. The computer source code is
licensed under the MIT license. The data is available as public domain with no
copyright restrictions. If you make use of the copyrighted material use a full
citation to the paper to acknowledge the use and include the relevant
license(s).

Build Instructions
==================

Create a ``config.yml`` file in the repository directory with these contents::

   data-directory: /path/to/Trillingen-Project_Dec2024

The ``Trillingen-Project_Dec2024`` is a mountable directory share on TU Delft
currently.

Install conda then::

   conda env create -f baby-vibration-env.yml
   conda activate baby-vibration

Data Description
================

We mount five Shimmer IMUs on the vehicle. Each Shimmer produces a single CSV
file for collection time period.

There is a folder with this naming convention::

   ├── 2024-11-28_10.43.58_maxicos_1_SH_SD_Session1
   │   └── maxicos_1_SH_Session1_S_SeatHead_Calibrated_SD.csv
   ├── 2024-11-28_10.44.49_maxicos_1_FW_SD_Session1
   │   └── maxicos_1_FW_Session1_S_FrontWheel_Calibrated_SD.csv
   ├── 2024-11-28_10.45.23_maxicos_1_RW_SD_Session1
   │   └── maxicos_1_RW_Session1_S_RearWheel_Calibrated_SD.csv
   ├── 2024-11-28_10.45.57_maxicos_1_SB_SD_Session1
   │   └── maxicos_1_SB_Session1_S_SeatBot_Calibrated_SD.csv
   ├── 2024-11-28_10.46.45_maxicos_1_BT_SD_Session1
   │   └── maxicos_1_BT_Session1_S_BotTrike_Calibrated_SD.csv
   ├── 2024-11-28_12.14.31_YOYO_2_SH_SD_Session1
   │   └── YOYO_2_SH_Session1_S_SeatHead_Calibrated_SD.csv
   ├── 2024-11-28_12.15.11_YOYO_2_SB_SD_Session1
   │   └── YOYO_2_SB_Session1_S_SeatBot_Calibrated_SD.csv
   ├── 2024-11-28_12.15.32_YOYO_2_FW_SD_Session1
   │   └── YOYO_2_FW_Session1_S_FrontWheel_Calibrated_SD.csv
   ├── 2024-11-28_12.15.52_YOYO_2_RW_SD_Session1
   │   └── YOYO_2_RW_Session1_S_RearWheel_Calibrated_SD.csv
   ├── 2024-11-28_12.18.35_YOYO_2_BT_SD_Session1
   │   └── YOYO_2_BT_Session1_S_BotTrike_Calibrated_SD.csv
   ├── 2024-11-28_13.13.21_YOYO_LR3_SB_SD_Session1
   │   └── YOYO_LR3_SB_Session1_S_SeatBot_Calibrated_SD.csv
   ├── 2024-11-28_13.13.40_YOYO_LR3_BT_SD_Session1
   │   └── YOYO_LR3_BT_Session1_S_BotTrike_Calibrated_SD.csv
   ├── 2024-11-28_13.13.59_YOYO_LR3_RW_SD_Session1
   │   └── YOYO_LR3_RW_Session1_S_RearWheel_Calibrated_SD.csv
   ├── 2024-11-28_13.14.20_YOYO_LR3_FW_SD_Session1
   │   └── YOYO_LR3_FW_Session1_S_FrontWheel_Calibrated_SD.csv
   ├── 2024-11-28_13.14.47_YOYO_LR3_SH_SD_Session1
   │   └── YOYO_LR3_SH_Session1_S_SeatHead_Calibrated_SD.csv
   ├── 2024-11-28_15.38.52_Maxicos_3_SH_SD_Session1
   │   └── Maxicos_3_SH_Session1_S_SeatHead_Calibrated_SD.csv
   ├── 2024-11-28_15.39.20_Maxicos_3_FW_SD_Session1
   │   └── Maxicos_3_FW_Session1_S_FrontWheel_Calibrated_SD.csv
   ├── 2024-11-28_15.40.51_Maxicos_3_SB_SD_Session1
   │   └── Maxicos_3_SB_Session1_S_SeatBot_Calibrated_SD.csv
   ├── 2024-11-28_15.41.18_Maxicos_3_RW_SD_Session1
   │   └── Maxicos_3_RW_Session1_S_RearWheel_Calibrated_SD.csv
   ├── 2024-11-28_15.42.14_Maxicos_3_BT_SD_Session1
   │   └── Maxicos_3_BT_Session1_S_BotTrike_Calibrated_SD.csv
   ├── 2024-11-29_17.04.45_Maxicos_6_SB_SD_Session1
   │   └── Maxicos_6_SB_Session1_S_SeatBot_Calibrated_SD.csv
   ├── 2024-11-29_17.05.14_Maxicos_6_FW_SD_Session1
   │   └── Maxicos_6_FW_Session1_S_FrontWheel_Calibrated_SD.csv
   ├── 2024-11-29_17.05.53_Maxicos_6_SH_SD_Session1
   │   └── Maxicos_6_SH_Session1_S_SeatHead_Calibrated_SD.csv
   ├── 2024-11-29_17.06.34_Maxicos_6_RW_SD_Session1
   │   └── Maxicos_6_RW_Session1_S_RearWheel_Calibrated_SD.csv
   └── 2024-11-29_17.07.03_Maxicos_6_BT_SD_Session1
      └── Maxicos_6_BT_Session1_S_BotTrike_Calibrated_SD.csv

Shimmer IMU names:

- ``BotTrike``:  a sensor placed underneath the baby seat, for the strollers
- ``FrontWHeel`` : attached to the axle of the front wheel (closest possible
  point to the ground)
- ``RearWheel`` : attached to the rotating rear wheel (used for speed
  measurements)
- ``SeatBot`` : attached to the seat just under the baby's buttocks
- ``SeatHead`` : attached to the seat just under the back of the baby's head

Vehicle name:

- ``Maxicos``: Maxicosi stroller
- ``YOYO``: Yoyo stroller

The shimmers are set to +/- 16 g and +/- 2000 deg/s.

The Shimmers are placed in the base station and their clocks are sycronized
with each other. This means we assume that the time stamp values represents the
same real time value in each shimmer.

The first four lines of a raw Shimmer file look like::

   "sep=,"
   S_SeatHead_Timestamp_Unix_CAL,S_SeatHead_Accel_WR_X_CAL,S_SeatHead_Accel_WR_Y_CAL,S_SeatHead_Accel_WR_Z_CAL,S_SeatHead_Gyro_X_CAL,S_SeatHead_Gyro_Y_CAL,S_SeatHead_Gyro_Z_CAL,
   ms,m/(s^2),m/(s^2),m/(s^2),deg/s,deg/s,deg/s,
   1.7327890275714417E12,-0.8421052631578947,-0.6889952153110047,10.488038277511961,3.4756097560975614,-0.6097560975609757,-1.9512195121951221,

- ``Timestamp_Unix`` : milliseconds since epoch
