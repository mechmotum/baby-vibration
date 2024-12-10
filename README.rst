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

Session
   A "session" is a continuous collection of data from a set of Shimmer IMUs.
   The IMU's internal clocks are time synchronized on the base station,
   attached to the vehicle, and each IMU is started. After a time period of
   collecting data of possibly multiple trials and calibrations the IMUs are
   stopped. Each IMU produces a single acquisition CSV file for that session.
Trial
   A trial is defined as a continuous time segment selected from a session. The
   trial may be a static calibration period, a time synchronization motion
   period, or a constant speed period of vehicle motion.

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
- ``Bugaboo``: Bugaboo stroller

The shimmers are set to +/- 16 g and +/- 2000 deg/s. The values are recorded to
16 bit floating point precision other than the time stamp which is a 16 bit
integer. The Shimmers are placed in the base station and their clocks are
synchronized with each other. This means we assume that the time stamp values
represents the same real time value in each shimmer. The following column order
is consistent among the files.

- ``S_SENSORNAME_Timestamp_Unix_CAL`` : milliseconds since epoch
- ``S_SENSORNAME_Accel_WR_X_CAL``: m/s/s
- ``S_SENSORNAME_Accel_WR_Y_CAL``: m/s/s
- ``S_SENSORNAME_Accel_WR_Z_CAL``: m/s/s
- ``S_SENSORNAME_Gyro_X_CAL``: deg/s
- ``S_SENSORNAME_Gyro_Y_CAL``: deg/s
- ``S_SENSORNAME_Gyro_Z_CAL``: deg/s

Data Processing
===============

1. Load each acquisition file into a Pandas data frame with the timestamp as the
   index.
2. Combine all sensor data frames from a single session into a single data
   frame. These can be 700 Mb in size. NaNs are used to represent mismatches in
   the sample times.
3. Extract the trial start/stop times from the CSV files for the session.

Resources
=========

- https://github.com/tobias-bettinger/comfpy
