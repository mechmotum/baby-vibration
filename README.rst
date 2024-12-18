================================================
Vibration of Babies in Cargo Bikes and Strollers
================================================

This repository houses the analysis and paper that investigates the vibrations
experienced by babies (0-9 months old) when carried in cargo bicycles and
strollers.

Preliminary results can be viewed at:

https://mechmotum.github.io/baby-vibration/

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

or on Windows::

   data-directory: C:\path\to\Trillingen-Project_Dec2024

The ``Trillingen-Project_Dec2024`` is a mountable directory share on TU Delft
currently.

Install conda then create and activate the environment::

   conda env create -f baby-vibration-env.yml
   conda activate baby-vibration

Usage
=====

Run a sample processing of a single session::

   python src/data.py

Process all sessions with::

   python src/process.py

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

- ``maxicosi``: Maxicosi stroller
- ``yoyo``: Yoyo stroller
- ``bugaboo``: Bugaboo stroller

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

#. Load each acquisition file into a Pandas data frame with the timestamp as the
   index.
#. Combine all sensor data frames from a single session into a single data
   frame. These can be 700 Mb in size. NaNs are used to represent mismatches in
   the sample times.
#. Extract the trial start/stop times from the CSV files for the session.
#. Use a period of no motion, "static", in the session to find the direction of
   gravity in all sensors assuming that one axis of each sensor is aligned with
   the lateral axis of the vehicle.
#. Remove bad data points (random spikes and maybe the repeated values).
#. Low pass filter the time series (not sure if this matters so much if we are
   converting to frequency spectrum and just taking means of things).
#. Calculate linear speed of the vehicle using wheel radius and rear wheel
   rate gyro. Calculate the mean speed per trial.
#. Calculate the frequency spectrum of component and magnitude of acceleration
   and angular rate for all sensors expect the speed sensor and then find the
   RMS of the frequency spectrum. This will give 8 RMS values per trial. These
   should be stored in a tidy data table with each row being a trial.
#. Same as above but apply the ISO 2631 filters before calculating RMS.

Final data table should have these columns:

- Trial ID
- Vehicle [bugaboo|yoyo|maxicosi|urbanarrow|keiler]
- Vehicle Type [stroller|bicycle]
- Baby Age [0|3|9] (implies seat configuration for vehicles with multiple seat
  setups)
- Surface [stoeptegels|tarmac|klinkers]
- Duration [s]
- Mean of Speed [m/s]
- Standard Deviation of Speed [m/s]
- Speed Category [slow|medium|fast]
- SENSOR_N lateral acceleration RMS [m/s/s]
- SENSOR_N longitudinal acceleration RMS [m/s/s]
- SENSOR_N vertical acceleration RMS [m/s/s]
- SENSOR_N acceleration magnitude RMS [m/s/s]
- SENSOR_N pitch angular rate RMS [deg/s]
- SENSOR_N yaw angular rate RMS [deg/s]
- SENSOR_N roll angular rate RMS [deg/s]
- SENSOR_N angular rate magnitude RMS [deg/s]
- SENSOR_N ISO filtered lateral acceleration RMS [m/s/s]
- SENSOR_N ISO filtered longitudinal acceleration RMS [m/s/s]
- SENSOR_N ISO filtered vertical acceleration RMS [m/s/s]
- SENSOR_N ISO filtered acceleration magnitude RMS [m/s/s]
- SENSOR_N ISO filtered pitch angular rate RMS [deg/s]
- SENSOR_N ISO filtered yaw angular rate RMS [deg/s]
- SENSOR_N ISO filtered roll angular rate RMS [deg/s]
- SENSOR_N ISO filtered angular rate magnitude RMS [deg/s]

ISO 2631 Filters
----------------

Code to covert Georgios's CSV filse of the ISO filter tables into CSV files:

.. code:: python

   import numpy as np
   from scipy.io import loadmat
   d = loadmat('filter_ISO_01.mat')
   np.savetxt('data/iso-2631-filter-01.csv', d['filter_ISO_01'], fmt='%1.12f', delimiter=',')
   d = loadmat('ISO_Filters/filter_ISO_02.mat')
   np.savetxt('data/iso-2631-filter-02.csv', d['filter_ISO_02'], fmt='%1.12f', delimiter=',')

The two filter files have amplitude weightings versus frequency from 0 to 400
Hz. The weights must be divided by 1000 to have multiplicative factors from 0
to 1. Different k values are mutiplied to the weightings depending on if you
are seated, standing, supine, etc.

filter_ISO_01::

   frequency_hz,
   vertical_acceleration_z,
   col3,
   translational_acceleration_xy,
   col5,
   motion_sickness_z,
   col7,
   motion_sickness_x,
   motion_sickness_y,
   rotational_speed_xyz,
   col11

filter_ISO_02::

   frequency_hz,
   col2,
   col3,
   rotation_acceleration_xyz,
   col5,col6,col7

Resources
=========

- Partial implementation of vibration comfort filters:
  https://github.com/tobias-bettinger/comfpy
- NFFT implementations: https://github.com/jakevdp/nfft & https://github.com/pyNFFT/pyNFFT
