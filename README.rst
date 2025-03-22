===================================================================================
Vibration Characterisation of Strollers and Cargo Bicycles for Transporting Infants
===================================================================================

This repository houses the analysis and paper that investigates the vibrations
experienced by infants (0-9 months old) when carried in cargo bicycles and
strollers.

Preliminary results can be viewed at:

https://mechmotum.github.io/baby-vibration/

Preprint "Vibration Characterisation of Strollers and Cargo Bicycles for
Transporting Infants" by Gabriele Dell'Orto, Brecht Daams, Riender Happee,
Georgios Papaioannou, Arjo Loeve, Jesper Meijerink, Thomas Valk, and Jason K.
Moore can be viewed at:

https://doi.org/10.31224/4415

Private links to the related editable documents:

- Experiment paper on Overleaf: https://www.overleaf.com/4478681329cgkzxtynvvzz#e49cd7
- Lit Study on Google Docs: https://drive.google.com/drive/folders/1y4d1djsXC8zveT1g1Zd_FxibJYxMT0O1

License
=======

The paper content (text, figures, etc.) and the generated paper itself are
licensed under the Creative Commons BY-4.0 license. The computer program source
code is licensed under the MIT license. The data is available as public domain
with no copyright restrictions. If you make use of the copyrighted material use
a full citation to the paper to acknowledge the use and include the relevant
license(s). If you make use of the data, we kindly ask that you also cite us.

Build Instructions
==================

Create a ``config.yml`` file in the repository directory with these contents::

   data-directory: /path/to/Trillingen-Project_Dec2024/raw_data

or on Windows::

   data-directory: C:\path\to\Trillingen-Project_Dec2024\raw_data

**Do not commit the configuration file to Git.**

The ``Trillingen-Project_Dec2024`` is currently a private mountable directory
share on TU Delft network. The data will be shared in a public repository in
the future.

Install conda then create and activate the environment::

   conda env create -f baby-vibration-env.yml
   conda activate baby-vibration

Usage
=====

Run a sample processing of a single session::

   python src/data.py

Or name a specific session and trial and show sample plots::

   python src/data.py -s session020 -t klinkers -r 200 --plot

Process sessions 0 through 3 for specific signal at sample rate 200::

   python src/process.py 0 3 SeatBotacc_ver 200

Process all sessions in ``data/sessions.yml`` with::

   python src/run.py

Data Description
================

We mount five Shimmer_ IMUs on the vehicle that collect data simultaneously
during different trials.

.. _Shimmer: https://www.shimmersensing.com/

Scenario
   We test a combination of vehicle, infant seat, infant dummy, road surface
   (including shock), and speed. A unique combination of these factors is
   called a "scenario".
Session
   A "session" is a continuous collection of data from a set of Shimmer IMUs
   which contains one or more scenarios. The IMU's internal clocks are time
   synchronized on the base station, attached to the vehicle, and each IMU is
   started. After a time period of collecting data of possibly multiple trials
   and calibrations the IMUs are stopped. Each IMU produces a single
   acquisition CSV file for that session.
Trial
   A trial is defined as a continuous time segment selected from a session that
   represents a specific scenario. The trial may be a static calibration
   period, a time synchronization motion period, or a constant speed period of
   vehicle motion.
Repetition
   We split trials into shorter segments to have an average trial segment
   duration of about 20 seconds.

Shimmer IMU names:

- ``BotTrike``:  a sensor placed underneath the infant seat on the frame
  structure of the vehicle
- ``FrontWHeel`` : attached to the axle of the front wheel (closest possible
  point to the ground)
- ``RearWheel`` : attached to the rotating rear wheel (used for speed
  measurements)
- ``SeatBot`` : attached to the seat just under the infant's buttocks
- ``SeatHead`` : attached to the seat just under the back of the infant's head

Vehicle name:

- ``bugaboo``: Bugaboo Fox 5 stroller
- ``greenmachine``: an old-style (~70's) cot for very young infants
- ``maxicosi``: Maxi-Cosi Street Plus stroller
- ``oldrusty``: an old-style (~70's) seat for older infants
- ``keiler``: Keiler cargo tricycle (tadpole wheel arrangement with some extra
  mass added to represent propulsion motor and battery)
- ``urbanarrow``: Urban Arrow cargo electric bicycle
- ``yoyo``: Stokke BABYZEN YOYO 0+ stroller

The Shimmer IMUs are set to +/- 16 g and +/- 2000 deg/s. The values are
recorded to 16 bit floating point precision other than the time stamp which is
a 16 bit positive integer. The IMUs are placed in the base station and their
clocks are synchronized with each other. This means we assume that the time
stamp values represents the same real time value in each IMU. The following
column order is consistent among the files.

- ``S_SENSORNAME_Timestamp_Unix_CAL`` : milliseconds since epoch
- ``S_SENSORNAME_Accel_WR_X_CAL``: m/s/s
- ``S_SENSORNAME_Accel_WR_Y_CAL``: m/s/s
- ``S_SENSORNAME_Accel_WR_Z_CAL``: m/s/s
- ``S_SENSORNAME_Gyro_X_CAL``: deg/s
- ``S_SENSORNAME_Gyro_Y_CAL``: deg/s
- ``S_SENSORNAME_Gyro_Z_CAL``: deg/s

Data Processing
===============

#. Load each acquisition file into a Pandas sparse data frame with the
   time stamp as the index.
#. Combine all sensor data frames from a single session into a single data
   frame. These can be up to 2 Gb in size. NaNs are used to represent
   mismatches in the sample times.
#. Extract the trial start/stop times for trials from the manually created CSV
   files for each session.
#. Use a period of no motion, "static", in the session to find the direction of
   gravity in all sensors assuming that one axis of each sensor is aligned with
   the lateral axis of the vehicle.
#. Calculate the vibration dose value (VDV) from the unfiltered time series for
   the first 10 seconds of each repitition, skipping shock data.
#. Down sample the time series from ~900 Hz to 400 Hz.
#. Set any values greater than +/-16 g or +/-2000 deg/s to those maximum
   values, as the sensors are not valid at higher values.
#. Low pass filter the time series at 120 Hz (ISO 2631-1 recommended 1.5*80 Hz)
   with a 2nd Order zero-lag Butterworth filter.
#. Calculate linear speed of the vehicle using wheel radius and rear wheel rate
   gyro. Calculate the mean speed and standard deviation per trial.
#. Calculate the crest factor from unweighted maximum and unweighted RMS.
#. Calculate the bandwidth containing 80% of the spectrum area from unweighted
   frequency spectrum.
#. Calculate the frequency spectrum of the buttocks sensor's vertical
   acceleration component for health assessment and magnitude of acceleration
   for comfort assessment.
#. Apply the ISO 2631-1 spectrum weights for health and comfort assessments.
#. Smooth the frequency spectrums with low pass filter.
#. Calculate the root mean square (RMS) from the weighted spectrums.
#. Calculate the peak frequency and peak amplitude from the spectrum.

Final data table should have these columns:

- Trial ID
- Vehicle [bugaboo|yoyo|maxicosi|urbanarrow|keiler|greenmachine|oldrusty]
- Vehicle Type [stroller|bicycle]
- Seat Type [cot|seat]
- Baby Age [month] [0|3|9]
- Baby Mass [kg] [3.48|5.9|8.9]
- Surface [aula|stoeptegels|tarmac|klinkers|pave]
- Duration [s]
- Mean of Speed [m/s]
- Standard Deviation of Speed [m/s]
- Speed Category [5 kph|12 kph|20 kph|25 kph]
- Peak Frequency [Hz]
- Peak Spectrum Amplitude [m/s/s]
- 80% Bandwidth [Hz]
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
- SENSOR_N lateral acceleration VDV [m/s/s]
- SENSOR_N longitudinal acceleration VDV [m/s/s]
- SENSOR_N vertical acceleration VDV [m/s/s]
- SENSOR_N acceleration magnitude VDV [m/s/s]
- SENSOR_N pitch angular rate VDV [deg/s]
- SENSOR_N yaw angular rate VDV [deg/s]
- SENSOR_N roll angular rate VDV [deg/s]
- SENSOR_N angular rate magnitude VDV [deg/s]
