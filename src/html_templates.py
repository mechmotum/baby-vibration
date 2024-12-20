IMG = '<img src="fig/{}/{}"></img>'
H1 = '<h1>{}</h1>'
H2 = '<h2>{}</h2>'
H3 = '<h3>{}</h3>'
HR = '<hr>'
P = '<p>{}</p>'

INDEX = """
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>Baby Vehicle Vibration Results</title>
  </head>
  <body>

  <h1>Baby Vehicle Vibration Results</h1>
  <hr>
  <p>
    <strong>Warning: These results are preliminary, do not rely on them until a
    supporting paper is published.</strong>
  </p>
  <p>
    This results page examines the signal: <strong>{signal}</strong>.
  </p>

  <h1>Duration Weighted Mean of the RMS of {signal}</h1>
  <hr>
{mean_table}

  <h1>Comparison Plots: RMS of {signal} </h1>
  <hr>
{boxp_html}

  <h1>Sessions Segmented into Trials</h1>
  <p>This section shows how the sessions are segmented into trials.</p>
{sess_html}

  <h1>Trials</h1>
  <hr>
  <p>This lists all of the trials in long form data forma (tidy data).</p>
{trial_table}

  <h1>ISO 2631-1 Weights</h1>
  <hr>
  <p>
    Plots of the filter weights versus frequency we apply to the data.
  </p>
  <img src='fig/iso-filter-weights-01.png'</img>
  <br>
  <img src='fig/iso-filter-weights-02.png'</img>

  <h1>Seat Pan Vertical Acceleration Spectrums</h1>
  <hr>
{spec_html}

  <h1>Sensor Rotation Check</h1>
  <hr>
  <p>
    These plots give the information necessary to determine if the sensor
    orientations are working correctly. These are time histories of the
    "static" event (no motion) in each session. The left column shows the raw
    accelerometer data for each sensor and each body fixed sensor axis. If the
    background is shaded grey, that axis is the sensor's axis that aligned with
    the vehicle's lateral axis (pitch axis) and the title on that subplot gives
    the axis letter (x, y, or z) and the sign (axis points to the left - or
    right +). The column on the right shows the accelerometer time histories
    projected on the vehicle's body fixed axes: longitudinal, lateral, and
    vertical. The vertical axis should show a positive ~10 m/s/s value and the
    other two should axes should show approximately 0 m/s/s if the rotations
    are applied correctly. This indicates that the vertical axes has been
    aligned with gravity.
  </p>
{srot_html}

  <h1>Seat Pan Vertical Acceleration Time Histories</h1>
  <hr>
  <p>
  </p>
{trial_html}

  </body>
</html>
"""
