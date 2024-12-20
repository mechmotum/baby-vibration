IMG = '<img src="fig/{}/{}"></img>'
H1 = '<h1>{}</h1>'
H2 = '<h2>{}</h2>'
H3 = '<h3>{}</h3>'
HR = '<hr>'

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

  <h1>Box Plots of RMS of {signal} </h1>
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
  <img src='fig/iso-filter-weights-02.png'</img>

  <h1>Seat Pan Vertical Acceleration Spectrums</h1>
  <hr>
{spect_html}

  <h1>Sensor Rotations</h1>
  <hr>
{srot_html}

  <h1>Seat Pan Vertical Acceleration Time Histories</h1>
  <hr>
{trial_html}

  </body>
</html>
"""
