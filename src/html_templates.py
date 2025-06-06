H1 = '<h1>{}</h1>'
H2 = '<h2>{}</h2>'
H3 = '<h3>{}</h3>'
H4 = '<h4>{}</h4>'
HR = '<hr>'
IMG = '<div class="w-75 p-3"><img src="fig/{}/{}" class="img-fluid"></img></div>'
P = '<p>{}</p>'

INDEX = """\
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Infant Vehicle Transport Vibration Results</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-QWTKZyjpPEjISv5WaRU9OFeRpok6YctnYmDr5pNlyT2bRjXh0JMhjY6hW+ALEwIH" crossorigin="anonymous">
  </head>
  <body>

  <div class="container">
  <h1>Infant Vehicle Transport Vibration Results</h1>
  <hr>
  <div class="alert alert-danger" role="alert">
    <p>
      Warning: These results are a work-in-progress, we caution reliance on
      them until a supporting paper is published. See the latest preprint for
      more context: <a
      href="https://doi.org/10.31224/4415">doi.org/10.31224/4415</a>.
    </p>
  </div>
  <p>
    Last updated: {date}
    <br>
    Generated from Git hash: {githash}
    <br>
    License: CC-BY 4.0
  </p>
  <p>
    This page examines the signal: <strong>{signal}</strong>.
  </p>

  <h1>Table of Contents</h1>
  <ol>
    <li><a href="#mean_table">Scenario Mean Table</a></li>
    <li><a href="#boxp">Comparison Plots</a></li>
    <li><a href="#shock">Shock Test</a></li>
    <li><a href="#iso">ISO 2631-1 Weights</a></li>
    <li><a href="#statistics">Statistical Regression Results</a></li>
    <li><a href="#sess">Sessions Segmented into Trials</a></li>
    <li><a href="#trial_table">Table of All Repetitions</a></li>
    <li><a href="#trial">Seat Pan Vertical Acceleration Time Histories</a></li>
    <li><a href="#spec">Seat Pan Vertical Acceleration Spectrums</a></li>
    <li><a href="#srot">Sensor Rotation Checks</a></li>
    <li><a href="#sync">Sensor Time Synchronization Checks</a></li>
  </ol>

  <h1 id="mean_table">Mean Over Scenarios for Signal: {signal}</h1>
  <hr>
  <p>
    Counts and mean values for each scenario, i.e. combination of vehicle,
    seat, baby mass, road surface, and speed.
    Download summary tables for the strollers and bicycles in excel format: <a
    href='stroller-summary.xlsx'>stroller-summary.xlsx</a> and <a
    href='bicycle-summary.xlsx'>bicycle-summary.xlsx</a>.
  </p>
{mean_table}

  <h1 id="boxp">Comparison Plots of the Signal: {signal}</h1>
  <hr>
  <p>
    Descriptive statistical plots of the data.
  </p>
{boxp_html}

{shock_html}

  <h1 id="iso">ISO 2631-1 Weights</h1>
  <hr>
  <p>
    Plots of the ISO 2631-1 filter weights versus frequency we apply to the
    data.
  </p>

  <div class="w-75 p-3">
    <img src='fig/iso-filter-weights-01.png' class="img-fluid"></img>
  </div>
  <br>
  <div class="w-75 p-3">
  <img src='fig/iso-filter-weights-02.png' class="img-fluid"></img>
  </div>

  <h1 id="statistics">Statistical Regression Results</h1>
  <hr>
  <p>
    Ordinary linear least squares regression models fit to the data.
  </p>

  <h2>Strollers</h2>
{stroller_stats}

  <h3>Stroller Comparisons</h3>
{stroller_comp}

  <h2>Cargo Bicycles</h2>
{bicycle_stats}

  <h3>Cargo Bicycle Comparisons</h3>
{bicycle_comp}

  <h1 id="sess">Sessions Segmented into Trials</h1>
  <hr>
  <p>This section shows how the sessions are segmented into trials.</p>
{sess_html}

  <h1 id="trial_table">Repetitions</h1>
  <hr>
  <p>
  This lists all of the repetitions in long form (tidy data). Download as an
  excel spreadsheet: <a href='all-reps.xlsx'>all-reps.xlsx</a>.
  </p>
{trial_table}

  <h1 id="trial">{signal} Repetition Time Histories</h1>
  <hr>
  <p>
    The time history of the signal of each trial broken into repetitions
    selected from each scenario.
  </p>
{trial_html}

  <h1 id="spec">{signal} Amplitude Spectra (with ISO 2631-1 weighting)</h1>
  <hr>
{spec_html}

  <h1 id="srot">Sensor Rotation Check</h1>
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

  <h1 id="sync">Time Synchronization Check</h1>
  <hr>
{sync_html}

  </div>  <!--container-->
  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.min.js" integrity="sha384-0pUGZvbkm6XF6gxjEnlmuGrJXVbNuzT9qBBavbLwCsOGabYfZo0T0to5eqruptLy" crossorigin="anonymous"></script>
  </body>
</html>"""
