#import "@preview/cetz:0.4.2"

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  // Global styles to match your aesthetic
  set-style(
    stroke: (thickness: 1.5pt, cap: "round", join: "round"),
    mark: (fill: black, scale: 0.8)
  )

  // 1. Draw Title
  content((5, 12), text(weight: "bold", size: 12pt, "Typical Neuron Action Potential"))

  // 2. Draw Axes and Grid
  // X-axis (Time in ms)
  line((-0.5, 0), (10, 0), mark: (end: ">", fill:black), stroke: (thickness: 1pt, paint: black))
  content((10.5, 0), "Time (ms)")
  for i in range(1, 6) {
    let x = i * 2 // Scaling factor
    line((x, -0.1), (x, 0.1), stroke: (thickness: 1pt))
    content((x, -0.5), [#i])
  }
  line((0, -0.1), (0, 0.1), stroke: (thickness: 1pt))
  content((0, -0.5), "0")

  // Y-axis (Membrane Potential in mV)
  line((0, -0.5), (0, 10), mark: (end: ">", fill:black), stroke: (thickness: 1pt, paint: black))
  content((0, 10.5), "Membrane Potential (mV)")
  
  // Ticks and horizontal reference lines
  let y-levels = ((-80, 1.3), (-70, 2), (-55, 3.5), (+30, 8)) // (potential, scaled y)
  for level in y-levels {
    let y = level.at(1)
    let label = level.at(0)
    line((-0.1, y), (0.1, y), stroke: (thickness: 1pt))
    content((-1.2, y), [#label])
    
    // Horizontal dashed lines
    if label == -70 or label == -55 {
       line((0.1, y), (10, y), stroke: (dash: "dashed", paint: gray, thickness: 1pt))
    }
  }

  // Define Reference Points (scaled for the plot)
  let rest-y = 2 // -70 mV
  let peak-y = 8 // +30 mV
  let thresh-y = 3.5 // -55 mV
  let undershoot-y = 1.3 // -80 mV

  // 3. Define the Main Action Potential Trace (Blue curve)
  let trace-points = (
    (0, rest-y),   // Resting membrane potential
    (1, rest-y),   // Slow drift
    (1.5, thresh-y), // (1) Threshold level
    (2, peak-y),    // (B) Overshoot/Peak
    (2.3, 3),      // Falling phase (Repolarization)
    (2.8, undershoot-y), // (C) Hyperpolarization/Undershoot
    (3.5, 1.8),    // Slow return
    (5, rest-y),   // Back to rest
    (10, rest-y)   // Post-undershoot
  )
  
  // FIXED: Using `catmull` to draw a smooth curve through multiple points
  catmull(..trace-points, stroke: (paint: blue.darken(20%), thickness: 2pt))

  // 4. Add Annotation Callouts (A-C)
  // (A) Stimulus/Depolarization
  line((0.8, rest-y + 0.3), (1, rest-y + 0.3), mark: (start: ">", fill: black))
  content((0.8, rest-y + 0.8), text(size: 8pt, "(A) Stimulus /\nDepolarization"))

  // (B) Peak Overshoot
  let peak-pt = (2, peak-y)
  circle(peak-pt, radius: 0.15, fill: black, stroke: none)
  content((3.5, peak-y + 0.3), text(size: 8pt, "(B) Peak Overshoot\n(+30 mV)"))
  line((2.2, peak-y + 0.1), (2, peak-y + 0.1), mark: (start: ">", fill: black), stroke: (thickness: 0.8pt))
  line((peak-pt.at(0), 0.1), (peak-pt.at(0), -0.1), stroke: (thickness: 1pt)) // Vertical spike mark on x-axis

  // Repolarization Callout
  line((2.1, 6), (2.1, 5), mark: (end: ">", fill: black), stroke: (thickness: 1pt))
  content((2.5, 5.5), text(size: 8pt, "Repolarization"))

  // (C) Hyperpolarization/Undershoot
  let undershoot-pt = (2.8, undershoot-y)
  line((undershoot-pt.at(0), undershoot-y - 0.2), (undershoot-pt.at(0), 0), mark: (end: ">", fill: black), stroke: (thickness: 1pt, paint: black))
  content((4, undershoot-y - 0.5), text(size: 8pt, "(C) Undershoot\n/Hyperpolarization"))

  // 5. Mark Refractory Periods (D, E) below the axis
  let t_absolute = 2.0 * 2 // Ends at peak
  let t_relative = 2.8 * 2 // Covers repolarization

  line((0, -1), (t_absolute, -1), mark: (start: "|", end: "|", fill: black), stroke: (thickness: 1pt, paint: black))
  content((t_absolute / 2, -1.3), text(size: 8pt, "(D) Absolute Refractory\nPeriod"))
  
  line((t_absolute, -1.8), (t_relative, -1.8), mark: (start: "|", end: "|", fill: black), stroke: (thickness: 1pt, paint: black))
  content(((t_absolute + t_relative) / 2, -2.1), text(size: 8pt, "(E) Relative Refractory\nPeriod"))
})
