#import "@preview/cetz:0.4.2"

#cetz.canvas(length: 0.9cm, {
  import cetz.draw: *

  set-style(
    stroke: (thickness: 1.6pt, cap: "round", join: "round"),
    mark: (fill: black, scale: 1.0)
  )

  line((-0.1, 0), (10.5, 0), mark: (end: ">", fill:black))
  content((12, 0), "Time (ms)")
  for i in range(1, 6) {
    let x = i * 2 // Scaling factor
    line((x, -0.1), (x, 0.1))
    content((x, -0.5), [#i])
  }
  line((0, -0.1), (0, 0.1))
  content((0, -0.5), "0")

  // Y-axis (Membrane Potential in mV)
  line((0, -0.1), (0, 10), mark: (end: ">", fill:black))
  content((0, 10.5), "Membrane Potential (mV)")

  // Ticks and horizontal reference lines
  let y-levels = ((-80, 1.3), (-70, 2), (-55, 3.5), (+40, 9)) // (potential, scaled y)
  for level in y-levels {
    let y = level.at(1)
    let label = level.at(0)
    line((-0.1, y), (0.1, y))
    content((-0.6, y), [#label])

    // Horizontal dashed lines
    if label == -70 or label == -55 {
       line((0.1, y), (10, y), stroke: (dash: "dashed", paint: gray, ))
    }
  }

  // Define Reference Points (scaled for the plot)
  let rest-y = 2 // -70 mV
  let peak-y = 9 // +40 mV
  let thresh-y = 3.5 // -55 mV
  let undershoot-y = 1.3 // -80 mV

  // 3. Define the Main Action Potential Trace (Blue curve)
  let trace-points = (
    (0, rest-y),   // Resting membrane potential
    (2.8, rest-y+0.1),   // Slow drift
    (3.5, thresh-y), // (1) Threshold level
    (4, peak-y),    // (B) Overshoot/Peak
    (4.3, 3),      // Falling phase (Repolarization)
    (4.8, undershoot-y), // (C) Hyperpolarization/Undershoot
    (5.5, 1.8),    // Slow return
    (7, rest-y+0),   // Back to rest
    (10, rest-y)   // Post-undershoot
  )

  // FIXED: Using `catmull` to draw a smooth curve through multiple points
  catmull(..trace-points,tension:.5, stroke: (paint: blue.darken(50%)))

  // 4. Add Annotation Callouts (A-C)
  // (A) Stimulus/Depolarization
  line((3.2, rest-y + 1.2), (3, rest-y + 0.4), mark: (start: ">", fill: black))
  content((1.7, rest-y + 0.7), text(size: 8pt, "(A) Stimulus /\nDepolarization"))

  // (B) Peak Overshoot
  let peak-pt = (2, peak-y)
  content((4.8, peak-y + 0.6), text(size: 8pt, "(B) Peak Overshoot\n(+40 mV)"))
  line((peak-pt.at(0), 0.1), (peak-pt.at(0), -0.1)) // Vertical spike mark on x-axis

  // Repolarization Callout
  line((4.4, 6), (4.4, 5), mark: (end: ">", fill: black))
  content((5.7, 5.5), text(size: 8pt, "Repolarization"))

  // (C) Hyperpolarization/Undershoot
  let undershoot-pt = (4.9, undershoot-y)
  line((undershoot-pt.at(0), undershoot-y + 1), (undershoot-pt.at(0), 1.5), mark: (end: ">", fill: black))
  content((6.8, undershoot-y + 1.5), text(size: 8pt, "(C) Undershoot\n/Hyperpolarization"))

  // 5. Mark Refractory Periods (D, E) below the axis
  let t_absolute = 2.0 * 2 // Ends at peak
  let t_relative = 2.8 * 2 // Covers repolarization
})
