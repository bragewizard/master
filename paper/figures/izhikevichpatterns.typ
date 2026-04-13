#import "@preview/cetz:0.4.2"

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  set-style(
    stroke: (thickness: 1.6pt, cap: "round", join: "round"),
    mark: (fill: black, scale: 1.0)
  )

  // Reusable helper function for consistent panel setup
  let setup-panel(name) = {
    content((2.1, 2.6), text(weight: "bold", size: 9pt, name))
  }

  let trace-style = (paint: blue.darken(40%))

  // ---------------------------------------------------------
  // (A) Tonic Spiking - Constant frequency firing
  // ---------------------------------------------------------
  group({
    translate((0, 0))
    setup-panel("(A) Tonic Spiking")

    let pts = ((0,0), (0.4,0))
    for t in (0.7, 1.2, 1.7, 2.2, 2.7, 3.2) {
      pts.push((t - 0.15, 0.0))   // Integration rise
      pts.push((t, 2.2))          // Spike peak
      pts.push((t + 0.05, -0.6))  // Fast reset
      pts.push((t + 0.20, 0)) // Slightly deeper reset due to adaptation
    }
    pts.push((3.6, 0))
    pts.push((4.2, 0))
    line(..pts, stroke: trace-style)
  })

  // ---------------------------------------------------------
  // (B) Phasic Spiking - Single spike at stimulus onset
  // ---------------------------------------------------------
  group({
    translate((5.5, 0)) // Shift right
    setup-panel("(B) Phasic Spiking")

    let pts = ((0,0), (0.4,0))
    for t in (0.7,) {
      pts.push((t - 0.15, 0.0))   // Integration rise
      pts.push((t, 2.2))          // Spike peak
      pts.push((t + 0.05, -0.6))  // Fast reset
      pts.push((t + 0.2, 0)) // Slightly deeper reset due to adaptation
    }
    pts.push((3.6, 0))
    pts.push((4.2, 0))
    line(..pts, stroke: trace-style)
  })

  group({
    translate((0, -5.0)) // Shift down
    setup-panel("(C) Spike Frequency Adaptation")

    let pts = ((0,0), (0.4,0))
    // Inter-spike intervals progressively widen
    for t in (0.6, 1, 1.4, 1.95, 2.6, 3.35) {
      pts.push((t - 0.15, 0.0))
      pts.push((t, 2.2))
      pts.push((t + 0.05, -0.7)) // Slightly deeper reset due to adaptation
      pts.push((t + 0.2, 0)) // Slightly deeper reset due to adaptation
    }
    pts.push((3.6, 0))
    pts.push((3.6, 0))
    pts.push((4.2, 0))
    line(..pts, stroke: trace-style)
  })

  // ---------------------------------------------------------
  // (D) Bursting - Clusters of spikes separated by rest
  // ---------------------------------------------------------
  group({
    translate((5.5, -5.0)) // Shift right and down
    setup-panel("(D) Bursting")

    let pts = ((0,0), (0.4,0))

    // First Burst
    for t in (0.6, 0.75, 0.9) {
      pts.push((t - 0.05, 0.0))
      pts.push((t, 2.2))
      pts.push((t + 0.05, -0.4))
    }
    // Slow afterhyperpolarization wave
    pts.push((0.96, -0.6))
    pts.push((1.6, 0))
    pts.push((2.1, 0))

    // Second Burst
    for t in (2.3, 2.45, 2.6) {
      pts.push((t - 0.05, 0.4))
      pts.push((t, 2.2))
      pts.push((t + 0.05, -0.4))
    }
    // Return to baseline
    pts.push((2.66, -0.6))
    pts.push((3.4, 0))
    pts.push((4.1, 0))

    line(..pts, stroke: trace-style)
  })
})
