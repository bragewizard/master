#import "@preview/cetz:0.4.2"

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  // Global aesthetics matching your minimalist style
  set-style(
    stroke: (thickness: 1.2pt, cap: "round", join: "round")
  )

  // Reusable helper function for consistent panel setup
  let setup-panel(name) = {
    content((2.1, 2.6), text(weight: "bold", size: 9pt, name))
    
    // Baseline Voltage Reference
    line((0, 0), (4.2, 0), stroke: (paint: gray, dash: "dashed", thickness: 0.8pt))
    
    // Stimulus Current (Square wave)
    line((0, -1.0), (0.4, -1.0), (0.4, -0.5), (3.6, -0.5), (3.6, -1.0), (4.2, -1.0), 
         stroke: (paint: red.darken(10%), thickness: 1pt))
    content((2.1, -1.4), text(size: 8pt, fill: red.darken(10%), "Stimulus Current (I)"))
  }

  let trace-style = (paint: blue.darken(20%), thickness: 1.2pt)

  // ---------------------------------------------------------
  // (A) Tonic Spiking - Constant frequency firing
  // ---------------------------------------------------------
  group({
    translate((0, 0))
    setup-panel("(A) Tonic Spiking")
    
    let pts = ((0,0), (0.4,0))
    for t in (0.7, 1.2, 1.7, 2.2, 2.7, 3.2) {
      pts.push((t - 0.15, 0.1))   // Integration rise
      pts.push((t - 0.05, 0.4))   // Threshold crossed
      pts.push((t, 2.2))          // Spike peak
      pts.push((t + 0.05, -0.6))  // Fast reset
    }
    pts.push((3.6, 0.1))
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
    
    let pts = ((0,0), (0.4,0), (0.5, 0.1), (0.6, 0.4), (0.65, 2.2), (0.7, -0.6), 
               (0.8, 0.1), (1.5, 0.15), (3.6, 0.15), (3.6, 0), (4.2, 0))
    line(..pts, stroke: trace-style)
  })

  // ---------------------------------------------------------
  // (C) Spike Frequency Adaptation - Decreasing frequency
  // ---------------------------------------------------------
  group({
    translate((0, -5.0)) // Shift down
    setup-panel("(C) Spike Frequency Adaptation")
    
    let pts = ((0,0), (0.4,0))
    // Inter-spike intervals progressively widen
    for t in (0.6, 0.95, 1.4, 1.95, 2.6, 3.35) {
      pts.push((t - 0.15, 0.1))
      pts.push((t - 0.05, 0.4))
      pts.push((t, 2.2))
      pts.push((t + 0.05, -0.7)) // Slightly deeper reset due to adaptation
    }
    pts.push((3.6, 0.1))
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
      pts.push((t - 0.05, 0.4))
      pts.push((t, 2.2))
      pts.push((t + 0.05, -0.4))
    }
    // Slow afterhyperpolarization wave
    pts.push((1.1, -0.8))
    pts.push((1.4, -0.6))
    pts.push((1.8, -0.1))
    pts.push((2.1, 0.2))
    
    // Second Burst
    for t in (2.3, 2.45, 2.6) {
      pts.push((t - 0.05, 0.4))
      pts.push((t, 2.2))
      pts.push((t + 0.05, -0.4))
    }
    // Return to baseline
    pts.push((2.8, -0.8))
    pts.push((3.1, -0.6))
    pts.push((3.5, -0.1))
    pts.push((3.6, -0.1))
    pts.push((3.6, 0))
    pts.push((4.2, 0))
    
    line(..pts, stroke: trace-style)
  })
})
