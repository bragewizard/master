#import "@preview/cetz:0.4.2"

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  // Global styles to match your aesthetic
  set-style(
    stroke: (thickness: 1.5pt, cap: "round", join: "round"),
    mark: (fill: black, scale: 0.8)
  )

  // 1. Draw Axes
  line((-0.5, 0), (9, 0), mark: (end: ">", fill: black), stroke: (thickness: 1pt, paint: black)) // Time axis (t)
  line((0, -0.5), (0, 4.5), mark: (end: ">", fill: black), stroke: (thickness: 1pt, paint: black)) // Voltage axis (V)
  content((9.2, 0), $t$)
  content((0, 4.8), $V(t)$)

  // 2. Draw Threshold and Reset Reference Lines
  line((-0.2, 2.5), (8.5, 2.5), stroke: (dash: "dashed", paint: gray, thickness: 1pt))
  content((-0.6, 2.5), $theta.alt$)

  line((-0.2, 0.5), (8.5, 0.5), stroke: (dash: "dotted", paint: gray, thickness: 1pt))
  content((-0.8, 0.5), $V_"reset"$)

  // 3. Draw the LIF Dynamics (Voltage Trace)
  // We use a darker blue to make the membrane trajectory stand out against the black axes
  let trace-style = (paint: blue.darken(20%), thickness: 1.5pt)

  // First spiking cycle
  bezier((0, 0.5), (3, 2.5), (1.5, 2.0), stroke: trace-style) // (A) Integration
  line((3, 2.5), (3, 4), (3.05, 0.5), stroke: trace-style)    // (B) Spike and instantaneous reset
  line((3.05, 0.5), (4.5, 0.5), stroke: trace-style)          // (C) Refractory period

  // Second spiking cycle
  bezier((4.5, 0.5), (7.5, 2.5), (6.0, 2.0), stroke: trace-style)
  line((7.5, 2.5), (7.5, 4), (7.55, 0.5), stroke: trace-style)
  line((7.55, 0.5), (8.5, 0.5), stroke: trace-style)

  // 4. Add Annotations for (A), (B), and (C)
  // (A) Integrates inputs
  content((1.2, 2.2), text(size: 8pt, "(A) Integrate"))
  
  // (B) Spike and Reset
  content((4.4, 3.5), text(size: 8pt, "(B) Spike & Reset"))
  line((3.4, 3.5), (3.05, 3.5), mark: (end: ">"), stroke: (thickness: 0.8pt)) // Pointer arrow
  
  // (C) Clamped during refractory
  line((3.05, -0.3), (4.5, -0.3), mark: (start: "|", end: "|"), stroke: (thickness: 0.8pt))
  content((3.77, -0.7), text(size: 8pt, "(C) Clamped"))
})
