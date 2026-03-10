#import "@preview/cetz:0.4.2"

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  // Global styles to match your aesthetic
  set-style(
    stroke: (thickness: 1.5pt, cap: "round", join: "round"),
    mark: (fill: black, scale: 0.8)
  )

  // 1. Draw Axes
  line((-0.2, 0), (9, 0), mark: (end: ">"), stroke: (thickness: 1pt))
  line((0, -1.5), (0, 3.2), mark: (end: ">"), stroke: (thickness: 1pt))
  content((9.2, 0), $t$)
  content((0, 3.5), "Amplitude")

  // 2. Draw Background Reference Oscillation
  // Period = 4cm, so 90 degrees per 1cm. 
  let wave-pts = range(0, 85).map(i => {
    let x = i * 0.1
    let y = calc.sin(x * 90deg) // Creates a smooth sine wave
    (x, y)
  })
  line(..wave-pts, stroke: (paint: gray.lighten(10%), thickness: 1.5pt, dash: "dashed"))
  content((7.5, 1.4), text(fill: gray.darken(30%), size: 8pt, "Reference\nOscillation"))

  // 3. Draw Cycle Dividers
  line((4, -1.2), (4, 2.5), stroke: (paint: gray, thickness: 1pt, dash: "dotted"))
  line((8, -1.2), (8, 2.5), stroke: (paint: gray, thickness: 1pt, dash: "dotted"))
  content((2, -1.5), text(weight: "bold", size: 9pt, "Cycle 1"))
  content((6, -1.5), text(weight: "bold", size: 9pt, "Cycle 2"))

  // 4. Draw Spikes (Identical phase, different cycles)
  let spike-style = (paint: blue.darken(20%), thickness: 2pt)
  let phase-offset = 1.2 // x-offset within the cycle (108 degrees)
  let y-sine = calc.sin(phase-offset * 90deg)

  // Spike 1 (Cycle 1)
  line((phase-offset, y-sine), (phase-offset, 2.5), stroke: spike-style)
  circle((phase-offset, 2.5), radius: 0.08, fill: blue.darken(20%), stroke: none)
  content((phase-offset, 2.8), $phi_1$)

  // Spike 2 (Cycle 2)
  line((4 + phase-offset, y-sine), (4 + phase-offset, 2.5), stroke: spike-style)
  circle((4 + phase-offset, 2.5), radius: 0.08, fill: blue.darken(20%), stroke: none)
  content((4 + phase-offset, 2.8), $phi_2$)

  // 5. Annotate the Phase Equality / Ambiguity
  // Measurement bars showing identical offset
  line((0, -0.5), (phase-offset, -0.5), mark: (start: "|", end: "|"), stroke: (thickness: 1pt))
  content((phase-offset / 2, -0.8), $Delta t$)
  
  line((4, -0.5), (4 + phase-offset, -0.5), mark: (start: "|", end: "|"), stroke: (thickness: 1pt))
  content((4 + (phase-offset / 2), -0.8), $Delta t$)

  // The core problem statement callout
  content((4.5, 1.8), text(size: 9pt, weight: "bold", fill: red.darken(20%), "The Ambiguity:"))
  content((4.5, 1.3), text(size: 9pt, $phi_1 = phi_2 (mod 2pi)$))
  content((4.5, 0.8), text(size: 8pt, "Indistinguishable without\na cycle counter"))
  
  // Connect explanation to the spikes
  bezier((3.3, 1.3), (phase-offset + 0.2, 2.0), (2.5, 1.6), mark: (end: ">"), stroke: (paint: red.darken(20%), thickness: 1pt, dash: "dashed"))
  bezier((5.7, 1.3), (4 + phase-offset - 0.2, 2.0), (6.5, 1.6), mark: (end: ">"), stroke: (paint: red.darken(20%), thickness: 1pt, dash: "dashed"))
})
