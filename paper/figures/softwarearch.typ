#import "@preview/cetz:0.4.2"

#align(center)[
#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  // Define global styles for consistency
  set-style(
    stroke: (thickness: 1.6pt, cap: "round", join: "round"),
    rect: (radius: 0.15, fill: white, stroke: 1.2pt),
    mark: (fill: black, scale: 1.0)
  )

  // Sub-system box for the Saccade Simulation Loop
  rect((2.5, -4.5), (12.5, 3.5), fill: rgb("f4f8fb"), stroke: (dash: "dashed", paint: rgb("4682b4"), thickness: 1.5pt))
  content((7.5, 3), text(weight: "bold", fill: rgb("4682b4"))[Discrete Saccade Simulation Loop ($T_{max} = 64$)])

  // Nodes / Components
  rect((-2, -0.2), (1.2, 1.3))
  content((-0.4, 0.55), [Raw MNIST\nImage])

  rect((3.2, -0.2), (6.8, 1.3))
  content((5.0, 0.55), [TTFS Encoder\n(Intensity-to-Latency)])

  rect((8, 1.5), (11.5, 2.8))
  content((9.75, 2.15), [Hidden Layer ($L_1$)\nIntegration & Threshold])

  rect((8, -1.5), (11.5, -0.2))
  content((9.75, -0.85), [Output Layer ($L_2$)\nIntegration & WTA])

  rect((4.5, -3.8), (10.5, -2.5), fill: rgb("fff3cd"), stroke: (paint: rgb("b8860b")))
  content((7.5, -3.15), text(weight: "bold", fill: rgb("8b6508"))[STDP Plasticity Rule\n(Weight Update)])

  rect((14, -1.5), (16.5, -0.2))
  content((15.25, -0.85), text(weight: "bold")[Classification\nResult])

  // Data Flow Edges (Forward Pass)
  line((1.2, 0.55), (3.2, 0.55), mark: (end: ">"))
  content((2.2, 0.75), text(size: 8pt)[Normalized\nPixels], anchor: "south")

  // Encoder to Layer 1
  line((6.8, 0.55), (7.4, 0.55), (7.4, 2.15), (8, 2.15), mark: (end: ">"))
  content((7.2, 1.35), [$W^{(1)}$], anchor: "east")

  // Layer 1 to Layer 2
  line((9.75, 1.5), (9.75, -0.2), mark: (end: ">"))
  content((9.95, 0.65), [$W^{(2)}$], anchor: "west")
  content((9.55, 0.65), text(size: 8pt)[Spikes ($t_h$)], anchor: "east")

  // Layer 2 to Output
  line((11.5, -0.85), (14, -0.85), mark: (end: ">"))
  content((12.75, -0.65), text(size: 8pt)[Winner\nSpike], anchor: "south")

  // STDP Feedback Flow (Learning Pass)
  // Input spikes to STDP
  line((5.0, -0.2), (5.0, -2.5), mark: (end: ">"), stroke: (dash: "dotted", paint: rgb("666666")))
  content((4.8, -1.5), text(size: 8pt, fill: rgb("666666"))[$t_"pre"$], anchor: "east")

  // Layer 1 & 2 spikes to STDP
  line((11.5, 2.15), (12, 2.15), (12, -3.15), (10.5, -3.15), mark: (end: ">"), stroke: (dash: "dotted", paint: rgb("666666")))
  line((9.75, -1.5), (9.75, -2.5), mark: (end: ">"), stroke: (dash: "dotted", paint: rgb("666666")))
  content((12.2, -0.5), text(size: 8pt, fill: rgb("666666"))[$t_"post"$], anchor: "west")

  // Apply W1 and W2 updates from STDP
  line((8.5, -2.5), (8.5, -1.5), mark: (end: ">"), stroke: (paint: rgb("d9534f"), thickness: 1.5pt))
  line((8.5, -1.5), (8.5, 1.5), mark: (end: ">"), stroke: (paint: rgb("d9534f"), thickness: 1.5pt))
  content((8.3, -0.5), text(size: 8pt, fill: rgb("d9534f"))[$Delta W^{(1)}, Delta W^{(2)}$], anchor: "east")
})
]
