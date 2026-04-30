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
  rect(
    (0, 0), (15, 11),
    fill: rgb("f4f8fb"),
    stroke: (dash: "dashed", paint: rgb("4682b4"), thickness: 1.5pt)
  )
  group({
    translate((0.5,9.0))
    rect((0, 0), (3, 1.5))
    content((1.4, 0.75), [MNIST Image])
  })
  group({
    translate((0.5,7.5))
    line((1.5, 1.2), (1.5, 0.3), mark: (end: ">"))
  })
  group({
    translate((0.5,6.0))
    rect((0, 0), (3, 1.5))
    content((1.4, 0.75), [TTFS Encoder])
  })
  group({
    translate((4.5,6.0))
    rect((0, 0), (5, 1.5))
    content((2.5, 0.75), [Hidden Layer ($L_1$)\ Integration & Threshold])
  })
  group({
    translate((10.5,6.0))
    rect((0, 0), (4, 1.5))
    content((2, 0.75), [Output Layer ($L_2$)\ Integration & WTA])
  })
  group({
    translate((7,3.0))
    rect((0, 0), (5, 1.5), fill: rgb("fff3cd"), stroke: (paint: rgb("b8860b")))
    content((2.5, 0.75), text(weight: "bold", fill: rgb("8b6508"))[STDP Plasticity Rule\ (Weight Update)])
  })
  group({
    translate((2,3.0))
    rect((0, 0), (3, 1.5))
    content((1.5, 0.75), text(weight: "bold")[Classification\ Result])
  })

  line((2.0, 1), (5.0, 1), mark: (end: ">", fill:none), stroke: (dash: "dotted", paint:rgb("666666")))
  content((4.8, 2), text(size: 8pt, fill: rgb("666666"))[$t_"pre"$], anchor: "east")





  line((6.8, 0.55), (7.4, 0.55), (7.4, 2.15), (8, 2.15), mark: (end: ">"))
  content((7.2, 1.35), [$W^{(1)}$], anchor: "east")

  line((9.75, 1.5), (9.75, -0.2), mark: (end: ">"))
  content((9.95, 0.65), [$W^{(2)}$], anchor: "west")
  content((9.55, 0.65), text(size: 8pt)[Spikes ($t_h$)], anchor: "east")

})
]
