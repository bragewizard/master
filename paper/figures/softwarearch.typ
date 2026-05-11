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
    (0, 2), (15, 11),
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
    content((2.5, 0.75), [Hidden Layer ($L^1$)\ Integration & Threshold])
  })
  group({
    translate((10.5,6.0))
    rect((0, 0), (4, 1.5))
    content((2, 0.75), [Output Layer ($L^2$)\ Integration & WTA])
  })
  group({
    translate((1.5,3.0))
    rect((0, 0), (5, 1.5), fill: rgb("fff3cd"), stroke: (paint: rgb("b8860b")))
    content((2.5, 0.75), text(weight: "bold", fill: rgb("8b6508"))[STDP Plasticity Rule\ (Weight Update)])
  })
  group({
    translate((5.2,9.0))
    rect((0, 0), (3.4, 1.5), fill: rgb("fff3cd"), stroke: (paint: rgb("b8860b")))
    content((1.7, 0.75), text(weight: "bold", fill: rgb("8b6508"))[Homeostasis])
  })
  group({
    translate((7.5,3.0))
    rect((0, 0), (3.0, 1.5), fill: rgb("fff3cd"), stroke: (paint: rgb("b8860b")))
    content((1.5, 0.75), text(weight: "bold", fill: rgb("8b6508"))[Hidden Layer\ WTA])
  })
  group({
    translate((11,9.0))
    rect((0, 0), (3, 1.5))
    content((1.5, 0.75), text(weight: "bold")[Classification\ Result])
  })




  line((3.7, 6.8), (4.4, 6.8), mark: (end: ">"))
  line((9.7, 6.8), (10.4, 6.8), mark: (end: ">"))
  line((12.5, 7.8), (12.5, 8.8), mark: (end: ">"))

  line((6.8, 7.8), (6.8, 8.8), mark: (end: ">"))
  line((7.1, 8.8), (7.1, 7.8), mark: (end: ">"))

  line((5.8, 4.8), (5.8, 5.8), mark: (end: ">"))
  line((8.1, 5.8), (8.1, 4.8), mark: (end: ">"))
  line((7.3, 3.8), (6.6, 3.8), mark: (end: ">"))



})
]
