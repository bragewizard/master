#import "@preview/cetz:0.4.2"
#import "@preview/cetz-plot:0.1.3": plot, chart

// --- Data Preparation ---
#let data = (
  (0.0, -70), (0.5, -70.2), (1.0, -69.8), (1.5, -70),
  (1.8, -55),
  (1.9, -40), (2.0, -10), (2.1, 20), (2.2, 35), (2.25, 40),
  (2.3, 30), (2.4, 10), (2.5, -10), (2.6, -30), (2.7, -50), (2.8, -65),
  (3.0, -75), (3.5, -80), (4.0, -82), (4.5, -80), (5.0, -75),
  (6.0, -71), (7.0, -70.5), (8.0, -70)
)

#cetz.canvas(length: .6cm, {
  import cetz.draw: *

    set-style(axes: (
      stroke: (thickness: 1pt, paint: black),
      x: (mark: (end: ">", fill:black, size:1pt)),
      y: (mark: (end: ">", fill:black, size:1pt)),
      tick: (stroke: black + 1pt),
    ))
    plot.plot(
    size: (12, 5), 
    x-tick-step: 1, 
    y-tick-step: 20,
    y-min: -100,
    x-format: v => text(str(v)),
    y-format: v => text(str(v)),
    axis-style: "left", 
    name: "phase", 
    {    
    plot.add(
      data,
      style: (stroke: (paint: gray, thickness: 2pt)),
      label: "potential"
    )
  })
  // 1. Title and Annotations (Relative to Canvas)

  
  line((3.4, 6.2), (3.4, 5.2), mark: (end: ">"), stroke: (thickness: 1pt))
  content((3.4, 6.5), text(size: 9pt, "Peak (+40mV)"))

  // 2. The Lilaq Diagram
  // Placed at the center of the previous coordinate system
})
