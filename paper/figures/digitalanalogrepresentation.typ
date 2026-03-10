#import "@preview/cetz:0.4.2"

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  // Global aesthetics matching your minimalistic style
  set-style(
    stroke: (thickness: 1.5pt, cap: "round", join: "round"),
    mark: (fill: black, scale: 0.8)
  )

  // Reusable helper function for consistent axis setup
  let setup-axes(x-label, y-label) = {
    line((-0.2, 0), (7, 0), mark: (end: ">"), stroke: (thickness: 1pt))
    line((0, -0.2), (0, 4.2), mark: (end: ">"), stroke: (thickness: 1pt))
    content((7.2, 0), x-label)
    content((0, 4.4), y-label)
  }

  // ---------------------------------------------------------
  // Digital Representation (Left)
  // ---------------------------------------------------------
  group({
    translate((0, 0))
    setup-axes($t$, "Value")
    
    // Label for the panel
    content((3.5, 4.6), text(weight: "bold", size: 10pt, "Digital Representation"))
    
    // Grey reference line for comparison
    line((-0.2, 2), (6.8, 2), stroke: (paint: gray, dash: "dashed", thickness: 1pt))
    
    // Digital Signal (Blue with steps)
    let trace-style = (paint: blue.darken(20%), thickness: 1.5pt)
    line((0,0), (0.7,0), (0.7,0.7), (1.4,0.7), (1.4,1.4), (2.1,1.4), (2.1,2.1), (2.8,2.1), (2.8,2.8), (3.5,2.8), (3.5,3.5), (4.2,3.5), (4.2,3.8), (4.9,3.8), (4.9,2.8), (5.6,2.8), (5.6,1.4), (6.3,1.4), (6.3,0.7), stroke: trace-style)

    // Data points (Hollow circles for digital sampling)
    for t in (0, 0.7, 1.4, 2.1, 2.8, 3.5, 4.2, 4.9, 5.6, 6.3) {
      let pt = (t, 0)
      circle(pt, radius: 0.08, fill: white, stroke: black)
    }

    // Callout lines and labels
    line((2.8, 3.5), (3.2, 3.1), mark: (start: ">"), stroke: (thickness: 1pt))
    content((3.2, 3.7), text(size: 8pt, "(A) Quantization Steps"))

    line((0.7, -0.2), (0.7, 0), stroke: (paint: blue.darken(20%), thickness: 1pt))
    line((1.4, -0.2), (1.4, 0), stroke: (paint: blue.darken(20%), thickness: 1pt))
    line((2.1, -0.2), (2.1, 0), stroke: (paint: blue.darken(20%), thickness: 1pt))
    line((0.7, -0.4), (2.1, -0.4), mark: (start: "|", end: "|"), stroke: (thickness: 1pt))
    content((1.4, -0.8), text(size: 8pt, "(B) Discrete Time Sampling"))
  })

  // ---------------------------------------------------------
  // Analog Representation (Right)
  // ---------------------------------------------------------
  group({
    translate((8.5, 0)) // Shift the right panel
    setup-axes($t$, "Value")
    
    // Label for the panel
    content((3.5, 4.6), text(weight: "bold", size: 10pt, "Analog Representation"))
    
    // Grey reference line for comparison
    line((-0.2, 2), (6.8, 2), stroke: (paint: gray, dash: "dashed", thickness: 1pt))

    // Analog Signal (Smooth curve)
    let trace-style = (paint: purple.darken(10%), thickness: 1.5pt)
    bezier((0,0.5), (6,3), (1,3), stroke: trace-style)

    // Callout lines and labels
    line((1.5, 2.5), (1.1, 2), mark: (start: ">"), stroke: (thickness: 1pt))
    content((1.8, 2.7), text(size: 8pt, "(C) Continuous-Time"))

    line((4.2, 1.5), (3.8, 1.2), mark: (start: ">"), stroke: (thickness: 1pt))
    content((4.8, 1.7), text(size: 8pt, "(D) Infinite Precision"))

    line((3, -0.4), (3.7, -0.4), mark: (start: "|", end: "|"), stroke: (thickness: 1pt))
    content((3.35, -0.8), text(size: 8pt, "(E) Infinite Values"))
  })
})
