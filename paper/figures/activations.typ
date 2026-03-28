#import "@preview/cetz:0.4.2"

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  // Global styles
  set-style(
    stroke: (thickness: 1.4pt, cap: "round", join: "miter"),
    mark: (fill: black, scale: 1)
  )

  group({
    content((2.5, 5.2), text(weight: "bold", size: 10pt, "(A) Sigmoid"))

    // Draw Axes
    line((-6, 0), (7.5, 0), mark: (end: ">"))
    content((8.3, 0), [$x$])

    line((0, -0.5), (0, 4.5), mark: (end: ">"))
    content((0, 4.8), [$sigma(x)$])

    // Reference lines
    line((-6, 4), (6, 4), stroke: (paint: gray, dash: "dashed", thickness: 1pt))
    content((-0.8, 4.2), text(size: 8pt, fill: gray.darken(30%), "1.0"))
    content((-0.3, -0.3), text(size: 8pt, "0"))

    let pts = range(-60, 60, step: 2).map(i => {
       let x = i * 0.1
       let y = 4.0 / (1.0 + calc.exp(-x))
       (x, y)
    })
    line(..pts, stroke: (paint: blue.darken(30%), thickness: 2pt))

    let sat_col = red.darken(10%)

    content((-5, .7), text(fill: sat_col, size: 8pt, weight: "bold", "Saturates\n(kills gradients)"))
    content((5.2, 3.2), text(fill: sat_col, size: 8pt, weight: "bold", "Saturates\n(kills gradients)"))
    line((-6, 0.2),(-4, 0.2), stroke: (paint: sat_col, thickness: 1.5pt, dash: "dotted"))
    line((4, 3.8),(6, 3.8), stroke: (paint: sat_col, thickness: 1.5pt, dash: "dotted"))

    let d_pts = range(-60, 60, step: 2).map(i => {
       let x = i * 0.1
       let s = 1.0 / (1.0 + calc.exp(-x))
       let ds = s * (1.0 - s) * 4.0
       (x, ds)
    })
    line(..d_pts, stroke: (paint: gray, thickness: 1pt, dash: "dashed"))
    content((2.5, 1.3), text(size: 7pt, fill: gray.darken(30%), [$sigma'(x)$ \ approaches 0]))
  })

  group({
    translate((0, -8))
    content((2.5, 5.2), text(weight: "bold", size: 10pt, "(B) ReLU"))

    line((-6, 0), (7.5, 0), mark: (end: ">"), stroke: (thickness: 1pt))
    content((7.8, 0), [$x$])

    line((0, -0.5), (0, 4.5), mark: (end: ">"), stroke: (thickness: 1pt))
    content((0, 4.8), [$f(x)$])
    content((-0.3, -0.3), text(size: 8pt, "0"))

    line((0, 0), (4.5, 4.5), stroke: (paint: blue.darken(30%), thickness: 2pt))
    line((-6, 0), (0, 0), stroke: (paint: blue.darken(30%), thickness: 2pt))

    line((0, 1.5), (7.5, 1.5), stroke: (paint: gray, thickness: 1.2pt, dash: "dashed"))
    circle((0, 1.5), radius: 0.1, fill: white, stroke: (paint: gray, thickness: 1.2pt))

    line((-6, 0.1), (0, 0.1), stroke: (paint: red.darken(10%), thickness: 1.2pt, dash: "dashed"))

    content((3.5, 1.2), text(size: 7pt, fill: gray.darken(30%), [Derivative is constant 1.0]))
    content((-4, 0.4), text(size: 7pt, fill: gray.darken(30%), [Derivative is constant 0.0]))
  })
})
