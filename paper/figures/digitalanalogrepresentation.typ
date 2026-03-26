#import "@preview/cetz:0.4.2"

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  set-style(
    stroke: (thickness: 1.4pt, cap: "butt", join: "miter"),
    mark: (fill: black, scale: 1)
  )

  let setup-axes(x-label, y-label) = {
    line((-0.2, 0), (7, 0), mark: (end: ">"), stroke: (thickness: 1.4pt))
    line((0, -0.2), (0, 4.2), mark: (end: ">"), stroke: (thickness: 1.4pt))
    content((7.2, 0), x-label)
    content((0, 4.5), y-label)
  }

  group({
    translate((0, 0))
    setup-axes($t$, "Value")

    content((3.5, 4.6), text(weight: "bold", size: 10pt, "Digital Representation"))

    let trace-style = (paint: blue.darken(20%), thickness: 2pt)
    line((0,0.7), (0.7,0.7), (0.7,1.4), (1.4,1.4), (1.4,2.1), (2.1,2.1), (2.1,2.1), (2.8,2.1), (2.8,2.1), (3.5,2.1), (3.5,2.8), (4.2,2.8), (4.2,2.8), (4.9,2.8), (4.9,2.8), (5.6,2.8), (5.6,2.8), (6.3,2.8), (6.3,2.8), stroke: trace-style)

    for t in (0, 0.7, 1.4, 2.1, 2.8, 3.5, 4.2, 4.9, 5.6, 6.3) {
      let pt = (t, 0)
      circle(pt, radius: 0.08, fill: white, stroke: black)
    }
    for v in (0.7, 1.4, 2.1, 2.8, 3.5) {
      let pt = (0, v)
      circle(pt, radius: 0.08, fill: white, stroke: black)
    }

    line((0.68, -0.4), (1.42, -0.4), mark: (start: "|", end: "|"), stroke: (thickness: 1pt))
    content((1.4, -0.8), text(size: 8pt, "Discrete sampling"))
  })

  // ---------------------------------------------------------
  // Analog Representation (Right)
  // ---------------------------------------------------------
  group({
    translate((8.5, 0)) // Shift the right panel
    setup-axes($t$, "Value")

    content((3.5, 4.6), text(weight: "bold", size: 10pt, "Analog Representation"))


    let trace-style = (paint: green.darken(40%), thickness: 2pt)
    bezier((0,0.5), (6,3), (1,3), stroke: trace-style)


    line((3, -0.4), (3.7, -0.4), mark: (start: "|", end: "|"), stroke: (thickness: 1pt))
    content((3.35, -0.8), text(size: 8pt, "Infinite values"))
  })
})
