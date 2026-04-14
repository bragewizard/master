#import "@preview/cetz:0.4.2"

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  set-style(
    stroke: (thickness: 1.6pt, cap: "round", join: "round"),
    mark: (fill: black, scale: 1.0)
  )

  group({
    rect((0, 0), (3, 4), fill: blue.lighten(80%), stroke: (paint: blue.darken(60%)), radius:4pt)
    content((1.5, 3.5), text(weight: "bold", fill: blue.darken(60%), [Compute Unit]))

    // Internal Components
    rect((0.4, 0.4), (2.6, 1.6), fill: white, stroke: (paint: blue.darken(60%)))
    content((1.5, 1.0), text(size: 8pt, [ALU / Cores]))

    rect((0.4, 1.9), (2.6, 2.9), fill: white, stroke: (paint: blue.darken(60%)))
    content((1.5, 2.4), text(size: 8pt, [Registers /\ Cache]))

    // Speed metric
    content((1.5, -0.6), text(size: 9pt, weight: "bold", fill: blue.darken(60%), "Fast Execution\n(~GHz speeds)"))
  })

  group({
    rect((8, 0), (11, 4), fill: yellow.lighten(60%),stroke:yellow.darken(60%),radius:4pt)
    content((9.5, 3.5), text(weight: "bold", fill: yellow.darken(60%), "Memory Unit"))

    // Draw a grid to represent dense memory arrays
    for i in range(3) {
      for j in range(4) {
        rect((8.3 + i*0.8, 0.4 + j*0.6), (9.1 + i*0.8, 0.9 + j*0.6), fill: white, stroke: (paint: yellow.darken(60%)))
      }
    }

    content((9.5, -0.6), text(size: 9pt, weight: "bold", fill: yellow.darken(60%), [High Capacity\ (High Latency)]))
  })

  group({
    rect((3.4, 0.8), (7.6, 3.2), fill:red.lighten(80%), stroke: (paint: red.darken(60%), dash: "dashed"),radius:4pt)
    line((8, 2.5), (3, 2.5), mark: (end: ">", fill: black), stroke: (thickness: 2pt))
    content((5.5, 2.8), text(size: 9pt, "Instructions & Data (Read)"))

    // Bottom arrow: Compute to Memory (Write)
    line((3, 1.5), (8, 1.5), mark: (end: ">", fill: black), stroke: (thickness: 2pt))
    content((5.5, 1.2), text(size: 9pt, "Computed Results (Write)"))

    content((5.5, 4.0), text(weight: "bold", fill: red.darken(60%), size: 9pt, "The Von Neumann Bottleneck"))
    content((5.5, 3.6), text(size: 7pt, fill: red.darken(60%), "(Limited Bandwidth)"))

    for x in (4.5, 5.5, 6.5) {
      line((x, 0.8), (x, -0.8), mark: (end: ">", stroke:(paint:black, dash:none)), stroke: (paint: red.darken(60%), dash: "dotted"))
    }

    content((5.5, -1.3), text(size: 9pt, weight: "bold", fill: red.darken(60%), "Massive Energy Expenditure"))
    content((5.5, -1.7), text(size: 9pt, fill: red.darken(60%), "Data Transport > Compute Cost"))
  })
})
