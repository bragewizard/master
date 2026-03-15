#import "@preview/cetz:0.4.2"

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  // Global aesthetics matching minimalist academic style
  set-style(
    stroke: (thickness: 1.2pt, cap: "round", join: "round"),
    mark: (fill: black, scale: 0.8)
  )

  // 1. Title
  content((5.5, 5.5), text(weight: "bold", size: 10pt, "The Von Neumann Architecture & Bottleneck"))

  // ------------------------------------------------------------------
  // LEFT: Compute Unit (CPU/GPU)
  // ------------------------------------------------------------------
  group({
    rect((0, 0), (3, 4), fill: blue.lighten(80%), stroke: (paint: blue.darken(20%), thickness: 1.5pt))
    content((1.5, 3.5), text(weight: "bold", fill: blue.darken(20%), "Compute Unit"))
    
    // Internal Components
    rect((0.4, 0.4), (2.6, 1.6), fill: white, stroke: (paint: blue.darken(20%), thickness: 1pt))
    content((1.5, 1.0), text(size: 8pt, "ALU / Cores"))
    
    rect((0.4, 2.0), (2.6, 2.8), fill: white, stroke: (paint: blue.darken(20%), thickness: 1pt))
    content((1.5, 2.4), text(size: 8pt, "Registers / Cache"))

    // Speed metric
    content((1.5, -0.6), text(size: 8pt, weight: "bold", fill: blue.darken(20%), "Fast Execution\n(~GHz speeds)"))
  })

  // ------------------------------------------------------------------
  // RIGHT: Memory Unit (RAM)
  // ------------------------------------------------------------------
  group({
    rect((8, 0), (11, 4), fill: purple.lighten(80%), stroke: (paint: purple.darken(10%), thickness: 1.5pt))
    content((9.5, 3.5), text(weight: "bold", fill: purple.darken(10%), "Memory Unit"))

    // Draw a grid to represent dense memory arrays
    for i in range(3) {
      for j in range(4) {
        rect((8.3 + i*0.8, 0.4 + j*0.6), (9.1 + i*0.8, 0.9 + j*0.6), fill: white, stroke: (paint: purple.darken(10%), thickness: 0.8pt))
      }
    }

    // Capacity/Speed metric
    content((9.5, -0.6), text(size: 8pt, weight: "bold", fill: purple.darken(10%), "High Capacity\n(High Latency)"))
  })

  // ------------------------------------------------------------------
  // MIDDLE: The Bus & Bottleneck
  // ------------------------------------------------------------------
  group({
    // Top arrow: Memory to Compute (Read)
    line((8, 2.5), (3, 2.5), mark: (end: ">", fill: black), stroke: (thickness: 2pt))
    content((5.5, 2.8), text(size: 8pt, "Instructions & Data (Read)"))
    
    // Bottom arrow: Compute to Memory (Write)
    line((3, 1.5), (8, 1.5), mark: (end: ">", fill: black), stroke: (thickness: 2pt))
    content((5.5, 1.2), text(size: 8pt, "Computed Results (Write)"))

    // The Constriction Zone (Bottleneck)
    rect((4.2, 0.8), (6.8, 3.2), fill: rgb(255, 0, 0, 30), stroke: (paint: red.darken(10%), thickness: 1.5pt, dash: "dashed"))
    content((5.5, 4.0), text(weight: "bold", fill: red.darken(10%), size: 9pt, "The Von Neumann Bottleneck"))
    content((5.5, 3.6), text(size: 7pt, fill: red.darken(10%), "(Limited Bandwidth)"))

    // ------------------------------------------------------------------
    // ANNOTATION: Massive Energy Expenditure
    // ------------------------------------------------------------------
    // Draw downward arrows to represent energy dissipation/loss during transport
    for x in (4.5, 5.5, 6.5) {
      line((x, 0.8), (x, -0.8), mark: (end: ">", fill: red.darken(10%)), stroke: (paint: red.darken(10%), thickness: 1.2pt, dash: "dotted"))
    }
    
    content((5.5, -1.3), text(size: 8pt, weight: "bold", fill: red.darken(10%), "Massive Energy Expenditure"))
    content((5.5, -1.7), text(size: 7pt, fill: red.darken(10%), "Data Transport > Compute Cost"))
  })
})
