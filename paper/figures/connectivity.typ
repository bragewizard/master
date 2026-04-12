#import "@preview/cetz:0.4.2"

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  set-style(
    stroke: (thickness: 1.6pt, cap: "butt", join: "miter"),
    mark: (fill: black, scale: 1.0)
  )

  // Helper function to draw connections that stop exactly at the node boundaries
  let connect(p1, p2) = {
    let dx = p2.at(0) - p1.at(0)
    let dy = p2.at(1) - p1.at(1)
    let d = calc.sqrt(dx * dx + dy * dy)
    let r = 0.3 // Node radius

    // Calculate points on the perimeter
    let start-x = p1.at(0) + dx * r / d
    let start-y = p1.at(1) + dy * r / d
    let end-x = p2.at(0) - dx * r / d
    let end-y = p2.at(1) - dy * r / d

    line((start-x, start-y), (end-x, end-y),
         mark: (end: ">"))
  }

  // Define node layers (Shared coordinates for both panels)
  let inputs = ((0, 1), (0, 3))
  let hiddens = ((2, 0.5), (2, 2), (2, 3.5))
  let outputs = ((4, 1.5), (4, 2.5))

  // ---------------------------------------------------------
  // (A) Feed-Forward Network
  // ---------------------------------------------------------
  group({
    content((2, 4.8), text(weight: "bold", size: 10pt, "(A) Feed-Forward"))

    // Draw standard Forward edges
    for i in inputs {
      for h in hiddens { connect(i, h) }
    }
    for h in hiddens {
      for o in outputs { connect(h, o) }
    }

    // Draw Nodes (Input = White, Hidden = Light Gray, Output = Dark Gray)
    for p in inputs { circle(p, radius: 0.3, fill: white, stroke:3pt) }
    for p in hiddens { circle(p, radius: 0.3, fill: white,stroke: 3pt) }
    for p in outputs { circle(p, radius: 0.3, fill: white, stroke: 3pt) }

    // Optional Layer Labels
    content((0, 0), text(size: 8pt, "Input"))
    content((2, 0), text(size: 8pt, "Hidden"))
    content((4, 0), text(size: 8pt, "Output"))
  })

  // ---------------------------------------------------------
  // (B) Recurrent Network
  // ---------------------------------------------------------
  group({
    translate((7.5, 0)) // Shift right panel
    content((2, 4.8), text(weight: "bold", size: 10pt, "(B) Recurrent"))

    // Draw standard Forward edges
    for i in inputs {
      for h in hiddens { connect(i, h) }
    }
    for h in hiddens {
      for o in outputs { connect(h, o) }
    }

    // --- Draw Recurrent Edges (Highlighted in Blue) ---
    let rec-color = blue.darken(20%)

    // 1. Self-loops on hidden nodes
    // Top node (loop goes up)
    bezier((1.85, 3.75), (2.15, 3.75), (1.4, 4.5), (2.6, 4.5),
           mark: (end: ">", scale: 0.6, fill: rec-color), stroke: (paint: rec-color, thickness: 1pt))
    // Middle node (loop goes up)
    bezier((1.85, 2.25), (2.15, 2.25), (1.4, 3.0), (2.6, 3.0),
           mark: (end: ">", scale: 0.6, fill: rec-color), stroke: (paint: rec-color, thickness: 1pt))
    // Bottom node (loop goes down so it doesn't cross other connections)
    bezier((1.85, 0.25), (2.15, 0.25), (1.4, -0.5), (2.6, -0.5),
           mark: (end: ">", scale: 0.6, fill: rec-color), stroke: (paint: rec-color, thickness: 1pt))

    // 2. Bidirectional connections between hidden nodes
    // H1 <-> H2
    bezier((2.2, 0.7), (2.2, 1.8), (2.6, 1.25), mark: (end: ">", scale: 0.6, fill: rec-color), stroke: rec-color)
    bezier((1.8, 1.8), (1.8, 0.7), (1.4, 1.25), mark: (end: ">", scale: 0.6, fill: rec-color), stroke: rec-color)

    // H2 <-> H3
    bezier((2.2, 2.2), (2.2, 3.3), (2.6, 2.75), mark: (end: ">", scale: 0.6, fill: rec-color), stroke: rec-color)
    bezier((1.8, 3.3), (1.8, 2.2), (1.4, 2.75), mark: (end: ">", scale: 0.6, fill: rec-color), stroke: rec-color)

    // Draw Nodes
    for p in inputs { circle(p, radius: 0.3, fill: white, stroke: black) }
    for p in hiddens { circle(p, radius: 0.3, fill: gray.lighten(70%), stroke: black) }
    for p in outputs { circle(p, radius: 0.3, fill: gray.darken(20%), stroke: black) }

    // Optional Layer Labels
    content((0, 0), text(size: 8pt, "Input"))
    content((2, -1.0), text(size: 8pt, "Hidden\n(Recurrent)", fill: rec-color))
    content((4, 0), text(size: 8pt, "Output"))


  })
})
