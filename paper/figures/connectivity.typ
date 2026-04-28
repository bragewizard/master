#import "@preview/cetz:0.4.2"

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  set-style(
    stroke: (thickness: 1.6pt, cap: "round", join: "round"),
    mark: (fill: black, scale: 1.0)
  )

  let connect(p1, p2) = {
    let r = 0.3 // Node radius
    let start-x = p1.at(0)
    let start-y = p1.at(1)
    let end-x = p2.at(0)
    let end-y = p2.at(1)
    start-y = (0.06 * end-y) + (0.94 * start-y)
    end-y = (end-y - start-y) * 0.92 + start-y
    end-x = end-x - .4
    start-x = start-x + .4

    bezier((start-x, start-y), (end-x, end-y), ((end-x - start-x)*0.5 + start-x, start-y),((end-x - start-x)*0.7 +start-x,end-y), mark: (end: ">", scale:0.6))
  }

  let inputs = ((0, 1), (0, 3))
  let hiddens = ((2, 0.0), (2, 2), (2, 4.0))
  let outputs = ((4, 1.0), (4, 3.0))

  // ---------------------------------------------------------
  // (A) Feed-Forward Network
  // ---------------------------------------------------------
  group({
    content((2, 4.8), text(weight: "bold", "(A) Feed-Forward"))

    // Draw standard Forward edges
    for i in inputs {
      for h in hiddens { connect(i, h) }
    }
    for h in hiddens {
      for o in outputs { connect(h, o) }
    }

    // Draw Nodes (Input = White, Hidden = Light Gray, Output = Dark Gray)
    for p in inputs { circle(p, radius: 0.3, fill: yellow.lighten(50%), stroke:2pt) }
    for p in hiddens { circle(p, radius: 0.3, fill: yellow.lighten(50%),stroke: 2pt) }
    for p in outputs { circle(p, radius: 0.3, fill: yellow.lighten(50%), stroke: 2pt) }

  })

  // ---------------------------------------------------------
  // (B) Recurrent Network
  // ---------------------------------------------------------
  group({
    translate((7.5, 0)) // Shift right panel
    content((2, 4.8), text(weight: "bold", "(B) Recurrent"))

    // Draw standard Forward edges
    for i in inputs {
      for h in hiddens { connect(i, h) }
    }
    for h in hiddens {
      for o in outputs { connect(h, o) }
    }

    // --- Draw Recurrent Edges (Highlighted in Blue) ---
    let rec-color = blue.darken(20%)

    bezier((1.80, -0.4), (2.20, -0.4), (1.4, -1.1), (2.6, -1.1),
           mark: (end: ">", scale: 0.6, fill: rec-color), stroke: (paint: rec-color))

    bezier((2.2, 0.4), (2.2, 1.6), (2.6, 1.10), mark: (end: ">", scale: 0.6, fill: rec-color), stroke: rec-color)
    bezier((1.8, 1.6), (1.8, 0.4), (1.4, 1.10), mark: (end: ">", scale: 0.6, fill: rec-color), stroke: rec-color)

    bezier((2.2, 2.4), (2.2, 3.6), (2.6, 3.0), mark: (end: ">", scale: 0.6, fill: rec-color), stroke: rec-color)
    bezier((1.8, 3.6), (1.8, 2.4), (1.4, 3.0), mark: (end: ">", scale: 0.6, fill: rec-color), stroke: rec-color)

    // Draw Nodes
    for p in inputs { circle(p, radius: 0.3, fill: yellow.lighten(50%), stroke: 2pt) }
    for p in hiddens { circle(p, radius: 0.3, fill: yellow.lighten(50%), stroke: 2pt) }
    for p in outputs { circle(p, radius: 0.3, fill: yellow.lighten(50%), stroke: 2pt) }

    // Optional Layer Labels
    content((2, -1.4), text(weight:"bold",size: 9pt, "Recurrent", fill: rec-color))


  })
})
