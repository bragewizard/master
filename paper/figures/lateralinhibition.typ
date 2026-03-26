#import "@preview/cetz:0.4.2"

// Note: You can apply a global scale to the canvas to ensure the stacked figure fits on a page
#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  // Global styles
  set-style(
    stroke: (thickness: 1.2pt, cap: "round", join: "round"),
    mark: (fill: black, scale: 0.8)
  )

  // Helper function to draw exact node-to-node connections
  let connect(p1, p2, mark-type, color, line-thickness: 1.2pt) = {
    let dx = p2.at(0) - p1.at(0)
    let dy = p2.at(1) - p1.at(1)
    let d = calc.sqrt(dx * dx + dy * dy)
    let r = 0.3 // Node radius

    let start-x = p1.at(0) + dx * r / d
    let start-y = p1.at(1) + dy * r / d
    let end-x = p2.at(0) - dx * r / d
    let end-y = p2.at(1) - dy * r / d

    line((start-x, start-y), (end-x, end-y),
         mark: (end: mark-type, scale: 0.8, fill: color),
         stroke: (paint: color, thickness: line-thickness))
  }

  // ------------------------------------------------------------------
  // TOP PANEL: (A) Neural Circuit
  // ------------------------------------------------------------------
  group({
    // Increased title height for stacked layout
    content((3, 5.2), text(weight: "bold", size: 10pt, "(A) Neural Circuit"))

    let xs = (0, 1.5, 3, 4.5, 6)
    let y_in = 0
    let y_out = 2.5

    // 1. Draw Edges
    // Feed-forward Excitatory
    for x in xs {
      connect((x, y_in), (x, y_out), ">", black)
    }

    // Lateral Inhibitory (from the strongly stimulated center node)
    connect((3, y_in), (1.5, y_out), "|", red.darken(10%), line-thickness: 1.5pt)
    connect((3, y_in), (4.5, y_out), "|", red.darken(10%), line-thickness: 1.5pt)

    // 2. Draw Nodes
    for (i, x) in xs.enumerate() {
      // Input nodes
      circle((x, y_in), radius: 0.3, fill: white, stroke: black)

      // Output nodes (Center is highly active, neighbors are suppressed)
      let out_fill = if i == 2 { gray.darken(40%) } else if i == 1 or i == 3 { white } else { gray.lighten(60%) }
      circle((x, y_out), radius: 0.3, fill: out_fill, stroke: black)
    }

    // 3. Draw Stimulus Inputs
    // Center (Strong)
    line((3, -1.2), (3, -0.4), mark: (end: ">"), stroke: (thickness: 2.5pt, paint: blue.darken(20%)))
    content((3, -1.6), text(size: 8pt, weight: "bold", fill: blue.darken(20%), "Strong\nStimulus"))

    // Neighbors (Weak)
    line((1.5, -0.8), (1.5, -0.4), mark: (end: ">"), stroke: (thickness: 1pt, paint: blue.lighten(20%)))
    line((4.5, -0.8), (4.5, -0.4), mark: (end: ">"), stroke: (thickness: 1pt, paint: blue.lighten(20%)))

    // 4. Legend
    line((0.2, -1.3), (0.8, -1.3), mark: (end: ">"), stroke: black)
    content((1.6, -1.3), text(size: 8pt, "Excitation"))
    line((0.2, -1.8), (0.8, -1.8), mark: (end: "|"), stroke: red.darken(10%))
    content((1.6, -1.8), text(size: 8pt, "Inhibition"))

    // Layer Labels
    content((-1, y_out), text(size: 9pt, "Output\nLayer"))
    content((-1, y_in), text(size: 9pt, "Input\nLayer"))
  })
})
