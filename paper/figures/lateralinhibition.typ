#import "@preview/cetz:0.4.2"

// Note: You can apply a global scale to the canvas to ensure the stacked figure fits on a page
#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  // Global styles
  set-style(
    stroke: (thickness: 1.6pt, cap: "butt", join: "miter"),
    mark: (fill: black, scale: 1.0)
  )

  // Helper function to draw exact node-to-node connections
  let connect(p1, p2, mark-type, color ) = {
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
         stroke: (paint: color))
  }

  group({
    // Increased title height for stacked layout
    content((3, 4.8), text(weight: "bold", size: 10pt, "(A) Neural Circuit"))

    let xs = (0, 1.5, 3, 4.5, 6)
    let y_in = 0
    let y_out = 2.5

    // 1. Draw Edges
    // Feed-forward Excitatory
    for x in xs {
      connect((x, y_in), (x, y_out), ">", black)
    }

    // Lateral Inhibitory (from the strongly stimulated center node)
    connect((3, y_in), (1.5, y_out), "|", red.darken(10%))
    connect((3, y_in), (4.5, y_out), "|", red.darken(10%))

    // 2. Draw Nodes
    for (i, x) in xs.enumerate() {
      // Input nodes
      circle((x, y_in), radius: 0.3, fill: white, stroke: black+3pt)
      circle((x, y_out), radius: 0.3, fill: white, stroke: black+3pt)
    }

    line((3, -1.2), (3, -0.4), mark: (end: ">", fill:blue), stroke: (thickness: 2.5pt, paint: blue))
    content((3, -1.8), text(weight: "bold", fill: blue, "Strong\nStimulus"))

    // Neighbors (Weak)
    line((1.5, -0.8), (1.5, -0.4), mark: (end: ">", fill:blue), stroke: (thickness: 1pt, paint: blue))
    line((4.5, -0.8), (4.5, -0.4), mark: (end: ">", fill:blue), stroke: (thickness: 1pt, paint: blue))

    // 4. Legend
    line((-1.0, -1.5), (0.0, -1.5), mark: (end: ">"), stroke: black)
    content((1.0, -1.5), [Excitation])
    line((-1.0, -2.1), (0.0, -2.1), mark: (end: "|"), stroke: red.darken(10%))
    content((1.0, -2.1), [Inhibition])

    // Layer Labels
    content((-1.1, y_out), text("Output\nLayer"))
    content((-1.1, y_in), text("Input\nLayer"))
  })
  group({
    // FIXED: Translated -8.5cm on the y-axis to move it below Panel A
    translate((7.3, 0))
    content((2.5, 4.8), text(weight: "bold", [(B) Contrast Enhancement]))

    // 1. Draw Axes
    line((-0.2, 0), (5.5, 0), mark: (end: ">"))
    line((0, -0.8), (0, 3.8), mark: (end: ">"))
    content((6.2, 0), "Space")
    content((0, 4.1), "Activity")

    // 2. Input Stimulus Curve (Broad Gaussian)
    let input-pts = range(0, 51).map(i => {
       let x = i * 0.1
       let y = 2.0 * calc.exp(- calc.pow(x - 2.5, 2) / 1.5)
       (x, y)
    })
    line(..input-pts, stroke: (paint: blue.darken(20%), dash: "dashed"))

    // 3. Output Response Curve (Difference of Gaussians / Mexican Hat)
    let output-pts = range(0, 51).map(i => {
       let x = i * 0.1
       // Excitatory center minus broad inhibitory surround
       let y = 3.2 * calc.exp(-calc.pow(x - 2.5, 2) / 0.4) - 1.2 * calc.exp(-calc.pow(x - 2.5, 2) / 2.5)
       (x, y)
    })
    line(..output-pts, stroke: (paint: green.darken(40%)))

    // 4. Labels and Callouts
    content((4.4, 2.2), text(fill: blue.darken(20%), [Input\ Stimulus]))

    content((4.4, 3.4), text(fill: green.darken(40%), [Output\ Response]))

    // Highlight the suppression zones (negative values)
    content((2.5, -1.6), text(fill: red.darken(10%), [Suppressed by lateral inhibition]))
    line((2.5, -1.3), (1.5, -0.8), mark: (end: ">", fill:red),stroke: (paint: red.darken(10%)))
    line((2.5, -1.3), (3.5, -0.8), mark: (end: ">", fill:red),stroke: (paint: red.darken(10%)))
    })
})
