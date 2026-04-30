#import "@preview/cetz:0.4.2"

// Note: You can apply a global scale to the canvas to ensure the stacked figure fits on a page
#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  // Global styles
  set-style(
    stroke: (thickness: 1.6pt, cap: "round", join: "round"),
    mark: (fill: black, scale: 1.0)
  )

  let connect(p1, p2, color) = {
    let r = 0.3 // Node radius
    let start-x = p1.at(0)
    let start-y = p1.at(1)
    let end-x = p2.at(0)
    let end-y = p2.at(1)
    start-y = (0.17 * end-y) + (0.82 * start-y)
    end-y = (end-y - start-y) * 0.82 + start-y
    end-x = end-x + (start-x - end-x) * 0.1
    start-x = end-x + (start-x - end-x) * 0.85

    bezier(
      (start-x, start-y), (end-x, end-y),
      ((end-x - start-x)*0.4 + start-x, start-y + 0.3),
      ((end-x - start-x)*0.7 +start-x,end-y - 0.3),
      mark: (end: ">", scale:0.6),
      stroke:color
    )
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
      connect((x, y_in), (x, y_out), black)
    }

    // Lateral Inhibitory (from the strongly stimulated center node)
    connect((3, y_in), (1.5, y_out), red.darken(15%))
    connect((3, y_in), (4.5, y_out), red.darken(15%))

    // 2. Draw Nodes
    for (i, x) in xs.enumerate() {
      // Input nodes
      circle((x, y_in), radius: 0.3, fill: green.lighten(70%), stroke: 2pt)
      circle((x, y_out), radius: 0.3, fill: green.lighten(70%), stroke: 2pt)
    }

    line((3, -1.2), (3, -0.4), mark: (end: ">", fill:blue), stroke: (thickness: 2.5pt, paint: blue))
    content((3, -1.8), text("Strong\nStimulus"))

    // Neighbors (Weak)
    line((1.5, -0.8), (1.5, -0.4), mark: (end: ">", fill:blue), stroke: (thickness: 1pt, paint: blue))
    line((4.5, -0.8), (4.5, -0.4), mark: (end: ">", fill:blue), stroke: (thickness: 1pt, paint: blue))

    // 4. Legend
    line((-1.8, -1.5), (-0.8, -1.5), stroke: black)
    content((0.2, -1.5), [Excitation])
    line((-1.8, -2.1), (-0.8, -2.1), stroke: red.darken(15%))
    content((0.2, -2.1), [Inhibition])

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
