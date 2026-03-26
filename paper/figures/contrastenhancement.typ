#import "@preview/cetz:0.4.2"

// Note: You can apply a global scale to the canvas to ensure the stacked figure fits on a page
#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  // Global styles
  set-style(
    stroke: (thickness: 1.2pt, cap: "round", join: "round"),
    mark: (fill: black, scale: 0.8)
  )

  group({
    // FIXED: Translated -8.5cm on the y-axis to move it below Panel A
    translate((0.5, -8.5))
    content((2.5, 4.8), text(weight: "bold", size: 10pt, "(B) Contrast Enhancement"))

    // 1. Draw Axes
    line((-0.2, 0), (5.5, 0), mark: (end: ">"), stroke: (thickness: 1pt))
    line((0, -1.8), (0, 3.8), mark: (end: ">"), stroke: (thickness: 1pt))
    content((5.7, 0), "Space")
    content((0, 4.1), "Activity")

    // Reference baseline
    line((-0.2, 0), (5.5, 0), stroke: (dash: "dashed", paint: gray, thickness: 1pt))

    // 2. Input Stimulus Curve (Broad Gaussian)
    let input-pts = range(0, 51).map(i => {
       let x = i * 0.1
       let y = 2.0 * calc.exp(- calc.pow(x - 2.5, 2) / 1.5)
       (x, y)
    })
    line(..input-pts, stroke: (paint: blue.darken(20%), thickness: 1.5pt, dash: "dashed"))

    // 3. Output Response Curve (Difference of Gaussians / Mexican Hat)
    let output-pts = range(0, 51).map(i => {
       let x = i * 0.1
       // Excitatory center minus broad inhibitory surround
       let y = 3.2 * calc.exp(-calc.pow(x - 2.5, 2) / 0.4) - 1.2 * calc.exp(-calc.pow(x - 2.5, 2) / 2.5)
       (x, y)
    })
    line(..output-pts, stroke: (paint: purple.darken(10%), thickness: 1.8pt))

    // 4. Labels and Callouts
    content((4.4, 2.2), text(fill: blue.darken(20%), size: 8pt, "Input\nStimulus"))
    line((3.7, 2.2), (3.3, 1.8), mark: (end: ">"), stroke: (paint: blue.darken(20%), thickness: 0.8pt))

    content((4.4, 3.4), text(fill: purple.darken(10%), size: 8pt, "Output\nResponse"))
    line((3.7, 3.4), (2.9, 2.8), mark: (end: ">"), stroke: (paint: purple.darken(10%), thickness: 0.8pt))

    // Highlight the suppression zones (negative values)
    content((2.5, -1.6), text(fill: red.darken(10%), size: 8pt, "Suppressed by lateral inhibition"))
    line((2.5, -1.3), (1.3, -0.6), mark: (end: ">"), stroke: (paint: red.darken(10%), thickness: 0.8pt, dash: "dotted"))
    line((2.5, -1.3), (3.7, -0.6), mark: (end: ">"), stroke: (paint: red.darken(10%), thickness: 0.8pt, dash: "dotted"))
    })
})
