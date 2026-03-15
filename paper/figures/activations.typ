#import "@preview/cetz:0.4.2"

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  // Global styles
  set-style(
    stroke: (thickness: 1.2pt, cap: "round", join: "round"),
    mark: (fill: black, scale: 0.8)
  )

  // ------------------------------------------------------------------
  // LEFT PANEL: (A) Sigmoid
  // ------------------------------------------------------------------
  group({
    content((2.5, 5.2), text(weight: "bold", size: 10pt, "(A) Sigmoid"))
    
    // Draw Axes
    line((-2.5, 0), (7.5, 0), mark: (end: ">"), stroke: (thickness: 1pt))
    content((7.8, 0), "Input ($x$)")
    
    line((0, -0.5), (0, 4.5), mark: (end: ">"), stroke: (thickness: 1pt))
    content((0, 4.8), "Output ($sigma(x)$)")

    // Reference lines
    line((-2.5, 4), (7.5, 4), stroke: (paint: gray, dash: "dashed", thickness: 1pt))
    content((-0.8, 4), text(size: 8pt, fill: gray.darken(30%), "1.0"))
    content((-0.3, -0.3), text(size: 8pt, "0"))

    // Procedural math curve for Sigmoid: sigma(x) = 1 / (1 + exp(-x))
    let pts = range(-25, 76, step: 2).map(i => {
       let x = i * 0.1
       let y = 4.0 / (1.0 + calc.exp(-x))
       (x, y)
    })
    line(..pts, stroke: (paint: blue.darken(20%), thickness: 2pt))

    // 1. Highlight Saturation Regions (Negative and Positive)
    let sat_col = red.darken(10%) // Red for gradient killing regions
    
    // Negative Saturation callout
    content((-1.5, 1.2), text(fill: sat_col, size: 8pt, weight: "bold", "Saturates\n(kills gradients)"))
    line((-1.5, 0.9), (-1.2, 0.4), mark: (end: ">"), stroke: (paint: sat_col, thickness: 0.8pt))
    
    // Positive Saturation callout
    content((6.5, 3.2), text(fill: sat_col, size: 8pt, weight: "bold", "Saturates\n(kills gradients)"))
    line((6.5, 3.5), (6.2, 3.8), mark: (end: ">"), stroke: (paint: sat_col, thickness: 0.8pt))
    
    // Highlight these regions visually on the curve
    bezier((-2.5, 0.4), (-0.5, 0.4), (-1.8, 0.2), (-1.2, 0.6), stroke: (paint: sat_col, thickness: 1.5pt, dash: "dotted"))
    bezier((5.5, 3.8), (7.5, 3.8), (6.2, 3.6), (6.8, 4.0), stroke: (paint: sat_col, thickness: 1.5pt, dash: "dotted"))

    // Add illustrative derivative plot (sigma'(x) = sigma(x) * (1 - sigma(x)))
    let d_pts = range(-25, 76, step: 2).map(i => {
       let x = i * 0.1
       let s = 1.0 / (1.0 + calc.exp(-x))
       let ds = s * (1.0 - s) * 4.0 // Scaled derivative
       (x, ds)
    })
    line(..d_pts, stroke: (paint: gray, thickness: 1pt, dash: "dashed"))
    content((2.5, 1.3), text(size: 7pt, fill: gray.darken(30%), "Derivative ($sigma'(x)$)\napproaches 0"))
  })

  // ------------------------------------------------------------------
  // RIGHT PANEL: (B) ReLU
  // ------------------------------------------------------------------
  group({
    translate((0, 8)) // Shift right panel
    content((2.5, 5.2), text(weight: "bold", size: 10pt, "(B) ReLU"))
    
    // Draw Axes
    line((-2.5, 0), (7.5, 0), mark: (end: ">"), stroke: (thickness: 1pt))
    content((7.8, 0), "Input ($x$)")
    
    line((0, -0.5), (0, 4.5), mark: (end: ">"), stroke: (thickness: 1pt))
    content((0, 4.8), "Output ($f(x)$)")
    content((-0.3, -0.3), text(size: 8pt, "0"))

    // Procedural math curve for ReLU: f(x) = max(0, x)
    // Blue for active preservation region
    line((0, 0), (4.5, 4.5), stroke: (paint: blue.darken(20%), thickness: 2pt))
    
    // Grey/Red for inactive/gradient killing region
    line((-2.5, 0), (0, 0), stroke: (paint: red.darken(10%), thickness: 2pt))
    
    // 2. Highlight Preservation Region
    content((4.5, 2.5), text(fill: blue.darken(20%), size: 8pt, weight: "bold", "Preserves\nGradient Magnitude"))
    line((4.5, 2.8), (3.5, 3.5), mark: (end: ">"), stroke: (paint: blue.darken(20%), thickness: 0.8pt))
    
    // 3. Highlight Derivative 
    // Show derivative f'(x) = 1 for x > 0 and 0 for x < 0
    line((0, 1.5), (7.5, 1.5), stroke: (paint: gray, thickness: 1.2pt, dash: "dashed"))
    circle((0, 1.5), radius: 0.1, fill: white, stroke: (paint: gray, thickness: 1.2pt)) // Discontinuity marker
    
    line((-2.5, 0.2), (0, 0.2), stroke: (paint: red.darken(10%), thickness: 1.2pt, dash: "dashed"))
    
    content((2.5, 2.0), text(size: 7pt, fill: gray.darken(30%), "Derivative ($f'(x)$)\nis constant 1.0"))
    line((1.5, 1.8), (0.8, 1.6), mark: (start: ">"), stroke: (thickness: 0.8pt))
  })
})
