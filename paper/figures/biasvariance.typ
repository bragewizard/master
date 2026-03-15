#import "@preview/cetz:0.4.2"

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  // Global aesthetics matching minimalist academic style
  set-style(
    stroke: (thickness: 1.2pt, cap: "round", join: "round"),
    mark: (fill: black, scale: 0.8)
  )

  // 1. Title
  content((3.5, 5.5), text(weight: "bold", size: 10pt, "The Bias-Variance Tradeoff"))

  // 2. Axes
  // X-axis: Model Complexity
  line((0, 0), (7.5, 0), mark: (end: ">"), stroke: (thickness: 1pt))
  content((3.75, -0.6), text(size: 9pt, "Model Complexity (Degrees of Freedom)"))
  
  // Y-axis: Error
  line((0, 0), (0, 4.5), mark: (end: ">"), stroke: (thickness: 1pt))
  content((-0.2, 2.25), text(size: 9pt, "Error"), angle: 90deg)

  // 3. Procedural Math Curves
  let num_samples = 50
  let x_max = 7.0
  let irr_error = 0.5 // Irreducible error constant

  let bias_pts = ()
  let var_pts = ()
  let total_pts = ()
  
  // Variables to track the global minimum of the total error
  let min_total = 100.0
  let opt_x = 0.0
  let opt_y = 0.0

  for i in range(num_samples + 1) {
    let x = i * (x_max / num_samples)
    
    // Bias^2: Exponential decay
    let bias_sq = 3.5 * calc.exp(-0.8 * x)
    // Variance: Polynomial growth
    let variance = 0.05 * calc.pow(x, 2.5)
    
    let total_error = bias_sq + variance + irr_error

    // Store points for rendering
    bias_pts.push((x, bias_sq))
    var_pts.push((x, variance))
    total_pts.push((x, total_error))

    // Track the minimum for the optimal model marker
    if total_error < min_total {
      min_total = total_error
      opt_x = x
      opt_y = total_error
    }
  }

  // Draw Irreducible Error (Baseline)
  line((0, irr_error), (x_max, irr_error), stroke: (paint: gray, thickness: 1pt, dash: "dashed"))
  content((x_max - 0.5, irr_error + 0.2), text(size: 7pt, fill: gray.darken(30%), "Irreducible\nError"))

  // Draw the three main curves
  line(..bias_pts, stroke: (paint: blue.darken(20%), thickness: 1.5pt))
  line(..var_pts, stroke: (paint: red.darken(10%), thickness: 1.5pt))
  line(..total_pts, stroke: (paint: purple.darken(10%), thickness: 2.5pt))

  // 4. Annotations and Labels
  // Label curves directly (cleaner than a boxed legend)
  content((1.0, 3.0), text(size: 9pt, fill: blue.darken(20%), "Bias$^2$"))
  content((6.2, 3.2), text(size: 9pt, fill: red.darken(10%), "Variance"))
  content((5.5, 4.2), text(size: 9pt, weight: "bold", fill: purple.darken(10%), "Total Error"))

  // Optimal Model Marker (Trough)
  line((opt_x, 0), (opt_x, opt_y), stroke: (paint: black, thickness: 1pt, dash: "dotted"))
  circle((opt_x, opt_y), radius: 0.1, fill: black, stroke: none)
  
  content((opt_x, opt_y + 0.6), text(size: 8pt, weight: "bold", "Optimal\nModel $\\theta^*$"))
  line((opt_x, opt_y + 0.3), (opt_x, opt_y + 0.1), mark: (end: ">"), stroke: (thickness: 0.8pt))

  // Underfitting vs Overfitting Zones
  content((opt_x / 2, -0.2), text(size: 8pt, fill: gray.darken(40%), "Underfitting"))
  content((opt_x + (x_max - opt_x) / 2, -0.2), text(size: 8pt, fill: gray.darken(40%), "Overfitting"))
  
  // Highlight arrows for the zones
  line((opt_x - 0.2, -0.2), (0.2, -0.2), mark: (end: ">"), stroke: (paint: gray.darken(20%), thickness: 0.8pt))
  line((opt_x + 0.2, -0.2), (x_max - 0.2, -0.2), mark: (end: ">"), stroke: (paint: gray.darken(20%), thickness: 0.8pt))
})
