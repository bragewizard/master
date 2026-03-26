#import "@preview/cetz:0.4.2"

#cetz.canvas(length: 0.8cm, {
  import cetz.draw: *

  // Global styles to match your aesthetic
  set-style(
    stroke: (thickness: 2pt, cap: "round", join: "round"),
    mark: (fill: black, scale: 0.8)
  )

  // Helper function for procedural math curves (Gaussians for tuning)
  let gaussian(x, height, width, center) = {
    height * calc.exp(- calc.pow(x - center, 2) / (2 * calc.pow(width, 2)))
  }

  // ------------------------------------------------------------------
  // LEFT PANEL: (A) Tuning Curves
  // ------------------------------------------------------------------
  group({
    content((4, 6.2), text(weight: "bold", size: 10pt, "(A) Overlapping Tuning Curves"))

    // Draw Axes for (A)
    // Scale X-axis so x units are degrees / 10 (e.g., -9 to 9)
    line((-1.5, 0), (9.5, 0), mark: (end: ">"), stroke: (thickness: 1.5pt))
    content((9.8, 0), "Stimulus Orientation (deg)")

    // Y-axis: Response Rate
    line((4, -0.5), (4, 5.5), mark: (end: ">"), stroke: (thickness: 1pt))
    content((4, 5.8), "Response Rate")

    // Draw reference line for base response
    line((-1.5, 1.2), (9.5, 1.2), stroke: (dash: "dashed", paint: gray, thickness: 1pt))

    // Define preferred angles and colors for neurons
    let pref_angles = (-60, 0, 60)
    let colors = (blue.darken(20%), green.darken(10%), purple.darken(10%))
    let height = 3.8
    let width = 20 // broad width in deg

    // Scale coordinates on plot (x-center, y-offset, colors)
    for (center_x, c) in pref_angles.zip(colors) {
      // FIXED: Added `step: 5` to the range function
      let pts = range(-90, 91, step: 5).map(x => {
        let x_plot = 4 + (x / 10)
        let y_val = 1.2 + gaussian(x, height, width, center_x)
        (x_plot, y_val)
      })

      line(..pts, stroke: (paint: c, thickness: 2pt))

      // Mark peak and add label with preferred orientation
      let peak_x = 4 + (center_x / 10)
      let peak_y = 1.2 + height
      line((peak_x, peak_y), (peak_x, 0), stroke: (paint: c, thickness: 0.8pt, dash: "dotted"))
      content((peak_x, -0.6), text(fill: c, size: 7pt, str(center_x) + "°"))
    }

    // Add callout with arrow for explanation
    let callout_pt = (4 + (pref_angles.at(0) / 10) + 1.2, 1.2 + 2.5)
    content((callout_pt.at(0) + 0.8, callout_pt.at(1) + 0.4), text(size: 8pt, "Preferred\nOrientation"))
    line((callout_pt.at(0) + 0.3, callout_pt.at(1) + 0.3), (callout_pt.at(0) - 0.2, callout_pt.at(1) - 0.2), mark: (end: ">"), stroke: (thickness: 0.8pt))
  })

  // ------------------------------------------------------------------
  // RIGHT PANEL: (B) Population Vector Sum Decoder
  // ------------------------------------------------------------------
  group({
    translate((4, -8)) // Shift right panel
    content((0, 5.2), text(weight: "bold", size: 10pt, "(B) Population Vector Sum"))

    // Large circle to represent orientation space decoder
    circle((0,0), radius: 3, stroke: (thickness: 1.5pt, paint: gray))
    content((0, 3.6), "Orientation space")

    // Draw cross axes with labels
    line((-3.2, 0), (3.2, 0), stroke: (thickness: 0.8pt, paint: gray, dash: "dotted"))
    line((0, -3.2), (0, 3.2), stroke: (thickness: 0.8pt, paint: gray, dash: "dotted"))
    content((3.6, 0), text(size: 8pt, "$0^\circ$"))
    content((-3.6, 0), text(size: 8pt, "$\pm 180^\circ$"))
    content((0, -3.6), text(size: 8pt, "$+90^\circ$"))

    // Define stimulus and vector sum calculation details
    let stimulus_angle = 20
    let pref_angles = (-60, 0, 60)
    let colors = (blue.darken(20%), green.darken(10%), purple.darken(10%))
    let height = 3.8
    let width = 25

    // Calculate vector magnitudes (response rates) based on tuning curves from (A)
    let response_rates = pref_angles.map(center => gaussian(stimulus_angle, height, width, center))
    let total_response = response_rates.fold(0, (sum, val) => sum + val)

    // Normalize vector lengths and plot them
    let scale_vectors = 3.5 / total_response // scale to fit circle
    let vectors = ()

    for (angle, rate, c) in pref_angles.zip(response_rates, colors) {
      let len = rate * scale_vectors
      let v_pt = (len * calc.cos(angle * 1deg), len * calc.sin(angle * 1deg))

      line((0,0), v_pt, mark: (end: ">", fill: c), stroke: (paint: c, thickness: 1pt))
      vectors.push(v_pt)

      // Label vector details
      let callout_pt = (v_pt.at(0) * 1.15, v_pt.at(1) * 1.15)
      content(callout_pt, text(fill: c, size: 7pt, str(angle) + "° pref."))
    }

    // Calculate vector sum
    let pop_v = (vectors.at(0).at(0) + vectors.at(1).at(0) + vectors.at(2).at(0),
                 vectors.at(0).at(1) + vectors.at(1).at(1) + vectors.at(2).at(1))

    // Plot final Population Vector with black marking and label
    line((0,0), pop_v, mark: (end: ">", scale: 0.8, fill: black), stroke: (thickness: 2.2pt, paint: black))
    circle(pop_v, radius: 0.1, fill: black, stroke: none)
    content((pop_v.at(0) + 1.2, pop_v.at(1) + 0.2), text(weight: "bold", size: 8pt, "Population Vector $\vec{V}_{pop}$"))

    // Add annotation with text callout comparing Decoded vs Stimulus
    content((2.5, -2.5), text(size: 8pt, "Decoded $\hat{\theta} \\approx 20^\circ$\nStimulus $\theta = 20^\circ$"))
  })
})
