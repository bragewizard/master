#import "@preview/cetz:0.4.2"

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  // Global aesthetics matching your academic style
  set-style(
    stroke: (thickness: 1.2pt, cap: "round", join: "round"),
    mark: (fill: black, scale: 0.8)
  )

  // 1. Title
  content((0, 4.5), text(weight: "bold", size: 10pt, "Spike-Timing-Dependent Plasticity (STDP)"))

  // 2. Draw Axes
  // X-axis: Delta t (Time difference)
  line((-5.5, 0), (5.5, 0), mark: (end: ">"), stroke: (thickness: 1pt))
  content((5.8, -0.4), [$Delta t = t_"post" - t_"pre"$ (ms)])

  // Y-axis: Delta w (Weight change)
  line((0, -3.5), (0, 3.5), mark: (end: ">"), stroke: (thickness: 1pt))
  content((0, 3.8), [Synaptic Weight Change, $Delta w$ (%)])

  // Add origin label
  content((-0.3, -0.3), text(size: 8pt, "0"))

  // 3. Procedural Math Curves for STDP
  // Left side: Post-before-Pre (LTD) -> Red curve
  let ltd_pts = range(-50, 1, step: 2).map(i => {
     let x = i * 0.1
     // Exponential decay for LTD. Amplitude is typically slightly smaller than LTP.
     let y = -2.0 * calc.exp(x / 1.5)
     (x, y)
  })
  line(..ltd_pts, stroke: (paint: red.darken(10%), thickness: 2pt))

  // Right side: Pre-before-Post (LTP) -> Blue curve
  let ltp_pts = range(0, 51, step: 2).map(i => {
     let x = i * 0.1
     // Exponential decay for LTP.
     let y = 2.8 * calc.exp(-x / 1.5)
     (x, y)
  })
  line(..ltp_pts, stroke: (paint: blue.darken(20%), thickness: 2pt))

  // Draw open circles at the discontinuity to show the mathematical boundary
  circle((0, -2.0), radius: 0.08, fill: white, stroke: (paint: red.darken(10%), thickness: 1.2pt))
  circle((0, 2.8), radius: 0.08, fill: white, stroke: (paint: blue.darken(20%), thickness: 1.2pt))

  // 4. Annotations and Callouts
  // LTP Region (Quadrant 1)
  content((2.5, 2.2), [Long-Term Potentiation (LTP)])
  content((2.5, 1.7), [Pre-before-Post ($Delta t > 0$])
  line((2.5, 1.4), (1.5, 0.8), mark: (start: ">"), stroke: (paint: blue.darken(20%), thickness: 0.8pt))

  // LTD Region (Quadrant 3)
  content((-2.5, -2.2), [Long-Term Depression (LTD)])
  content((-2.5, -2.7), [Post-before-Pre ($Delta t < 0$])
  line((-2.5, -1.9), (-1.5, -0.8), mark: (start: ">"), stroke: (paint: red.darken(10%), thickness: 0.8pt))
})
