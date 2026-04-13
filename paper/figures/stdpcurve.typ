#import "@preview/cetz:0.4.2"

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  // Global aesthetics matching your academic style
  set-style(
    stroke: (thickness: 1.6pt, cap: "round", join: "round"),
    mark: (fill: black, scale: 1.0)
  )

  line((-5.5, 0), (5.5, 0), mark: (end: ">"))
  content((5.8, -0.4), [$Delta t = t_"post" - t_"pre"$ (ms)])

  // Y-axis: Delta w (Weight change)
  line((0, -3.5), (0, 3.5), mark: (end: ">"))
  content((0, 4.0), [Synaptic Weight Change, $Delta w$ (%)])

  let ltd_pts = range(-50, 1, step: 2).map(i => {
     let x = i * 0.1
     let y = -2.0 * calc.exp(x / 1.5)
     (x, y)
  })
  line(..ltd_pts, stroke: (paint: red.darken(10%)))

  let ltp_pts = range(0, 51, step: 2).map(i => {
     let x = i * 0.1
     // Exponential decay for LTP.
     let y = 2.8 * calc.exp(-x / 1.5)
     (x, y)
  })
  line(..ltp_pts, stroke: (paint: blue.darken(20%)))

  // Draw open circles at the discontinuity to show the mathematical boundary
  circle((0, -2.0), radius: 0.08, fill: white, stroke: (paint: red.darken(10%)))
  circle((0, 2.8), radius: 0.08, fill: white, stroke: (paint: blue.darken(20%)))

  // 4. Annotations and Callouts
  // LTP Region (Quadrant 1)
  content((3.8, 2.2), [Long-Term Potentiation (LTP)])
  content((3.5, 1.7), [Pre-before-Post ($Delta t > 0$])
  line((2.2, 1.3), (1.9, 0.8), mark: (start: ">", fill:blue), stroke: (paint: blue))

  // LTD Region (Quadrant 3)
  content((-3.5, -1.6), [Long-Term Depression (LTD)])
  content((-3.8, -2.1), [Post-before-Pre ($Delta t < 0$])
  line((-2.2, -1.3), (-1.9, -0.6), mark: (start: ">", fill:red), stroke: (paint: red))
})
