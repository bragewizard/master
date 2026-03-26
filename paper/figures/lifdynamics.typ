#import "@preview/cetz:0.4.2"

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  // Global aesthetics matching your minimalist academic style
  set-style(
    stroke: (thickness: 1.0pt, cap: "round", join: "round"),
    mark: (fill: black, scale: 0.8)
  )

  line((-0.1, 0), (8.5, 0), mark: (end: ">"), stroke: (thickness: 1pt))
  line((0, -0.2), (0, 5.0), mark: (end: ">"), stroke: (thickness: 1pt))
  content((9.0, 0), $t$)
  content((0, 5.5), $V(t)$)

  line((-0.2, 2.5), (8.5, 2.5), stroke: (dash: "dashed", paint: gray, thickness: 1pt))
  content((-0.8, 2.5), $theta.alt$)

  line((-0.2, 0.5), (8.5, 0.5), stroke: (dash: "dotted", paint: gray, thickness: 1pt))
  content((-0.8, 0.5), $V_"reset"$)

  let v_rest = 0.5
  let tau = 1.0
  let amp = 1.2
  let spikes = (1.0, 3.0, 3.8, 4.4, 7.0)

  line((0, v_rest), (8.5, v_rest), stroke: (paint: gray, dash: "dashed", thickness: 1pt))

  let v_th = 3.2

  let pts = ()
  let dt = 0.02
  for i in range(426) { // 8.5 / 0.02
      let t = i * dt
      let v = v_rest
      for ts in spikes {
          if t >= ts {
              v += amp * calc.exp(-(t - ts) / tau)
          }
      }
      pts.push((t, v))
  }
  line(..pts, stroke: (paint: blue.darken(10%), thickness: 2pt, join: "round"))
})
