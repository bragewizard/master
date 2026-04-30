#import "@preview/cetz:0.4.2"

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  set-style(
    stroke: (thickness: 1.6pt, cap: "round", join: "round"),
    mark: (fill: black, scale: 1.0)
  )

  group({
    line((-0.2, 0), (8.5, 0), mark: (end: ">"))
    line((0, -0.2), (0, 4.0), mark: (end: ">"))

    content((9.0, 0), $t$)
    content((0, 4.5), $u(t)$)

    let tresh = 3
    let v_rest = 0.5
    let tau = 1.5
    let amp = 0.8
    let spikes = (1.0, 3.0, 3.8, 4.4, 7.0)

    content((-0.8, tresh), $theta.alt$)
    line((-0.2, tresh), (8.5, tresh), stroke: (dash: "dashed", paint: gray))
    line((0, v_rest), (8.5, v_rest), stroke: (paint: gray, dash: "dashed"))
    content((-0.8, 0.5), $u_"rest"$)

    let pts = ()
    let dt = 0.02
    let v = v_rest
    let j = 0
    for i in range(426) {
        let t = i * dt

        if (v > tresh) {
            v = v_rest
        }
        while j < spikes.len() and spikes.at(j) < t {
            j += 1
            v += amp
        }
        // v += amp * calc.exp(-(t - spikes.at(j)) / tau)
        pts.push((t, v))
    }
    line(..pts, stroke: (paint: green.darken(30%), join: "round"))
  })

  group({
    translate((0, -1.0))

    // Axes
    line((-0.2, 0), (8.5, 0), mark: (end: ">"))
    content((9, -0.0), text(size: 9pt, [$t$]))
    content((-1.0, 0.0), text(size: 9pt, [*Incoming\ Spikes*]))

    let spikes = (1.0, 3.0, 3.8, 4.4, 7.0)

    for (i, ts) in spikes.enumerate() {
        line((ts, 0), (ts, 0.5), stroke: (paint: blue.darken(30%)))
        circle((ts, 0.5), radius: 0.07, fill: blue.darken(30%), stroke: none)

        line((ts, .7), (ts, 1.0), stroke: (paint: gray.lighten(30%), dash: "dotted"))
    }

    content((1.0, -0.4), text(size: 7pt, [$t_1$]))
    content((3.0, -0.4), text(size: 7pt, [$t_2$]))
    content((3.8, -0.4), text(size: 7pt, [$t_3$]))
    content((4.4, -0.4), text(size: 7pt, [$t_4$]))
    content((7.0, -0.4), text(size: 7pt, [$t_5$]))
  })
  group({
    translate((0, -2.4))

    // Axes
    line((-0.2, 0), (8.5, 0), mark: (end: ">"))
    content((9, -0.0), text(size: 9pt, [$t$]))
    content((-1.0, 0.0), text(size: 9pt, [*Outgoing\ Spikes*]))

    let spikes = (4.4,)

    for (i, ts) in spikes.enumerate() {
        line((ts, 0), (ts, 0.5), stroke: (paint: red.darken(30%)))
        circle((ts, 0.5), radius: 0.07, fill: red.darken(30%), stroke: none)

        // line((ts, 1.2), (ts, 2.0), stroke: (paint: gray.lighten(30%), dash: "dotted"))
    }

  })
})
