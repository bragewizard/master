#import "@preview/cetz:0.4.2"

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  set-style(
    stroke: (thickness: 1.5pt, cap: "round", join: "round"),
    mark: (fill: black, scale: 0.8)
  )

  line((-0.2, 0), (9, 0), mark: (end: ">"), stroke: (thickness: 1pt))
  line((0, -1.5), (0, 3.2), mark: (end: ">"), stroke: (thickness: 1pt))
  content((9.2, 0), $t$)

  let wave-pts = range(0, 85).map(i => {
    let x = i * 0.1
    let y = calc.sin(x * 90deg)
    (x, y)
  })
  line(..wave-pts, stroke: (paint: gray.lighten(10%), thickness: 1.5pt, dash: "dashed"))
  line((4, -1.2), (4, 2.5), stroke: (paint: gray, thickness: 1pt, dash: "dotted"))
  line((8, -1.2), (8, 2.5), stroke: (paint: gray, thickness: 1pt, dash: "dotted"))
  content((2, -1.5), text(weight: "bold", size: 9pt, "Cycle 1"))
  content((6, -1.5), text(weight: "bold", size: 9pt, "Cycle 2"))

  let spike-style = (paint: blue.darken(20%), thickness: 2pt)
  let phase-offset = 1.2
  let y-sine = calc.sin(phase-offset * 90deg)

  line((phase-offset, y-sine), (phase-offset, 2.5), stroke: spike-style)
  circle((phase-offset, 2.5), radius: 0.08, fill: blue.darken(20%), stroke: none)
  content((phase-offset, 2.8), $phi_1$)

  line((4 + phase-offset, y-sine), (4 + phase-offset, 2.5), stroke: spike-style)
  circle((4 + phase-offset, 2.5), radius: 0.08, fill: blue.darken(20%), stroke: none)
  content((4 + phase-offset, 2.8), $phi_2$)

  line((0, -0.5), (phase-offset, -0.5), mark: (start: "|", end: "|"), stroke: (thickness: 1pt))
  content((phase-offset / 2, -0.8), $Delta t$)

  line((4, -0.5), (4 + phase-offset, -0.5), mark: (start: "|", end: "|"), stroke: (thickness: 1pt))
  content((4 + (phase-offset / 2), -0.8), $Delta t$)

  content((12, 1.6), text(size: 9pt, $phi_1 = phi_2 (mod 2pi)$))
  content((12, 0.8), text(size: 8pt, "Indistinguishable without\na cycle counter"))
})
