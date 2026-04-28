#import "@preview/cetz:0.4.2"

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  set-style(
    stroke: (thickness: 2pt, cap: "round", join: "round"),
    mark: (fill: black, scale: 1.0)
  )

  circle((3, 3), radius: 0.07, fill: black)
  circle((3, -1), radius: 0.07, fill: black)
  circle((5.5, 3), radius: 0.07, fill: black)
  circle((5.5, -1), radius: 0.07, fill: black)

  line((0, 3), (6, 3))
  line((6, 3), (8, 3))

  line((3, 3), (3, 2.5))

  group({
    translate((3,1.75))
    rect((-0.25, -0.75), (0.25, 0.75))
    content((0.75, 0), $R$)
  })

  line((3, 1), (3, 0.15))

  group({
    translate((3,0))
    line((-0.5, -0.15), (0.5, -0.15))
    line((-0.2, 0.15), (0.2, 0.15))
    content((1, 0), $V_"rest"$)
  })

  line((3, -0.15), (3, -1))

  line((5.5, 3), (5.5, 1.3))
  line((5.5, 1), (5.5, -1))

  group({
    translate((5,1))
    line((0.0, 0.0), (1, 0.0))
    line((0.0, 0.3), (1, 0.3))
    content((1.5, 0.15), $C$)
  })

  group({
    translate((8,0))
    line((-0.5, -0.15), (0.5, -0.15))
    line((-0.2, 0.15), (0.2, 0.15))
    content((1, 0), $V_"reset"$)
  })

  group({
    translate((8,2))
    line((-0.4, -0.1), (0.0, -0.8))
    circle((0, 0), radius:0.07, fill:black)
    circle((0, -0.8), radius:0.07, fill:black)
    content((0.6, -0.4), $V_"th"$)
  })

  group({
    translate((0,1))
    circle((0, 0), radius:0.5)
    content((1.0, 0.0), $I(t)$)
    line((0, -0.3), (0, 0.3), mark:(end:">"))
  })

  group({
    translate((4,3))
    line((0, 0.0), (0, 0.5))
    line((-0.2, 0.5), (0.2, 0.5))
    line((-0.2, 0.5), (0, 0.8))
    line((0.2, 0.5), (0, 0.8))
    content((0.5, 0.7), $u$)
  })

  group({
    translate((4,-1))
    line((0, 0.0), (0, -0.5))
    line((-0.3, -0.5), (0.3, -0.5))
    line((-0.2, -0.6), (0.2, -0.6))
    line((-0.1, -0.7), (0.1, -0.7))
  })

  line((8, -1), (8, -0.15))
  line((8, 1.2), (8, 0.15))
  line((8, 3), (8, 2))

  line((0, -1), (8, -1))
  line((0, -1), (0, 0.5))
  line((0, 1.5), (0, 3))
})
