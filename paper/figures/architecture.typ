#import "@preview/cetz:0.4.2"

#cetz.canvas(length: 1cm, {
  import cetz.draw: *
  set-style(
    stroke: (thickness: 1.6pt, cap: "round", join: "round"),
    mark: (fill: black, scale: 1.0)
  )
  circle((0,-4),radius:.3,fill:yellow.lighten(50%), stroke:2pt)
  circle((0,4),radius:.3,fill:yellow.lighten(50%), stroke:2pt)
  circle((4,-2),radius:.3,fill:yellow.lighten(50%), stroke:2pt)
  circle((4,2),radius:.3,fill:yellow.lighten(50%), stroke:2pt)

  content((0,0),[*700*])
  content((4,0),[*156*])
  content((8,0),[*10*])
  for i in range(3) {
    let dy = i * 0.2
    circle((0,1 + dy), radius:2pt)
  }
  for i in range(3) {
    let dy = i * 0.2
    circle((0,-1 - dy), radius:2pt)
  }

})
