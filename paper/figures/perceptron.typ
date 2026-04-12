#import "@preview/cetz:0.4.2"
// #show math.equation : set text(font:"TeX Gyre Schola Math", size: 11pt)
// #set page(width: auto, height: auto, margin: 0.6pt)

#show math.equation : set text(size: 16pt, weight: "medium")

#cetz.canvas(length: 1cm,{
  import cetz.draw: *

  set-style(
    stroke: (thickness: 1.6pt, cap: "butt", join: "miter"),
    mark: (fill: black, scale: 1.0)
  )

  rect((-.6,.4),(.6,5.6), fill: blue.lighten(40%), stroke:none, radius:4pt)
  rect((-.2,6.6),(.4,7.4), fill: blue.lighten(60%), stroke:none, radius:2pt)

  rect((.7,.4),(3.3, 5.6), fill:green.lighten(40%),stroke:none, radius:4pt)
  rect((.4,6.6),(1.0,7.4), fill:green.lighten(60%), stroke:none, radius:2pt)

  rect((3.4, .4), (4.6, 5.6), fill:red.lighten(20%),stroke:none, radius:4pt)
  rect((4.2,6.2),(4.7,7.8), fill:red.lighten(50%), stroke:none, radius:2pt)

  for i in range(1,6) {
    bezier((0,i),(4,3),(2,i),(2,3), stroke: (paint:rgb(0,0,0,255)))
    circle((0,i),radius:.3,fill:white,stroke:3pt)
  }
  circle((4,3),radius:.3,fill:white,stroke:3pt)
  line((2,7),(3,7), mark:(end: ">"), stroke:2pt)
  content((0,7),align(center+ top)[$ sum_(i=0)^n x_i w_i $])
  content((5,7),align(center+ top)[$ cases(1 "if" >= b, 0 "if" <b) $])
})
