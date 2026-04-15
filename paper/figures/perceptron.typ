#import "@preview/cetz:0.4.2"
// #show math.equation : set text(font:"TeX Gyre Schola Math", size: 11pt)
// #set page(width: auto, height: auto, margin: 0.6pt)

#show math.equation : set text(size: 16pt, weight: "medium")

#cetz.canvas(length: 1cm,{
  import cetz.draw: *

  set-style(
    stroke: (thickness: 1.6pt, cap: "round", join: "round"),
    mark: (fill: black, scale: 1.0)
  )

  rect((-.6,.4),(.6,5.6), fill: blue.lighten(50%), stroke:blue.darken(50%), radius:4pt)
  rect((-.35,6.6),(.4,7.4), fill: blue.lighten(60%), stroke:blue.darken(50%), radius:2pt)

  rect((.7,.4),(3.3, 5.6), fill:green.lighten(50%),stroke:green.darken(50%), radius:4pt)
  rect((.6,6.6),(1.4,7.4), fill:green.lighten(60%), stroke:green.darken(50%), radius:2pt)

  rect((3.4, .4), (4.6, 5.6), fill:yellow.lighten(50%),stroke:yellow.darken(50%), radius:4pt)
  rect((4.15,6.3),(4.75,7.7), fill:yellow.lighten(60%), stroke:yellow.darken(50%), radius:2pt)

  for i in range(1,6) {
    bezier((0,i),(4,3),(2,i),(2,3) )
    circle((0,i),radius:.3,fill:white, stroke:2pt)
  }
  circle((4,3),radius:.3,fill:white, stroke:2pt)
  line((2,7),(3,7), mark:(end: ">"))
  content((0,7),align(center+ top)[$ sum_(i=0)^n space x_i dot w_i $])
  content((5,7),align(center+ top)[$ cases(space 1 space "if" >= b, space 0 space "if" <b) $])
})
