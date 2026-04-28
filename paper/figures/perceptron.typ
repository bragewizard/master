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

  let connect(p1, p2) = {
    let r = 0.3 // Node radius
    let start-x = p1.at(0)
    let start-y = p1.at(1)
    let end-x = p2.at(0)
    let end-y = p2.at(1)
    start-y = (0.06 * end-y) + (0.94 * start-y)
    end-y = (end-y - start-y) * 0.92 + start-y
    end-x = end-x - .4
    start-x = start-x + .4

    bezier((start-x, start-y), (end-x, end-y), ((end-x - start-x)*0.5 + start-x, start-y),((end-x - start-x)*0.7 +start-x,end-y), mark: (end: ">", scale:0.6))
  }

  rect((-.6,.4),(.6,5.6), fill: blue.lighten(50%), stroke:blue.darken(50%), radius:4pt)
  rect((-.35,6.6),(.4,7.4), fill: blue.lighten(60%), stroke:blue.darken(50%), radius:2pt)

  rect((.7,.4),(3.3, 5.6), fill:green.lighten(50%),stroke:green.darken(50%), radius:4pt)
  rect((.6,6.6),(1.4,7.4), fill:green.lighten(60%), stroke:green.darken(50%), radius:2pt)

  rect((3.4, .4), (4.6, 5.6), fill:yellow.lighten(50%),stroke:yellow.darken(50%), radius:4pt)
  rect((4.15,6.3),(4.75,7.7), fill:yellow.lighten(60%), stroke:yellow.darken(50%), radius:2pt)

  line((2,7),(3,7), mark:(end: ">"))
  content((0,7),align(center+ top)[$ sum_(i=0)^n space x_i dot w_i $])
  content((5,7),align(center+ top)[$ cases(space 1 space "if" >= b, space 0 space "if" <b) $])

  let inputs = ((0, 1), (0, 2), (0,3), (0,4),(0,5))
  let outputs = ((4, 3),)

  for i in inputs {
    for o in outputs { connect(i, o) }
  }

  for p in inputs { circle(p, radius: 0.3, fill: white.lighten(50%), stroke:2pt) }
  for p in outputs { circle(p, radius: 0.3, fill: white.lighten(50%), stroke: 2pt) }

})
