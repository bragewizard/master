#import "@preview/cetz:0.4.2"

#cetz.canvas(length: 1cm, {
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

  let inputs = ((0, 1), (0, 2), (0,3),(0,4))
  let hiddens = ((4, 0), (4, 1), (4, 2), (4,3), (4,4), (4,5))
  let outputs = ((8, 1), (8, 2),(8,3),(8,4))


  for i in inputs {
    for h in hiddens { connect(i, h) }
  }
  for h in hiddens {
    for o in outputs { connect(h, o) }
  }

  for p in inputs { circle(p, radius: 0.3, fill: yellow.lighten(50%), stroke:2pt) }
  for p in hiddens { circle(p, radius: 0.3, fill: yellow.lighten(50%),stroke: 2pt) }
  for p in outputs { circle(p, radius: 0.3, fill: yellow.lighten(50%), stroke: 2pt) }
})
