#import "@preview/cetz:0.4.2"

#set math.mat(delim: "[")
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

  content((2,-0.6), [*A*])
  content((8.5,-0.6), [*B*])
  let inputs = ((0, 1), (0, 3))
  let hiddens = ((2, 0.0), (2, 2), (2, 4.0))
  let outputs = ((4, 1.0), (4, 3.0))

  for i in inputs {
    for h in hiddens { connect(i, h) }
  }
  for h in hiddens {
    for o in outputs { connect(h, o) }
  }

  for p in inputs { circle(p, radius: 0.3, fill: green.lighten(70%), stroke:2pt) }
  for p in hiddens { circle(p, radius: 0.3, fill: green.lighten(70%),stroke: 2pt) }
  for p in outputs { circle(p, radius: 0.3, fill: green.lighten(70%), stroke: 2pt) }

  group({
    translate((7,2.7))
    content((-0.9,1.2), [$W^1$])
    content((3.2,1.2), [$W^2$])
    content((0,0), [
      $ mat(w,w;
            w,w;
            w,w)
        mat(x;x) = mat(h;h;h)
    $])
    content((2.0,0), [$ -> $])
    content((4.2,0), [
      $ mat(w,w,w;
            w,w,w)
        mat(h;h;h) = mat(y;y)
    $])

    content((0,-2), [
      $ sum_i sum_j x_j w_(i j) = h_i$
    ])
    content((2,-1.9), [$ -> $])
    content((4.0,-2), [
      $ sum_i sum_j h_i w_(i j) = y_i$
    ])
  })

})
