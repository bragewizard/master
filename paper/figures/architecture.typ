#import "@preview/cetz:0.4.2"

#cetz.canvas(length: 1cm, {
  import cetz.draw: *
  set-style(
    stroke: (thickness: 1.6pt, cap: "round", join: "round"),
    mark: (fill: black, scale: 1.0)
  )
  rect((0,0),(1,7),radius:.3,fill:green.lighten(60%), stroke:green.darken(50%))
  rect((3,1.5),(4,5.5),radius:.3,fill:green.lighten(60%), stroke:green.darken(50%))
  rect((6,2.5),(7,4.5),radius:.3,fill:green.lighten(60%), stroke:green.darken(50%))
  bezier((1.5,3),(2.5,4),(2,3),(2,4), mark:(end:">", scale:.6))
  bezier((1.5,4),(2.5,3),(2,4),(2,3), mark:(end:">", scale:.6))
  content((2,5),text(size:9pt,[*Fully\ Connected*]))
  bezier((4.5,3),(5.5,4),(5,3),(5,4), mark:(end:">", scale:.6))
  bezier((4.5,4),(5.5,3),(5,4),(5,3), mark:(end:">", scale:.6))
  content((5,5),text(size:9pt,[*Fully\ Connected*]))
  content((0.5, 3.5),[*784*])
  content((3.5, 3.5),[*128*])
  content((6.5, 3.5),[*10*])

})
