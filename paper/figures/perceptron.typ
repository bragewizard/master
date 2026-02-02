#import "@preview/cetz:0.4.2"
// #show math.equation : set text(font:"TeX Gyre Schola Math", size: 11pt)
// #set page(width: auto, height: auto, margin: 0.6pt)

#cetz.canvas({
  import cetz.draw: *
  set-style(
    stroke:(thickness:2pt)
  )

  for i in range(4) {
    bezier((0,i -1.5),(4,0),(2, i -1.5),(2,0))
  }
  rect((4,-1),(6,1),radius:.2)
  line((6,0),(8,0))
  rect((8,-1),(10,1),radius:.2)
  line((8.4, -.6),(9,-.6),(9,.6),(9.6,.6))
  line((10,0),(12,0))

  content((5,0),align(center+ top)[$ sum_(i=0)^n x_i w_i $])
  content((1, -1.4),align(center+ top)[#box(fill:white,width:2em, height: 1.3em)[$w_n$]])
  content((1, -.5),align(center+ top)[#box(fill:white,width:3.2em,height: 1.3em)[$w_(n-1)$]])
  content((1, .5),align(center+ horizon)[#box(fill:white,width:2em,height: 1.3em)[$w_1$]])
  content((1, 1.4),align(center+ horizon)[#box(fill:white,width:2em,height: 1.3em)[$w_0$]])
  content((-.4, -1.4),align(center+ horizon)[$x_n$])
  content((-.6, -.5),align(center+ top)[$x_(n-1)$])
  content((-.4, .5),align(center+ top)[$x_1$])
  content((-.4, 1.5),align(center+ top)[$x_0$])
  content((.2,0), [$dots.v$])
  content((12.4,0), [$y$])
})
