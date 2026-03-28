#import "@preview/cetz:0.4.2"

#set math.mat(delim: "[")
#cetz.canvas(length: .8cm, {
  import cetz.draw: *

  set-style(
    stroke:(thickness:2pt)
  )

  for i in range(1,5) {
    for j in range(6) {
      bezier((0,i),(6,j),(3,i),(3,j), stroke: (paint:rgb(0,0,0,128),thickness:2pt))
      // bezier((6,j),(12,i),(9,j),(9,i), stroke: (paint:rgb(0,0,0,128),thickness:2pt))
    }
    circle((0,i),radius:.3,fill:white,stroke:3pt)
    // circle((12,i),radius:.3,fill:white,stroke:3pt)
  }
  for i in range(6) {
    circle((6,i),radius:.3,fill:white,stroke:3pt)
  }
  content((0,8), [$mat(w,w,w,w) mat(x;x;x;x)$])
  content((5,8), [
    $ mat(w,w,w,w;
          w,w,w,w;
          w,w,w,w;
          w,w,w,w;
          w,w,w,w;
          w,w,w,w)
      mat(x;x;x;x)
  $])

})
