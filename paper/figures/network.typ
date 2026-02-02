#import "@preview/cetz:0.4.2"
// #set text(font: "Geist", weight: "medium", size: 9pt)

#cetz.canvas(length: .7cm, {
  import cetz.draw: *
  set-style(
    stroke:(thickness:2pt)
  )
  for i in range(6) {
    circle((0,i),radius:4pt,fill:black)
    circle((4,i),radius:4pt,fill:black)
    for j in range(6) {
      bezier((0,i),(4,j),(2,i),(2,j), stroke: (paint:rgb(0,0,0,100),thickness:1pt))
    }
  }
  for i in range(1,5) {
    circle((8,i),radius:4pt,fill:black)
    circle((8,i),radius:4pt,fill:black)
    for j in range(6) {
      bezier((4,j),(8,i),(6,j),(6,i), stroke: (paint:rgb(0,0,0,100),thickness:1pt))
    }
  }

})
