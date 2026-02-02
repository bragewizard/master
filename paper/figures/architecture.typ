#import "@preview/cetz:0.4.2"
// #set page(width: auto, height: auto, margin: .5cm)

// #show math.equation: block.with(fill: white, inset: 1pt)
#set text(size: 9pt, font: "GeistMono NF", weight: "medium")

#cetz.canvas({
  import cetz.draw: *

  set-style(
    stroke: (paint: rgb("101010")),
    fill: rgb("D0D0D0"),
  )
  compound-path({
  rect((2.4,0),(rel:(.4,5.2)),radius:0%)
  rect((0,2.4),(rel:(5.2,.4)),radius:0%)
  }, stroke: (paint: rgb("101010")))
  rect((2.8,5.5),(rel:(.4,.4)),radius:15%, name:"r")
  content((v => cetz.vector.add(v, (1.5, 0)), "r.east"), "= interconnect")
  set-style(
    stroke: (paint: rgb("101010")),
    fill: rgb("909090"),
  )

  rect((0,5.5),(rel:(.4,.4)),radius:15%, name:"r")
  content((v => cetz.vector.add(v, (1, 0)), "r.east"), "= neuron")
  
  let pos_x = 0
  let pos_y = 0
  for i in range(8) {
    pos_y = 0
    if (calc.rem(i, 4) == 0 and (i > 0)) {
      pos_x += 0.6
    }
    for j in range(8) {
      if (calc.rem(j, 4) == 0 and (j > 0)) {
        pos_y += 0.6
      }
      rect((pos_x,pos_y),(rel:(.4,.4)),radius:15%, name:"r")
      pos_y += 0.6
    }
    pos_x += 0.6
  }
})
