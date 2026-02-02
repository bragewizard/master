
#import "@preview/cetz:0.4.2"
// #show math.equation : set text(font:"TeX Gyre Schola Math", size: 11pt)
// #set page(width: auto, height: auto, margin: 0.6pt)

#cetz.canvas({
  import cetz.draw: *
  set-style(
    stroke:(thickness:2pt)
  )
  rect((1,1),(rel:(2,2)), radius:4pt)
  rect((5,1),(rel:(2,2)), radius:4pt)
  rect((5,4),(rel:(2,2)), radius:4pt)
  rect((1,4),(rel:(2,2)), radius:4pt)
  rect((0,0),(rel:(4,7)), radius:4pt)
})
