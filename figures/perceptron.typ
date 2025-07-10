// From https://forum.typst.app/t/how-to-best-draw-a-3d-torus/4744/4
// Note: current settings use about 2 GiB of RAM and 20 s of compilation time.
#import "@preview/cetz:0.4.0"

#set page(width: auto, height: auto, margin: 0.6pt)

#let perceptron(
  fill: black,
  stroke: auto,
) = {
  cetz.draw.line((0,0),(7,0),(8,3))
  cetz.draw.line((0,1),(6.5,1),(8,3))
  cetz.draw.line((0,2),(6,2),(8,3))
  cetz.draw.line((0,3),(6,3),(8,3))
  cetz.draw.line((0,4),(6,4),(8,3))
  cetz.draw.line((0,5),(6.5,5),(8,3))
  cetz.draw.line((0,6),(7,6),(8,3))
  cetz.draw.circle((8,3), fill:white)
}

#cetz.canvas({
  import cetz.draw: *
  perceptron()
})
