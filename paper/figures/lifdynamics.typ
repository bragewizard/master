#import "@preview/cetz:0.4.2"
#import "@preview/cetz-plot:0.1.3": plot, chart

#cetz.canvas({
  import cetz.draw: *
  set-style(
    stroke:(thickness:2pt)
  )
  rect((0,0),(rel:(10,12)), radius:4pt)
})
