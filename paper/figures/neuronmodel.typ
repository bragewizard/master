#import "@preview/cetz:0.4.2"
#import "@preview/cetz-plot:0.1.3": plot, chart

#let data = (
  (0, 0), (2, 0), (2, 1), (6,3), (6, 4),(10,9), (10,10),(12,14)
)

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

    set-style(axes: (
      stroke: (thickness: 1pt, paint: black),
      x: (mark: (end: ">", fill:black, size:1pt)),
      y: (mark: (end: ">", fill:black, size:1pt)),
      tick: (stroke: black + 1pt),
    ))
    plot.plot(
    size: (10, 4), 
    x-tick-step: 1, 
    y-tick-step: 1,
    y-min: -2,
    x-format: v => text(8pt, str(v)),
    y-format: v => text(8pt, str(v)),
    axis-style: "left", 
    name: "phase", 
    {    
    plot.add(
      data,
      mark-style: (stroke:2pt + black),
      line:"linear",
      style: (stroke: 2pt),
    )
    plot.add-hline(
      (12),
    )
  })
})
