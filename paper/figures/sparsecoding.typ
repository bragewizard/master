#import "@preview/cetz:0.4.2"  
#import "@preview/cetz-plot:0.1.3": plot, chart

#let data = (
  (1, 7), (2, 1), (3, 6), (4, 7),
  (2, 5), (6, 4), (5, 1), (4, 3), (10, 2), (2, 4),
)

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

    content((-2,2), [A - 9])
    content((-2,1.5), [B - 3])
    content((-2,1), [C - 2])
    content((-2,0.5), [D - 2])
    set-style(axes: (
      stroke: (thickness: 1pt, paint: black),
      x: (mark: (end: ">", fill:black, size:1pt)),
      y: (mark: (end: ">", fill:black, size:1pt)),
      tick: (stroke: black + 1pt),
    ))
    plot.plot(
    size: (10, 2), 
    x-tick-step: 1, 
    y-tick-step: 1, 
    x-min: -1,
    // x-max: 10,
    y-min: 0,
    // y-max: 10,
    x-format: v => text(8pt, str(v)),
    y-format: v => text(8pt, str(v)),
    axis-style: "left", 
    name: "phase", 
    {    
    plot.add(
      data,
      mark: "|",
      mark-style: (stroke:2pt + black),
      line:"linear",
      style: (stroke: none),
    )
  })
})
