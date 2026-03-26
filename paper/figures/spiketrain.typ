#import "@preview/cetz:0.4.2"
#import "@preview/cetz-plot:0.1.3": plot

// 1. Load the data
#let raw-data = csv("actionpotentialfiltered.csv")

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  // Shared Parameters
  let plot-w = 14
  let t-min = 0.114
  let t-max = 0.133

set-style(axes: (
  stroke: (thickness: 1pt, paint: black),
  x: (mark: (end: ">", fill:black, size:1pt)),
  y: (mark: (end: ">", fill:black, size:1pt)),
  tick: (stroke: black + 1pt),
))
  // --- TOP PLOT: THE TRACE ---
  group(name: "trace-group", {
    plot.plot(
      size: (plot-w, 5),
      x-min: t-min, x-max: t-max,
      y-label: [Amplitude (V)],
      x-label: "Time (s)",
      axis-style: "left",
      x-format: v => text(11pt, str(v)),
      y-format: v => text(11pt, str(v)),
      {
        plot.add(
          raw-data.slice(1).map(row => (float(row.at(0)), float(row.at(1)))),
          style: (stroke: blue.darken(40%) + 1.5pt)
        )
      }
    )
  })

  // --- BOTTOM PLOT: THE SPIKE TRAIN ---
  group(name: "raster-group", {
    // Offset the second plot vertically by 5cm
    set-origin((0, -4))
    set-style(axes: (y: (stroke: 0pt, mark:none)))
    plot.plot(
      size: (plot-w, 1),
      x-min: t-min, x-max: t-max,
      y-min: 0.5, y-max: 1.5,
      x-label: "Time (s)",
      x-format: v => text(11pt, str(v)),
      y-label: none,
      y-tick-step: none,
      axis-style: "left",
      {
        // Extract spike times (filtering for "1")
        // Index 6 should be your Ch1_Spike column
        let spike-data = raw-data.slice(1)
          .filter(row => row.at(6) == "1")
          .map(row => (float(row.at(0)), 1))

        plot.add(
          spike-data,
          mark: "|",
          mark-style: (stroke: 2pt + blue.darken(40%)),
          mark-size: 0.4,
          style: (stroke: none)
        )
      }
    )
  })
})
