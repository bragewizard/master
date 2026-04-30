#import "@preview/cetz:0.4.2"

// 1. Load the data
#let raw-data = csv("actionpotentialfiltered.csv")

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  // Shared Parameters
  let plot-w = 14
  let t-min = 0.114
  let t-max = 0.133

  set-style(
    stroke: (thickness: 1.6pt, cap: "round", join: "round"),
    mark: (fill: black, scale: 1.0)
  )

  let spike_data = raw-data.slice(1)
    .filter(row => row.at(6) == "1")
    .map(row => (float(row.at(0)) * 700 - 79.8))

  let trace_data = raw-data.slice(1)
    .map(row => (float(row.at(0)) * 700 - 79.8, float(row.at(1)) * 10 + 3))

  let max_x = trace_data.last().at(0)
  let min_x = trace_data.first().at(0)

  line(..trace_data, stroke:(paint:blue.darken(60%)))

  line((-0.1, 0), (14, 0), mark: (end: ">", fill:black))
  content((13, -1), "Time (ms)")
  for i in range(1, 13) {
    let x = i * 1.03 // Scaling factor
    line((x, -0.1), (x, 0.1))
    content((x, -0.5), [#(i)])
  }
  line((0, -0.1), (0, 0.1))
  content((0, -0.5), "0")

  // Y-axis (Membrane Potential in mV)
  line((0, -0.1), (0, 7), mark: (end: ">", fill:black))
  content((-1.0, 3),angle: 90deg, "Membrane Potential (mV)")
  line((0.1, 3), (14, 3), stroke: (dash: "dashed", paint: gray, ))

  group({
    set-origin((0, -2))
    line((-0.1, 0), (14, 0), mark: (end: ">", fill:black))

    for (i, ts) in spike_data.enumerate() {
        line((ts, 0), (ts, 0.5), stroke: (paint: blue.darken(30%)))
        circle((ts, 0.5), radius: 0.07, fill: blue.darken(30%), stroke: none)

        line((ts, .7), (ts, 1.2), stroke: (paint: gray, dash: "dotted"))
    }

  })
})
