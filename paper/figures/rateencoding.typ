#import "@preview/cetz:0.4.2"

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  set-style(
    stroke: (thickness: 1.6pt, cap: "butt", join: "miter"),
    mark: (fill: black, scale: 1.0)
  )

  let setup-axes(x-label, y-label) = {
    line((-0.2, 0), (10, 0), mark: (end: ">"))
    line((0, -0.2), (0, 5.2), mark: (end: ">"))
    content((10.2, 0), x-label)
    content((0, 5.5), y-label)
  }

  let spike-data = (
    // Row 4: High Value (9) -> High firing rate (10 spikes)
    (0.5, 4), (1.5, 4), (2.5, 4), (3.5, 4), (4.5, 4),
    (5.5, 4), (6.5, 4), (7.5, 4), (8.5, 4), (9.5, 4),

    // Row 3: Medium Value (6) -> Medium firing rate (6 spikes)
    (1.2, 3), (2.8, 3), (4.4, 3), (6.0, 3), (7.6, 3), (9.2, 3),

    // Row 2: Low Value (3) -> Low firing rate (3 spikes)
    (2.0, 2), (5.0, 2), (8.0, 2),

    // Row 1: Very Low Value (1) -> Minimal firing rate (1 spike)
    (4.5, 1)
  )

  setup-axes($t$, "neuron")
  for (i) in range(spike-data.len()) {
    let data = spike-data.at(i)
    line(data, (data.at(0), data.at(1)+.2), stroke:3pt)
  }

})
