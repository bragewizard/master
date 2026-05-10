
#import "@preview/cetz:0.4.2"

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  set-style(
    stroke: (thickness: 1.6pt, cap: "round", join: "round"),
    mark: (fill: black, scale: 1.0),
  )

  let setup-axes(x-label, y-label) = {
    line((-0.2, 0), (6, 0), mark: (end: ">"))
    line((0, -0.2), (0, 6.2), mark: (end: "|"))
    content((6.2, 0), x-label)
    content((0, 6.5), y-label)
  }

  group({
    setup-axes($t$, "Weight")
    group({
      translate((0, 5))
      let spikes = (0.5,)
      line((0, 0), (6, 0), stroke: (paint: gray, dash: "dotted"))
      content((-0.5, 0), [100])
      line((-0.1, 0), (0.1, 0))
      for (i, ts) in spikes.enumerate() {
        line((ts, 0), (ts, 0.5), stroke: (paint: blue.darken(30%)))
        circle((ts, 0.5), radius: 0.07, fill: blue.darken(30%), stroke: none)
      }
    })
    group({
      translate((0, 4))
      let spikes = (1.5,)
      line((0, 0), (6, 0), stroke: (paint: gray, dash: "dotted"))
      content((-0.5, 0), [80])
      line((-0.1, 0), (0.1, 0))
      for (i, ts) in spikes.enumerate() {
        line((ts, 0), (ts, 0.5), stroke: (paint: blue.darken(30%)))
        circle((ts, 0.5), radius: 0.07, fill: blue.darken(30%), stroke: none)
      }
    })
    group({
      translate((0, 3))
      let spikes = (2.5,)
      line((0, 0), (6, 0), stroke: (paint: gray, dash: "dotted"))
      content((-0.5, 0), [60])
      line((-0.1, 0), (0.1, 0))
      for (i, ts) in spikes.enumerate() {
        line((ts, 0), (ts, 0.5), stroke: (paint: blue.darken(30%)))
        circle((ts, 0.5), radius: 0.07, fill: blue.darken(30%), stroke: none)
      }
    })
    group({
      translate((0, 2))
      let spikes = (3.5,)
      line((0, 0), (6, 0), stroke: (paint: gray, dash: "dotted"))
      line((-0.1, 0), (0.1, 0))
      content((-0.5, 0), [40])
      for (i, ts) in spikes.enumerate() {
        line((ts, 0), (ts, 0.5), stroke: (paint: blue.darken(30%)))
        circle((ts, 0.5), radius: 0.07, fill: blue.darken(30%), stroke: none)
      }
    })
    group({
      translate((0, 1))
      let spikes = (4.5,)
      line((0, 0), (6, 0), stroke: (paint: gray, dash: "dotted"))
      line((-0.1, 0), (0.1, 0))
      content((-0.5, 0), [20])
      for (i, ts) in spikes.enumerate() {
        line((ts, 0), (ts, 0.5), stroke: (paint: blue.darken(30%)))
        circle((ts, 0.5), radius: 0.07, fill: blue.darken(30%), stroke: none)
      }
    })
  })
  group({
    translate((8,0))
    setup-axes($t$, "Weight")
    group({
      translate((0, 1))
      let spikes = (0.5,)
      line((0, 0), (6, 0), stroke: (paint: gray, dash: "dotted"))
      content((-0.5, 0), [100])
      line((-0.1, 0), (0.1, 0))
      for (i, ts) in spikes.enumerate() {
        line((ts, 0), (ts, 0.5), stroke: (paint: blue.darken(30%)))
        circle((ts, 0.5), radius: 0.07, fill: blue.darken(30%), stroke: none)
      }
    })
    group({
      translate((0, 2))
      let spikes = (1.5,)
      line((0, 0), (6, 0), stroke: (paint: gray, dash: "dotted"))
      content((-0.5, 0), [80])
      line((-0.1, 0), (0.1, 0))
      for (i, ts) in spikes.enumerate() {
        line((ts, 0), (ts, 0.5), stroke: (paint: blue.darken(30%)))
        circle((ts, 0.5), radius: 0.07, fill: blue.darken(30%), stroke: none)
      }
    })
    group({
      translate((0, 3))
      let spikes = (2.5,)
      line((0, 0), (6, 0), stroke: (paint: gray, dash: "dotted"))
      content((-0.5, 0), [60])
      line((-0.1, 0), (0.1, 0))
      for (i, ts) in spikes.enumerate() {
        line((ts, 0), (ts, 0.5), stroke: (paint: blue.darken(30%)))
        circle((ts, 0.5), radius: 0.07, fill: blue.darken(30%), stroke: none)
      }
    })
    group({
      translate((0, 4))
      let spikes = (3.5,)
      line((0, 0), (6, 0), stroke: (paint: gray, dash: "dotted"))
      line((-0.1, 0), (0.1, 0))
      content((-0.5, 0), [40])
      for (i, ts) in spikes.enumerate() {
        line((ts, 0), (ts, 0.5), stroke: (paint: blue.darken(30%)))
        circle((ts, 0.5), radius: 0.07, fill: blue.darken(30%), stroke: none)
      }
    })
    group({
      translate((0, 5))
      let spikes = (4.5,)
      line((0, 0), (6, 0), stroke: (paint: gray, dash: "dotted"))
      line((-0.1, 0), (0.1, 0))
      content((-0.5, 0), [20])
      for (i, ts) in spikes.enumerate() {
        line((ts, 0), (ts, 0.5), stroke: (paint: blue.darken(30%)))
        circle((ts, 0.5), radius: 0.07, fill: blue.darken(30%), stroke: none)
      }
    })
  })
})
