#import "@preview/cetz:0.4.2"

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  set-style(
    stroke: (thickness: 1.6pt, cap: "round", join: "round"),
    mark: (fill: black, scale: 1.0)
  )

  let core_r = 0.5
  let spacing = 2.0

  group({
    let pts = ()
    for i in range(2) {
      for j in range(2) {
        let pt = (i * spacing + 0.5, j * spacing + 5.5)
        pts.push(pt)
      }
    }

    rect((0.2, 5.2), (3.8, 8.8), radius: 0.3, fill: none, stroke: (paint: gray, dash: "dashed"))
    content((2.0, 4.6), text(size: 9pt, [Fully Connected\ Neuromorphic Cluster]))

    for (i, p1) in pts.enumerate() {
      for (j, p2) in pts.enumerate() {
        if i < j {
          line(p1.map(x=>x+0.5), p2.map(x=>x+0.5), stroke: (paint: gray, dash: "dotted"))
        }
        rect(p1, (rel:(1,1)), radius: 2pt, fill: yellow.lighten(20%))
        content(p1.map(x=>x+0.5), text(size: 8pt, weight: "bold", [Core]))
      }
    }

    line((4.0, 7.0), (4.6, 7.0), mark: (end: ">"))
    line((4.6, 6.7), (4.0, 6.7), mark: (end: ">"))
    line((5.4, 7.0), (6.0, 7.0), mark: (end: ">"))
    line((6.0, 6.7), (5.4, 6.7), mark: (end: ">"))

  })

  group({
    translate((6, 0.0))
    let pts = ()
    for i in range(2) {
      for j in range(2) {
        let pt = (i * spacing + 0.5, j * spacing + 5.5)
        pts.push(pt)
      }
    }

    rect((0.2, 5.2), (3.8, 8.8), radius: 0.3, fill: none, stroke: (paint: gray, dash: "dashed"))
    content((2.0, 4.6), text(size: 9pt, [Fully Connected\ Neuromorphic Cluster]))

    for (i, p1) in pts.enumerate() {
      for (j, p2) in pts.enumerate() {
        if i < j {
          line(p1.map(x=>x+0.5), p2.map(x=>x+0.5), stroke: (paint: gray, dash: "dotted"))
        }
        rect(p1, (rel:(1,1)), radius: 2pt, fill: yellow.lighten(20%))
        content(p1.map(x=>x+0.5), text(size: 8pt, weight: "bold", [Core]))
      }
    }

    line((4.0, 7.0), (4.6, 7.0), mark: (end: ">"))
    line((4.6, 6.7), (4.0, 6.7), mark: (end: ">"))
    line((5.4, 7.0), (6.0, 7.0), mark: (end: ">"))
    line((6.0, 6.7), (5.4, 6.7), mark: (end: ">"))
  })
  group({
    translate((12.5, 0.0))
    circle((0.0,6.9),radius:1.6pt, fill:black)
    circle((0.3,6.9),radius:1.6pt, fill:black)
    circle((0.6,6.9),radius:1.6pt, fill:black)
  })
  group({
    translate((8.0, 3.0))
    circle((0.0,0.0),radius:1.6pt, fill:black)
    circle((0.0,0.3),radius:1.6pt, fill:black)
    circle((0.0,0.6),radius:1.6pt, fill:black)
  })
  group({
    translate((2.0, 3.0))
    circle((0.0,0.0),radius:1.6pt, fill:black)
    circle((0.0,0.3),radius:1.6pt, fill:black)
    circle((0.0,0.6),radius:1.6pt, fill:black)
  })

    content((6, 10.3), text(size: 9pt, [*AER Bus*]))
    line((0, 10.0), (12, 10.0), mark: (end: ">"))
    line((12, 9.7), (0, 9.7), mark: (end: ">"))
    line((4.85, 5), (4.85, 9.5), mark: (end: ">"))
    line((5.15, 9.5), (5.15, 5), mark: (end: ">"))
    line((10.85, 5), (10.85, 9.5), mark: (end: ">"))
    line((11.15, 9.5), (11.15, 5), mark: (end: ">"))
})
