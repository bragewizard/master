#import "@preview/cetz:0.4.2"

// Font and text settings as requested
// #set text(font: "GeistMono NF", weight: "medium", size: 9pt)

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  set-style(
    stroke: (thickness: 1.5pt, cap: "round", paint:black),
    mark: (fill: black, scale: 0.8)
  )

  let draw-gate(name, offset, logic, line-coords: none) = {

    translate(offset)
    group({
      content((0.3, 2.4), text(weight: "bold", size: 9pt, name))

      line((-0.5, 0), (1.8, 0), mark: (end: ">", fill:black), stroke: (thickness: 1pt, paint: black))
      line((0, -0.5), (0, 1.8), mark: (end: ">", fill:black), stroke: (thickness: 1pt, paint: black))
      content((2, 0), $x_1$)
      content((0, 2), $x_2$)

      if line-coords != none {
        line(..line-coords, stroke: (dash: "dashed", paint: green, thickness: 1.5pt))
        content((1.2, 1.7), text(fill: green, size: 8pt, "Separable"))
      } else {
        content((2.4, 1.7), text(fill: red, size: 8pt, "Not Linearly Separable"))
      }

      let inputs = ((0,0), (0,1), (1,0), (1,1))

      for i in range(4) {
        let pt = inputs.at(i)
        let is-active = logic.at(i)

        if is-active {
          circle(pt, radius: 0.15, fill: black, stroke: none)
          content((pt.at(0) + 0.3, pt.at(1) + 0.3), text(size: 7pt, "(1)"))
        } else {
          circle(pt, radius: 0.15, fill: white, stroke: black)
          content((pt.at(0) - 0.3, pt.at(1) - 0.3), text(size: 7pt, "(0)"))
        }
      }
    })
  }

  draw-gate("AND Gate", (0, 0), (false, false, false, true), line-coords: ((1.5, 0), (0, 1.5)))
  draw-gate("OR Gate", (3.5, 0), (false, true, true, true), line-coords: ((0.5, 0), (0, 0.5)))
  draw-gate("XOR Gate", (3.5, 0), (false, true, true, false), line-coords: none)

})
