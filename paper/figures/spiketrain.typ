#import "@preview/cetz:0.4.2"
#import "@preview/cetz-plot:0.1.3": plot, chart

#let data = (
  (1, 7), (2, 1), (3, 6), (4, 7),
  (2, 5), (6, 4), (5, 1), (4, 3), (10, 2), (2, 4),
)

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

    let common-x = 1
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

  group(
    name: "g2",
    {
      set-origin((0, 4))
      plot.plot(
        axis-style: "left", 
        name: "plot",
        x-tick-step: 1, 
        y-tick-step: 1, 
        x-min: -1,
        y-min: 0,
        size: (10, 2),
        {

          let f(x) = -(calc.pow(calc.e, 2 * x) - 1) / (calc.pow(calc.e, 2 * x) + 1)
          plot.add(
            domain: (-3, 3),
            f,
          )
          plot.add-anchor("p", (common-x, f(common-x)))
        },
      )
    },
  )
})



// #cetz.canvas(
//   length: 72pt,
//   {
//     import cetz.draw: *
//     import cetz-plot: *

//     let common-x = 1

//     plot.plot(
//       name: "p1",
//       x: 4,
//       y: 4,
//       size: (4, 2),
//       asix-style: "scientific",
//       {
//         let f(x) = {
//           if (x < -1.8) {
//             40
//           } else {
//             -0.2 * calc.pow(x + 1.8, 2) + 40
//           }
//         }

//         plot.add(
//           domain: (-3, 3),
//           f,
//         )

//         plot.add-anchor("p", (common-x, f(common-x)))
//       },
//     )

//     group(
//       name: "g2",
//       {
//         set-origin((0, -2.5))
//         plot.plot(
//           name: "plot",
//           x: 4,
//           y: 4,
//           size: (4, 2),
//           asix-style: "scientific",
//           {

//             let f(x) = -(calc.pow(calc.e, 2 * x) - 1) / (calc.pow(calc.e, 2 * x) + 1)

//             plot.add(
//               domain: (-3, 3),
//               f,
//             )

//             plot.add-anchor("p", (common-x, f(common-x)))
//           },
//         )
//       },
//     )

//     line("p1.p", "g2.plot.p", stroke: (dash: "dashed"))
//   },
// )
