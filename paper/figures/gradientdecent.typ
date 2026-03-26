#import "@preview/cetz:0.4.2"

#cetz.canvas(length: 6cm, {
  import cetz.draw: *

  set-style(
    stroke: (thickness: 1pt, cap: "round", join: "round"),
    mark: (fill: black, scale: 0.8)
  )

  // let get_height_color(i) = {
  //   let ratio = i / 100.0
  //   let r = int(calc.max(0.0, calc.min(255.0, 255.0 - ratio * 255.0))) 
  //   let g = int(calc.max(0.0, calc.min(255.0, ratio * 200.0)))         
  //   let b = int(calc.max(0.0, calc.min(255.0, ratio * 255.0)))         
  //   return rgb(r, g, b)
  // }

  let franke_function(x, y) = {
      let term1 = 0.75 * calc.exp(-(calc.pow((9 * x - 2), 2)) / 4.0 - calc.pow((9 * y - 2), 2) / 4.0)
      let term2 = 0.75 * calc.exp(-(calc.pow((9 * x + 1), 2)) / 49.0 - (9 * y + 1) / 10.0)
      let term3 = 0.5 * calc.exp(-(calc.pow((9 * x - 7), 2)) / 4.0 - calc.pow((9 * y - 3), 2) / 4.0)
      let term4 = -0.2 * calc.exp(-(calc.pow((9 * x - 4), 2)) - calc.pow((9 * y - 7), 2))
      return term1 + term2 + term3 + term4
    }

  ortho(x: 45deg, y: -20deg, {
    // We use a smaller step to get a "mesh" look
    let steps = 15 
    let delta = 1.0 / steps

    for i in range(0, steps) {
      for j in range(0, steps) {
        let x1 = i * delta
        let x2 = (i + 1) * delta
        let y1 = j * delta
        let y2 = (j + 1) * delta
        
        // Evaluate the 4 corners of the quad
        let z1 = franke_function(x1, y1)
        let z2 = franke_function(x2, y1)
        let z3 = franke_function(x2, y2)
        let z4 = franke_function(x1, y2)

        // Draw the quad face
        // We vary the fill color slightly based on Z to give it depth
        line(
          (x1, z1, y1),
          (x2, z2, y1),
          (x2, z3, y2),
          (x1, z4, y2),
          close: true,
          fill: blue.lighten(z1 * 80%), 
          stroke: (thickness: 0.1pt, paint: white.darken(60%))
        )
      }
    }
    
    // Optional: Add Axes for orientation
    line((0,0,0), (1.2,0,0), mark: (end: ">"))
    line((0,0,0), (0,1.2,0), mark: (end: ">"))
    line((0,0,0), (0,0,1.2), mark: (end: ">"))
    // on-xy(z: 0, {
    //   grid((-4, -4), (4, 4), stroke: gray.lighten(60%) + 0.5pt)
    //   // content((0, 4.5), text(size: 8pt, fill: gray.darken(30%), [$theta_1$ Axis]))
    //   // content((4.5, 0), text(size: 8pt, fill: gray.darken(30%), [$theta_2$ Axis]))
    // })
    // on-xz(y: -4, {
    //   grid((4, 0), (-4, 8), stroke: gray.lighten(60%) + 0.5pt)
    //   // content((0, 4.5), text(size: 8pt, fill: gray.darken(30%), [$theta_1$ Axis]))
    //   // content((4.5, 0), text(size: 8pt, fill: gray.darken(30%), [$theta_2$ Axis]))
    // })
    // on-yz(x: 4, {
    //   grid((0,-4), (8, 4), stroke: gray.lighten(60%) + 0.5pt)
    //   // content((0, 4.5), text(size: 8pt, fill: gray.darken(30%), [$theta_1$ Axis]))
    //   // content((4.5, 0), text(size: 8pt, fill: gray.darken(30%), [$theta_2$ Axis]))
    // })
  })
})
