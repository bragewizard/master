#import "@preview/cetz:0.4.2"

#cetz.canvas(length: 6cm, {
  import cetz.draw: *

  set-style(
    stroke: (thickness: 1.6pt, cap: "round", join: "round"),
    mark: (fill: black, scale: 1.0)
  )

  let get_height_color(i) = {
      // Ensure i is clamped between 0 and 1
      let t = calc.max(0.0, calc.min(1.0, i))

      let r = 0.0
      let g = 0.0
      let b = 0.0

      if (t < 0.33) {
          // Stage 1: Deep Blue to Sea Green
          let local_t = t / 0.33
          r = 0.0
          g = local_t * 180.0
          b = 130.0 + (local_t * 70.0)
      }
      else if (t < 0.66) {
          // Stage 2: Sea Green to Pure Yellow
          let local_t = (t - 0.33) / 0.33
          r = local_t * 255.0
          g = 180.0 + (local_t * 75.0)
          b = 200.0 - (local_t * 200.0)
      }
      else {
          // Stage 3: Yellow to Bright Orange
          let local_t = (t - 0.66) / 0.34
          r = 255.0
          g = 255.0 - (local_t * 155.0)
          b = 0.0
      }

      return rgb(int(r), int(g), int(b))
  }

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
          fill: get_height_color(z1),
          stroke: (thickness: 0.1pt, paint: gray.darken(50%))
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
