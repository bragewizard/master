#import "@preview/cetz:0.4.2"

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  // Global aesthetics matching minimalist technical style
  set-style(
    stroke: (thickness: 1pt, cap: "round", join: "round"),
    mark: (fill: black, scale: 0.8)
  )

  // 1. Procedural Noise & Math Functions
  let hash(x, y) = {
    let v = calc.sin((x * 12.9898 + y * 78.233) * 1rad) * 43758.5453
    return calc.rem(calc.abs(v), 1.0)
  }

  // Define the high-dimensional surface height mathematically
  let calc_J(x, y) = {
    let base = calc.pow(x, 2) + calc.pow(y, 2)
    let noise = 1.0 + hash(x + 20.0, y + 20.0) * 0.4
    return base * 0.15 * noise
  }

  // 2. Color gradient blend logic
  let get_height_color(i) = {
    let ratio = i / 100.0 // Normalize height
    let r = int(calc.max(0.0, calc.min(255.0, 255.0 - ratio * 255.0))) 
    let g = int(calc.max(0.0, calc.min(255.0, ratio * 200.0)))         
    let b = int(calc.max(0.0, calc.min(255.0, ratio * 255.0)))         
    return rgb(r, g, b)
  }

  // 3. Native 3D Orthographic Projection Context
  ortho(x: 30deg, y: -25deg, {
    
    // --- A. Base Grid (Projected natively on the XY plane) ---
    on-xy(z: -0.5, {
      grid((-4, -4), (4, 4), stroke: gray.lighten(60%) + 0.5pt)
      content((0, 4.5), text(size: 8pt, fill: gray.darken(30%), "$\\theta_1$ Axis"))
      content((4.5, 0), text(size: 8pt, fill: gray.darken(30%), "$\\theta_2$ Axis"))
    })

    // --- B. Draw the 3D Optimization Surface ---
    for x_step in range(-4, 5) {
      for y_step in range(-4, 5) {
        let X = x_step * 1.0
        let Y = y_step * 1.0
        
        let J1 = calc_J(X, Y)
        let p1 = (X, Y, J1)

        // Lines along Y-axis
        if y_step < 4 {
          let Y2 = Y + 1.0
          let J2 = calc_J(X, Y2)
          line(p1, (X, Y2, J2), stroke: (paint: get_height_color(J2 * 20), thickness: 0.8pt))
        }

        // Lines along X-axis
        if x_step < 4 {
          let X2 = X + 1.0
          let J2 = calc_J(X2, Y)
          line(p1, (X2, Y, J2), stroke: (paint: get_height_color(J2 * 20), thickness: 0.8pt))
        }
      }
    }

    // --- C. Global Minimum & Initial Point ---
    let theta_opt = (0.0, 0.0, calc_J(0.0, 0.0)) 
    // FIXED: Call circle directly as a drawing command
    circle(theta_opt, radius: 0.15, fill: black, stroke: none)
    content((0.0, 0.0, -1.2), text(weight: "bold", size: 10pt, "Global Minimum $\\theta^*$"))

    let p_start = (-3.0, -3.0, calc_J(-3.0, -3.0))
    // FIXED: Call circle directly as a drawing command
    circle(p_start, radius: 0.1, fill: black, stroke: none)
    content((-3.0, -3.0, calc_J(-3.0, -3.0) + 1.0), text(size: 8pt, "Initial Point $\\theta_0$"))
    
    // --- D. Gradient-Descent Path (3D Bezier) ---
    bezier(p_start, theta_opt, (-1.0, -2.5, 1.0), (-0.5, -1.0, 0.5), stroke: (paint: red.darken(20%), thickness: 1.5pt))
    content((-1.5, -1.0, 2.0), text(size: 9pt, weight: "bold", fill: red.darken(20%), "Descent Path"))

    // --- E. Gradient Compass (Hovering on a 2D plane above the mesh) ---
    on-xy(z: 4.5, {
      circle((-1.5, -1.5), radius: 0.8, stroke: (paint: gray, thickness: 1.2pt))
      line((-1.7, -1.5), (-1.3, -1.5), stroke: gray) // Crosshair
      line((-1.5, -1.7), (-1.5, -1.3), stroke: gray) // Crosshair
      
      // Arrow pointing towards optimal (0,0) direction
      line((-1.5, -1.5), (-0.8, -0.8), mark: (end: ">", fill: red.darken(10%)), stroke: (paint: red.darken(10%), thickness: 1.5pt))
      
      content((-1.5, -2.6), text(size: 9pt, weight: "bold", fill: red.darken(10%), "Gradient\nCompass $\\nabla J$"))
    })
  })
})
