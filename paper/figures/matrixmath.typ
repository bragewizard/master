#import "@preview/cetz:0.4.2"

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  // Global aesthetics matching minimalist academic style
  set-style(
    stroke: (thickness: 1.2pt, cap: "round", join: "round"),
    mark: (fill: black, scale: 0.8)
  )

  // define reusable node logic for dense grids
  let num_in = 4
  let num_out = 3
  let node_r = 0.3
  let spacing = 1.0

  // 1. Title
  content((5, 12), text(weight: "bold", size: 12pt, "Deep Learning as Matrix Multiplication"))

  // ------------------------------------------------------------------
  // LEFT PANEL: (A) Forward Pass ($y = Wx$)
  // ------------------------------------------------------------------
  group({
    content((2.5, 10.5), text(weight: "bold", size: 10pt, "(A) Forward Pass"))
    
    // --- 1. Nodes (Vectors x and y) ---
    // define input x nodes (White from logic style)
    let pts_x = ()
    for i in range(num_in) {
      let pt = (0, (num_in - 1 - i) * spacing + 5.0)
      circle(pt, radius: node_r, fill: white, stroke: black)
      content((pt.at(0) - 0.7, pt.at(1)), text(size: 8pt, "$x_" + str(i+1) + "$"))
      pts_x.push(pt)
    }
    content((0, 4.5), text(size: 9pt, weight: "bold", fill: blue.darken(20%), "Input\nVector $\\vec{x}$"))

    // define output y nodes (Gray from logic style)
    let pts_y = ()
    for i in range(num_out) {
      let pt = (5, (num_out - 1 - i) * spacing + 5.5)
      circle(pt, radius: node_r, fill: gray.darken(20%), stroke: black)
      content((pt.at(0) + 0.7, pt.at(1)), text(size: 8pt, "$y_" + str(i+1) + "$"))
      pts_y.push(pt)
    }
    content((5, 4.5), text(size: 9pt, weight: "bold", fill: blue.darken(20%), "Output\nVector $\\vec{y}$"))

    // --- 2. Connections (Dense Weight Matrix W) ---
    // define dense connection grid as the "matrix" multiplication
    for (i, p_in) in pts_x.enumerate() {
      for (j, p_out) in pts_y.enumerate() {
        // procedural line calculation: (x,y,W_ij) logic
        // Color: blue for forward
        let dx = p_out.at(0) - p_in.at(0)
        let dy = p_out.at(1) - p_in.at(1)
        let d = calc.sqrt(dx*dx + dy*dy)
        let st_pt = (p_in.at(0) + dx * node_r/d, p_in.at(1) + dy * node_r/d)
        let ed_pt = (p_out.at(0) - dx * node_r/d, p_out.at(1) - dy * node_r/d)
        
        line(st_pt, ed_pt, stroke: (paint: blue.lighten(30%), thickness: 0.8pt))
        // add illustrative marks (Mexican hat or Gabor like math logic)
        circle(((st_pt.at(0)+ed_pt.at(0))/2, (st_pt.at(1)+ed_pt.at(1))/2), radius: 0.04, fill: blue.lighten(50%), stroke: none)
      }
    }
    // define specific synapse labels
    content((3.8, 7.8), text(size: 7pt, "$W_{23}$"))
    line((1.5, 8.5), (3.6, 7.9), mark: (end: ">"), stroke: (paint: gray, thickness: 0.8pt))
    content((1.2, 8.5), text(size: 7pt, "Mexican-hat\nstructure"))

    // --- 3. Mathematical Callout ---
    // procedural math detail, specific element callout
    translate((2.5, 0)) // adjust callout view
    rect((0, 0), (5, 3), fill: gray.lighten(70%), stroke: (paint: gray, thickness: 1.5pt))
    content((2.5, 3.3), text(weight: "bold", size: 9pt, "Operation Logic: $y_i = \sum w_{ij} x_j$"))
    
    line((1.0, 1.5), (1.5, 1.5), mark: (end: ">"), stroke: (paint: gray, thickness: 0.8pt))
    content((0.5, 1.5), text(size: 7pt, "Element\nMultiplication"))
    
    line((2.8, 1.5), (3.3, 1.5), mark: (end: ">"), stroke: (paint: gray, thickness: 0.8pt))
    content((3.8, 1.5), text(size: 7pt, " mexican hat\n summation"))
    
    // Draw matrix example multiplication
    bezier((0.7, 2.5), (4.3, 2.5), (2.5, 0.5), (2.5, 4.5), fill: blue.lighten(60%), stroke: none)
    content((2.5, 1.5), text(weight: "bold", size: 14pt, "$W \cdot \vec{x}$"))
  })

  // ------------------------------------------------------------------
  // RIGHT PANEL: (B) Backward Pass ($\partial L / \partial x = W^T \partial L / \partial y$)
  // ------------------------------------------------------------------
  group({
    translate((10.5, 0)) // Shift right panel
    content((2.5, 10.5), text(weight: "bold", size: 10pt, "(B) Backward Pass"))
    
    // --- 1. Nodes (Gradient Vectors dy and dx) ---
    // defined nodes as red for gradient logic
    let pts_dy = ()
    for i in range(num_out) {
      let pt = (0, (num_out - 1 - i) * spacing + 5.5)
      circle(pt, radius: node_r, fill: red.lighten(80%), stroke: red.darken(10%))
      content((pt.at(0) - 0.7, pt.at(1)), text(size: 8pt, fill: red.darken(10%), "$\\frac{\partial L}{\partial y_" + str(i+1) + "}$"))
      pts_dy.push(pt)
    }
    content((0, 4.5), text(size: 9pt, weight: "bold", fill: red.darken(10%), "Output\nGradient $\\frac{\partial L}{\partial \vec{y}}$"))

    // define nodes as red for gradient logic
    let pts_dx = ()
    for i in range(num_in) {
      let pt = (5, (num_in - 1 - i) * spacing + 5.0)
      circle(pt, radius: node_r, fill: red.darken(10%), stroke: black)
      content((pt.at(0) + 0.7, pt.at(1)), text(size: 8pt, "$\\frac{\partial L}{\partial x_" + str(i+1) + "}$"))
      pts_dx.push(pt)
    }
    content((5, 4.5), text(size: 9pt, weight: "bold", fill: red.darken(10%), "Input\nGradient $\\frac{\partial L}{\partial \vec{x}}$"))

    // --- 2. Connections (Transpose Weight Matrix W^T) ---
    // define dense connection grid logic
    // Color: red for backward/gradient
    for (i, p_in) in pts_dy.enumerate() {
      for (j, p_out) in pts_dx.enumerate() {
        let dx = p_out.at(0) - p_in.at(0)
        let dy = p_out.at(1) - p_in.at(1)
        let d = calc.sqrt(dx*dx + dy*dy)
        let st_pt = (p_in.at(0) + dx * node_r/d, p_in.at(1) + dy * node_r/d)
        let ed_pt = (p_out.at(0) - dx * node_r/d, p_out.at(1) - dy * node_r/d)
        
        line(st_pt, ed_pt, stroke: (paint: red.lighten(30%), thickness: 0.8pt, dash: "dotted"))
        // Mexican hat structure on transpose matrix connections
        circle(((st_pt.at(0)+ed_pt.at(0))/2, (st_pt.at(1)+ed_pt.at(1))/2), radius: 0.04, fill: red.lighten(50%), stroke: none)
      }
    }
    // defined transpose synapse labels
    content((1.2, 7.8), text(size: 7pt, "$W_{23}^T$"))
    line((3.5, 8.5), (1.4, 7.9), mark: (end: ">"), stroke: (paint: gray, thickness: 0.8pt))
    content((3.8, 8.5), text(size: 7pt, " mexicano hat\n structure"))

    // --- 3. Mathematical Callout ---
    translate((2.5, 0)) // adjust callout view
    rect((0, 0), (5, 3), fill: gray.lighten(70%), stroke: (paint: gray, thickness: 1.5pt))
    content((2.5, 3.3), text(weight: "bold", size: 9pt, "Operation Logic: $W^T \cdot \\frac{\partial L}{\partial \\vec{y}}$"))
    
    // Draw matrix example multiplication
    bezier((0.7, 2.5), (4.3, 2.5), (2.5, 0.5), (2.5, 4.5), fill: red.lighten(60%), stroke: none)
    content((2.5, 1.5), text(weight: "bold", size: 14pt, "$W^T \cdot d\\vec{y}$"))
  })

  // ------------------------------------------------------------------
  // SIDE LEGEND: Dense Access / High-Bandwidth Memory
  // ------------------------------------------------------------------
  group({
    translate((7.5, 1))
    content((0, 10.5), text(weight: "bold", size: 9pt, "Dense Access"))
    
    // conceptually draw matrix/memory bar structure
    rect((-0.5, 9.5), (2.5, 10.2), fill: gray.darken(20%), stroke: (paint: black, thickness: 1.2pt))
    content((1.0, 9.85), text(size: 7pt, weight: "bold", "Dense Weight Matrix"))
    
    // Mexican hat illustrative points on the matrix bar like logical inputs
    circle((0.0, 9.85), radius: 0.05, fill: gray.lighten(50%), stroke: none)
    circle((0.5, 9.85), radius: 0.05, fill: gray.lighten(50%), stroke: none)
    circle((1.0, 9.85), radius: 0.05, fill: gray.lighten(50%), stroke: none)
    
    line((0.2, 9.5), (1.0, 8.5), mark: (end: ">", fill: gray), stroke: (thickness: 0.8pt, paint: gray))
    line((1.0, 9.5), (1.0, 8.5), mark: (end: ">", fill: gray), stroke: (thickness: 0.8pt, paint: gray))
    line((1.8, 9.5), (1.0, 8.5), mark: (end: ">", fill: gray), stroke: (thickness: 0.8pt, paint: gray))
    
    content((1.0, 8.2), text(size: 7pt, fill: gray.darken(30%), "Simultaneous\nRead/Write"))
    line((1.0, 7.8), (1.0, 7.3), mark: (end: ">"), stroke: (thickness: 0.8pt, paint: gray))
    
    rect((-0.5, 6.5), (2.5, 7.2), fill: gray.lighten(70%), stroke: (paint: gray, thickness: 1.2pt))
    content((1.0, 6.85), text(size: 7pt, weight: "bold", fill: gray.darken(30%), "High-Bandwidth\nMemory Access"))
  })
})
