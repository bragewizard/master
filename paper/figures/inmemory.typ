#import "@preview/cetz:0.4.2"

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  // Global aesthetics matching minimalist academic style
  set-style(
    stroke: (thickness: 1.2pt, cap: "round", join: "round"),
    mark: (fill: black, scale: 0.8)
  )

  // 1. Title
  content((4, 7.5), text(weight: "bold", size: 10pt, "In-Memory Computing (Crossbar Array)"))

  // 2. The Crossbar Architecture
  let n_lines = 4
  let spacing = 1.2
  let offset_x = 2.0
  let offset_y = 2.0

  // Draw Horizontal Wordlines (Inputs)
  for i in range(n_lines) {
    let y = offset_y + i * spacing
    line((0.5, y), (offset_x + (n_lines - 0.5) * spacing, y), stroke: (paint: blue.darken(20%), thickness: 1.5pt))
    
    // Input labels
    content((0, y), text(size: 9pt, fill: blue.darken(20%), "$V_" + str(n_lines - i) + "$"))
    if i == n_lines - 1 {
      content((-0.5, y + 0.5), text(size: 8pt, weight: "bold", fill: blue.darken(20%), "Input\nVoltages"))
    }
  }

  // Draw Vertical Bitlines (Outputs)
  for j in range(n_lines) {
    let x = offset_x + j * spacing
    line((x, offset_y - 0.5), (x, offset_y + (n_lines - 0.5) * spacing), stroke: (paint: red.darken(10%), thickness: 1.5pt))
    mark((x, offset_y - 0.5), (x, offset_y - 0.8), mark: (end: ">", fill: red.darken(10%)), stroke: none)
    
    // Output labels
    content((x, offset_y - 1.2), text(size: 9pt, fill: red.darken(10%), "$I_" + str(j + 1) + "$"))
    if j == 0 {
      content((x - 0.8, offset_y - 1.2), text(size: 8pt, weight: "bold", fill: red.darken(10%), "Output\nCurrents"))
    }
  }

  // Draw Memory Elements at the Junctions (Weights)
  for i in range(n_lines) {
    for j in range(n_lines) {
      let x = offset_x + j * spacing
      let y = offset_y + i * spacing
      
      // Memory element (e.g., Memristor / ReRAM cell) represented as a diamond
      let r = 0.25
      line((x - r, y), (x, y + r), (x + r, y), (x, y - r), (x - r, y), fill: purple.lighten(60%), stroke: (paint: purple.darken(10%), thickness: 1pt))
    }
  }

  // 3. Highlight a specific junction to explain the physics math
  let h_x = offset_x + 2 * spacing
  let h_y = offset_y + 1 * spacing
  
  // Callout circle
  circle((h_x, h_y), radius: 0.45, stroke: (paint: gray, dash: "dashed", thickness: 1.2pt))
  
  // Callout line
  line((h_x + 0.35, h_y + 0.35), (h_x + 1.2, h_y + 1.2), stroke: (paint: gray, thickness: 1pt))
  
  // Physics Math Explanation Box
  group({
    translate((h_x + 1.5, h_y + 1.5))
    rect((0, 0), (4.5, 2.8), fill: gray.lighten(80%), stroke: (paint: gray, thickness: 1pt))
    
    content((2.25, 2.4), text(weight: "bold", size: 8pt, "Physics as Computation"))
    
    // Multiplication
    content((0.2, 1.7), text(size: 8pt, "1. Multiplication (Ohm's Law)"), anchor: "west")
    content((0.5, 1.2), text(size: 8pt, fill: purple.darken(10%), "$I = V \\times G_{ij}$"), anchor: "west")
    
    // Accumulation
    content((0.2, 0.7), text(size: 8pt, "2. Accumulation (Kirchhoff's Law)"), anchor: "west")
    content((0.5, 0.2), text(size: 8pt, fill: red.darken(10%), "$I_{total} = \\sum I$"), anchor: "west")
  })

  // 4. Contrast Annotation (Solving the Bottleneck)
  group({
    translate((7.5, 0.5))
    line((0, 0), (0, 3), stroke: (paint: purple.darken(10%), thickness: 2pt))
    content((0.5, 2.5), text(weight: "bold", size: 9pt, fill: purple.darken(10%), "The Von Neumann Solution:"), anchor: "west")
    content((0.5, 1.5), text(size: 8pt, "Memory ($G$) and Compute ($I=VG$)\nare physically co-located."), anchor: "west")
    content((0.5, 0.5), text(size: 8pt, weight: "bold", fill: green.darken(20%), "Zero data transport cost."), anchor: "west")
  })
})
