#import "@preview/cetz:0.4.2"

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  // Global aesthetics
  set-style(
    stroke: (thickness: 1.2pt, cap: "round", join: "round"),
    mark: (fill: black, scale: 0.8)
  )

  // 1. Title
  content((5, 6), text(weight: "bold", size: 10pt, "The Dual-Stream Hypothesis of Vision"))

  // 2. Define Node Coordinates and Sizes
  let p_retina = (0, 2.5)
  let p_v1 = (3, 2.5)
  let p_dorsal = (8, 4.5)
  let p_ventral = (8, 0.5)

  let box_w = 2.8
  let box_h = 1.0

  // Helper to draw centered text boxes with rounded corners
  let draw_node(center, text_content, fill_color, stroke_color) = {
    let x = center.at(0)
    let y = center.at(1)
    rect((x - box_w/2, y - box_h/2), (x + box_w/2, y + box_h/2), 
         radius: 0.15, fill: fill_color, stroke: (paint: stroke_color, thickness: 1.5pt))
    content(center, text(size: 8pt, weight: "bold", text_content))
  }

  // 3. Draw the Pathways (Arrows)
  // Retina to V1
  line((p_retina.at(0) + box_w/2, p_retina.at(1)), (p_v1.at(0) - box_w/2, p_v1.at(1)), 
       mark: (end: ">"), stroke: (thickness: 1.5pt))
  content((1.5, 2.8), text(size: 7pt, fill: gray.darken(30%), "Optic\nRadiation"))

  // V1 to Dorsal (The "Where" Stream) - Blue
  bezier((p_v1.at(0) + box_w/2, p_v1.at(1) + 0.2), 
         (p_dorsal.at(0) - box_w/2, p_dorsal.at(1)), 
         (5, p_dorsal.at(1)), 
         mark: (end: ">", fill: blue.darken(20%)), stroke: (paint: blue.darken(20%), thickness: 2pt))
  
  content((5.5, 4.2), text(size: 9pt, weight: "bold", fill: blue.darken(20%), "Dorsal Stream"))
  content((5.5, 3.8), text(size: 8pt, fill: blue.darken(20%), "(\"Where\" / \"How\")"))

  // V1 to Ventral (The "What" Stream) - Purple
  bezier((p_v1.at(0) + box_w/2, p_v1.at(1) - 0.2), 
         (p_ventral.at(0) - box_w/2, p_ventral.at(1)), 
         (5, p_ventral.at(1)), 
         mark: (end: ">", fill: purple.darken(10%)), stroke: (paint: purple.darken(10%), thickness: 2pt))
         
  content((5.5, 1.2), text(size: 9pt, weight: "bold", fill: purple.darken(10%), "Ventral Stream"))
  content((5.5, 0.8), text(size: 8pt, fill: purple.darken(10%), "(\"What\")"))

  // 4. Draw the Nodes
  draw_node(p_retina, "Retina & LGN", gray.lighten(70%), gray.darken(20%))
  draw_node(p_v1, "Primary Visual\nCortex (V1)", gray.lighten(70%), black)
  draw_node(p_dorsal, "Posterior Parietal\nCortex", blue.lighten(80%), blue.darken(20%))
  draw_node(p_ventral, "Inferotemporal (IT)\nCortex", purple.lighten(80%), purple.darken(10%))

  // 5. Functional Visualizations (Mini-icons next to terminal nodes)
  
  // Dorsal Visualization: Spatial Navigation / Motion
  group({
    translate((10.5, 4.0)) // Position to the right of the Dorsal box
    // Mini coordinate axes
    line((0, 0), (1.5, 0), mark: (end: ">"), stroke: (thickness: 0.8pt, paint: gray))
    line((0, 0), (0, 1.2), mark: (end: ">"), stroke: (thickness: 0.8pt, paint: gray))
    
    // Trajectory and target
    circle((1.2, 0.8), radius: 0.1, fill: blue.darken(20%), stroke: none) // Target
    bezier((0.2, 0.2), (1.1, 0.7), (0.8, 0.1), mark: (end: ">", fill: blue.darken(20%)), stroke: (paint: blue.darken(20%), dash: "dashed", thickness: 1pt))
    
    content((0.7, -0.4), text(size: 7pt, fill: gray.darken(50%), "Spatial Target\n& Motion"))
  })

  // Ventral Visualization: Object Recognition
  group({
    translate((10.5, 0.0)) // Position to the right of the Ventral box
    // A distinct geometric shape (Cube representation)
    rect((0.2, 0.2), (0.8, 0.8), fill: purple.lighten(50%), stroke: (paint: purple.darken(10%), thickness: 1pt))
    line((0.2, 0.8), (0.5, 1.1), (1.1, 1.1), (1.1, 0.5), (0.8, 0.2), stroke: (paint: purple.darken(10%), thickness: 1pt))
    line((0.8, 0.8), (1.1, 1.1), stroke: (paint: purple.darken(10%), thickness: 1pt))
    
    // Recognition bounding box (corners)
    let c_len = 0.2
    line((0, c_len), (0, 0), (c_len, 0), stroke: (thickness: 1pt, paint: black)) // Bottom Left
    line((1.3, c_len), (1.3, 0), (1.3 - c_len, 0), stroke: (thickness: 1pt, paint: black)) // Bottom Right
    line((0, 1.3 - c_len), (0, 1.3), (c_len, 1.3), stroke: (thickness: 1pt, paint: black)) // Top Left
    line((1.3, 1.3 - c_len), (1.3, 1.3), (1.3 - c_len, 1.3), stroke: (thickness: 1pt, paint: black)) // Top Right

    content((0.7, -0.4), text(size: 7pt, fill: gray.darken(50%), "Object\nRecognition"))
  })
})
