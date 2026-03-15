#import "@preview/cetz:0.4.2"

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  // Global aesthetics matching your previous figures
  set-style(
    stroke: (thickness: 1.2pt, cap: "round", join: "round"),
    mark: (fill: black, scale: 0.8)
  )

  // Reusable node positions for both panels
  let pre_p = (0, 3) // pre-synaptic terminal center
  let post_d = (0.5, 0.5) // dendritic shaft start
  let post_e = (6, 0.5) // dendritic shaft end

  // 1. Title
  content((3.5, 5.5), text(weight: "bold", size: 10pt, "Structural Plasticity: Synaptogenesis and Pruning"))

  // ------------------------------------------------------------------
  // TOP PANEL: (A) Synaptogenesis
  // ------------------------------------------------------------------
  group({
    content((3.5, 4.5), text(weight: "bold", size: 10pt, "(A) Synaptogenesis"))
    
    // Draw Dendritic Shaft (Purple)
    line(post_d, post_e, stroke: (paint: purple.darken(10%), thickness: 2pt))
    content((post_e.at(0) + 1.2, post_e.at(1)), text(size: 8pt, "Dendrite\n(Post-Synaptic)"))

    // Draw Pre-synaptic Terminal (Blue)
    circle(pre_p, radius: 0.4, fill: blue.lighten(80%), stroke: (paint: blue.darken(20%), thickness: 1.5pt))
    line((pre_p.at(0) + 0.4, pre_p.at(1)), (pre_p.at(0) + 1.5, pre_p.at(1)), stroke: (paint: blue.darken(20%), thickness: 1.2pt))
    content((pre_p.at(0) + 2.0, pre_p.at(1)), text(size: 8pt, "Axon\nTerminal"))

    // Existing connection
    let spine_c1 = (0.5, 0.5 + 1.3)
    let neck_c1 = (0.5, 0.5)
    
    bezier((neck_c1.at(0) - 0.2, neck_c1.at(1)), (neck_c1.at(0) + 0.2, neck_c1.at(1)), (0.3, 1.3), (0.7, 1.3), fill: purple.lighten(80%), stroke: (paint: purple.darken(10%), thickness: 1.2pt))
    circle(spine_c1, radius: 0.25, fill: purple.lighten(80%), stroke: (paint: purple.darken(10%), thickness: 1pt))
    
    // Wire connection point 
    circle((0.2, 2.7), radius: 0.1, fill: black, stroke: 0pt)
    line((0.2, 2.7), (spine_c1.at(0) - 0.1, spine_c1.at(1) + 0.1), stroke: (paint: black, thickness: 1.2pt))

    // 2. Illustrate Synaptogenesis: Spine Growth
    let spine_grow = (4, 0.5 + 1.5)
    let neck_grow = (4, 0.5)
    
    // Dashed outlines indicating growth
    bezier((neck_grow.at(0) - 0.2, neck_grow.at(1)), (neck_grow.at(0) + 0.2, neck_grow.at(1)), (3.8, 1.5), (4.2, 1.5), stroke: (paint: purple.darken(10%), dash: "dashed", thickness: 1.2pt))
    circle(spine_grow, radius: 0.3, stroke: (paint: purple.darken(10%), dash: "dashed", thickness: 1pt))
    
    // Formation and connection point
    circle((4.3, 2.7), radius: 0.1, fill: black, stroke: 0pt)
    
    // Callouts with arrows
    content((5.5, 2.2), text(size: 8pt, "Synaptogenesis:\nNew Spine Growth"))
    line((5.2, 2.0), (4.3, 1.7), mark: (end: ">"), stroke: (thickness: 0.8pt))
    
    line((4.3, 2.7), (4.1, 1.9), mark: (start: ">"), stroke: (paint: black, thickness: 0.8pt))
    content((5.5, 3.2), text(size: 8pt, "Formation of\nNew Synapse"))

    line((0.5, 0.5), (4.0, 0.5), stroke: (paint: purple.darken(10%), dash: "dotted", thickness: 1.2pt))
  })

  // ------------------------------------------------------------------
  // BOTTOM PANEL: (B) Pruning
  // ------------------------------------------------------------------
  group({
    translate((0, -5)) // Offset bottom panel vertically
    content((3.5, 4.5), text(weight: "bold", size: 10pt, "(B) Pruning"))
    
    // Draw Dendritic Shaft (Purple)
    line(post_d, post_e, stroke: (paint: purple.darken(10%), thickness: 2pt))
    content((post_e.at(0) + 1.2, post_e.at(1)), text(size: 8pt, "Dendrite\n(Post-Synaptic)"))

    // Draw Pre-synaptic Terminal (Blue)
    circle(pre_p, radius: 0.4, fill: blue.lighten(80%), stroke: (paint: blue.darken(20%), thickness: 1.5pt))
    line((pre_p.at(0) + 0.4, pre_p.at(1)), (pre_p.at(0) + 1.5, pre_p.at(1)), stroke: (paint: blue.darken(20%), thickness: 1.2pt))
    content((pre_p.at(0) + 2.0, pre_p.at(1)), text(size: 8pt, "Axon\nTerminal"))

    // Established connection (initial state)
    let spine_c1 = (0.5, 0.5 + 1.3)
    let neck_c1 = (0.5, 0.5)
    
    bezier((neck_c1.at(0) - 0.2, neck_c1.at(1)), (neck_c1.at(0) + 0.2, neck_c1.at(1)), (0.3, 1.3), (0.7, 1.3), fill: purple.lighten(80%), stroke: (paint: purple.darken(10%), thickness: 1.2pt))
    circle(spine_c1, radius: 0.25, fill: purple.lighten(80%), stroke: (paint: purple.darken(10%), thickness: 1pt))
    circle((0.2, 2.7), radius: 0.1, fill: black, stroke: 0pt)
    line((0.2, 2.7), (spine_c1.at(0) - 0.1, spine_c1.at(1) + 0.1), stroke: (paint: black, thickness: 1.2pt))

    // 2. Illustrate Pruning: Spine Retraction
    let spine_prune = (4, 0.5 + 1.3)
    let neck_prune = (4, 0.5)
    
    // Dashed outlines indicating retraction action
    bezier((neck_prune.at(0) - 0.2, neck_prune.at(1)), (neck_prune.at(0) + 0.2, neck_prune.at(1)), (3.8, 1.3), (4.2, 1.3), stroke: (paint: purple.darken(10%), dash: "dashed", thickness: 1.2pt))
    circle(spine_prune, radius: 0.25, stroke: (paint: purple.darken(10%), dash: "dashed", thickness: 1pt))
    
    // Retraction and Pruning callouts
    line((4.3, 2.7), (4.2, 1.6), mark: (start: ">"), stroke: (paint: red.darken(10%), thickness: 1.2pt))
    content((5.5, 3.2), text(size: 8pt, fill: red.darken(10%), "Retraction\nand Pruning"))
    
    // FIXED: Using a safe text node instead of an invisible stroke line for the "X" mark
    content((4.1, 2.1), text(weight: "bold", size: 12pt, fill: red.darken(10%), "X"))
    
    // Metabolic Optimization Callout with dotted arrows
    content((2.5, 4.0), text(size: 8pt, "Metabolic Optimization:\nSynapse Removal"))
    line((3.5, 3.8), (4.1, 1.8), mark: (end: ">"), stroke: (thickness: 0.8pt, paint: gray, dash: "dotted"))
    line((1.5, 3.8), (0.7, 1.7), mark: (end: ">"), stroke: (thickness: 0.8pt, paint: gray, dash: "dotted"))

    // Label initial weak state
    content((5.5, 1.5), text(size: 8pt, "Weak Synapse:\nInitial state"))
    line((5.2, 1.3), (4.2, 1.1), mark: (end: ">"), stroke: (thickness: 0.8pt))
    
    line((0.5, 0.5), (4.0, 0.5), stroke: (paint: purple.darken(10%), dash: "dotted", thickness: 1.2pt))
  })
})
