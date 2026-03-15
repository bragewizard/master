#import "@preview/cetz:0.4.2"

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  // Global aesthetics matching minimalist academic style
  set-style(
    stroke: (thickness: 1.2pt, cap: "round", join: "round"),
    mark: (fill: black, scale: 0.8)
  )

  // ------------------------------------------------------------------
  // LEFT PANEL: (A) HIERARCHICAL CONVERGENCE (LGN to V1)
  // ------------------------------------------------------------------
  group({
    content((3.5, 9.5), text(weight: "bold", size: 10pt, "(A) Hierarchical Convergence"))
    
    // --- 1. Visual Space (Spatial Arrangement) ---
    content((3.5, 8.5), text(size: 8pt, fill: gray.darken(30%), "Visual Space"))
    rect((0.5, 5.0), (6.5, 8.2), stroke: (paint: gray, dash: "dashed", thickness: 1pt))

    // Define ON-center/OFF-surround circles
    let rf_y_centers = (7.5, 6.6, 5.7)
    for (i, y) in rf_y_centers.enumerate() {
      // Surround (-)
      circle((3.5, y), radius: 0.45, fill: gray.lighten(60%), stroke: (paint: black, thickness: 1pt))
      // Center (+)
      circle((3.5, y), radius: 0.2, fill: white, stroke: (paint: black, thickness: 1pt))
      content((3.5, y), text(weight: "bold", size: 10pt, "+"))
      
      // Index marker for wiring diagram
      content((0.8, y), text(size: 7pt, fill: gray.darken(20%), "[RF$_" + str(i+1) + "$]"))
    }
    
    // --- 2.Wiring Diagram (Nodes and Synapses) ---
    let p_lgn = rf_y_centers.map(y => (2.5, y - 5.0)) 
    let p_v1 = (5.5, (rf_y_centers.at(0) + rf_y_centers.at(2)) / 2 - 5.0)

    // Synapses from concentric LGN to V1 Simple Cell
    for (i, p) in p_lgn.enumerate() {
      line(p, p_v1, mark: (end: ">", scale: 0.8, fill: black), stroke: (thickness: 1.5pt))
      
      circle(p, radius: 0.25, fill: white, stroke: (paint: black, thickness: 1.2pt))
      content(p, text(size: 8pt, "LGN$_" + str(i+1) + "$"))
    }
    
    // V1 Simple Cell Node
    circle(p_v1, radius: 0.4, fill: gray.darken(20%), stroke: (paint: black, thickness: 1.5pt))
    content(p_v1, text(size: 9pt, weight: "bold", "V1$_{SC}$"))

    // Layer Labels
    content((-0.8, rf_y_centers.at(1) - 5.0), text(size: 9pt, "LGN\nLayer"))
    content((6.8, p_v1.at(1)), text(size: 9pt, "V1\nCortex"))

    // Signal explanation
    line((1.5, rf_y_centers.at(1)), (2.5, rf_y_centers.at(1)), mark: (end: ">"), stroke: (paint: gray, thickness: 1pt, dash: "dotted"))
    content((2.0, rf_y_centers.at(1) + 0.3), text(size: 7pt, fill: gray.darken(30%), "Input\nMapping"))
  })

  // ------------------------------------------------------------------
  // RIGHT PANEL: (B) EMERGENT FEATURE SELECTIVITY
  // ------------------------------------------------------------------
  group({
    translate((10.5, 0)) // Offset right panel
    content((3.5, 9.5), text(weight: "bold", size: 10pt, "(B) Emergent Selectivity"))

    // --- 1. Decoded Receptive Field ---
    content((1.8, 8.5), text(size: 8pt, fill: gray.darken(30%), "Emergent RF$_{V1}$"))
    rect((0, 5.0), (3.6, 8.2), stroke: (paint: gray, dash: "dashed", thickness: 1pt))
    
    // Visualize the vertical oriented bar
    bezier((1.4, 5.5), (2.2, 5.5), (1.8, 4.5), (1.8, 8.7), stroke: (paint: gray.lighten(60%), thickness: 0pt))
    line((1.4, 5.5), (1.4, 7.7), (2.2, 7.7), (2.2, 5.5), (1.4, 5.5), fill: gray.lighten(60%), stroke: none)
    
    content((1.8, 6.6), text(weight: "bold", size: 16pt, "+"))
    content((0.8, 6.6), text(weight: "bold", size: 16pt, "-"))
    content((2.8, 6.6), text(weight: "bold", size: 16pt, "-"))
    
    // --- 2. Tuning Curve ---
    let tune_center = 0 
    let tune_width = 20 
    let tune_height = 3.5

    // FIXED: Added `step: 5`
    let tune_pts = range(-60, 61, step: 5).map(theta => {
        let x = theta / 20 + 2.0 
        let y = tune_height * calc.exp(-calc.pow(theta - tune_center, 2) / (2 * calc.pow(tune_width, 2)))
        (x, y)
    })
    
    group({
      translate((0, -0.5)) 
      line((0.5, 0), (3.5, 0), mark: (end: ">", fill: black), stroke: (thickness: 1pt))
      line((2.0, -0.1), (2.0, 3.8), mark: (end: ">", fill: black), stroke: (thickness: 1pt))
      content((3.8, 0), text(size: 8pt, "$\\theta$"))
      content((2.0, 4.1), text(size: 8pt, "Freq $(f)$"))
      
      content((0.8, -0.3), text(size: 7pt, "$-90^\circ$"))
      content((3.2, -0.3), text(size: 7pt, "$+90^\circ$"))
      
      line(..tune_pts, stroke: (paint: blue.darken(20%), thickness: 2pt, join: "round"))
      
      circle((2.0, tune_height), radius: 0.1, fill: black, stroke: none)
      content((3.2, tune_height + 0.3), text(size: 8pt, weight: "bold", fill: blue.darken(20%), "Optimal $\\theta$"))
      line((2.1, tune_height + 0.1), (2.6, tune_height + 0.3), mark: (start: ">"), stroke: (paint: blue.darken(20%), thickness: 0.8pt))
    })

    // --- 3. Example Voltage Traces ---
    group({
      translate((4.5, -0.5))
      line((-0.2, 0), (2.8, 0), mark: (end: ">", fill: black), stroke: (thickness: 1pt))
      line((0, -0.2), (0, 3.8), mark: (end: ">", fill: black), stroke: (thickness: 1pt))
      content((3.0, 0), text(size: 8pt, "$t$"))
      content((0, 4.1), text(size: 8pt, "$V_m$"))
      
      let t_pts = range(0, 41).map(i => i * 0.1)
      let weak_sine = t_pts.map(t => 0.5 * calc.sin(t * 500deg))
      
      // Optimal Trace
      let strong_pts = t_pts.zip(weak_sine).map(tuple => {
          let (t, sine) = tuple
          let spikes = if calc.abs(calc.rem(t, 1.2)) < 0.1 { 3.0 } else { 0.0 }
          (t, 2.0 + sine + spikes)
      })
      
      // Orthogonal Trace 
      let weak_pts = t_pts.zip(weak_sine).map(tuple => {
          let (t, sine) = tuple
          (t, 0.5 + sine)
      })
      
      line(..strong_pts, stroke: (paint: blue.darken(20%), thickness: 1.2pt, join: "round"))
      line(..weak_pts, stroke: (paint: purple.darken(10%), thickness: 1.2pt, join: "round"))
      
      content((2.5, 3.2), text(size: 8pt, fill: blue.darken(20%), "Pref. $\\theta$"))
      content((2.5, 1.0), text(size: 8pt, fill: purple.darken(10%), "Ortho. $\\theta$"))
    })
  })

  // Global callouts linking panels A and B
  line((5.8, 0.2), (9.8, 0.2), mark: (end: ">"), stroke: (paint: gray, thickness: 1pt, dash: "dashed"))
  content((7.8, 0.5), text(size: 7pt, fill: gray.darken(30%), "Output Mapping\n$f \\propto \\sum w_i \\cdot I_i$"))
})
