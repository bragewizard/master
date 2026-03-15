#import "@preview/cetz:0.4.2"

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  // Global styles
  set-style(
    stroke: (thickness: 1.2pt, cap: "round", join: "round"),
    mark: (fill: black, scale: 0.8)
  )

  // Helper function for straight connections
  let connect(p1, p2, mark-type, color, line-thickness: 1.2pt) = {
    let dx = p2.at(0) - p1.at(0)
    let dy = p2.at(1) - p1.at(1)
    let d = calc.sqrt(dx * dx + dy * dy)
    let r = 0.4 // Node radius
    
    let start-x = p1.at(0) + dx * r / d
    let start-y = p1.at(1) + dy * r / d
    let end-x = p2.at(0) - dx * r / d
    let end-y = p2.at(1) - dy * r / d
    
    line((start-x, start-y), (end-x, end-y), 
         mark: (end: mark-type, scale: 0.8, fill: color), 
         stroke: (paint: color, thickness: line-thickness))
  }

  // ------------------------------------------------------------------
  // LEFT PANEL: (A) Neural Circuit
  // ------------------------------------------------------------------
  group({
    content((2, 4.2), text(weight: "bold", size: 10pt, "(A) Feedback Circuit"))
    
    let p_in = (0, 1)
    let p_target = (2, 1)
    let p_inh = (2, 2.8)
    let p_out = (4, 1)

    // 1. Draw Feed-Forward Edges
    connect(p_in, p_target, ">", black, line-thickness: 1.5pt)
    connect(p_target, p_out, ">", black, line-thickness: 1.5pt)

    // 2. Draw Recurrent Feedback Loop (using Bezier curves to avoid overlap)
    let r = 0.4
    // Target -> Inhibitor (Excitation going UP and RIGHT)
    bezier((2 + r*0.7, 1 + r*0.7), (2 + r*0.7, 2.8 - r*0.7), (2.8, 1.9), 
           mark: (end: ">", scale: 0.8, fill: black), stroke: (thickness: 1.5pt))
           
    // Inhibitor -> Target (Inhibition going DOWN and LEFT)
    bezier((2 - r*0.7, 2.8 - r*0.7), (2 - r*0.7, 1 + r*0.7), (1.2, 1.9), 
           mark: (end: "|", scale: 0.8, fill: red.darken(10%)), stroke: (paint: red.darken(10%), thickness: 1.5pt))

    // 3. Draw Nodes
    circle(p_in, radius: r, fill: white, stroke: black)
    circle(p_inh, radius: r, fill: gray.lighten(60%), stroke: red.darken(10%))
    circle(p_target, radius: r, fill: gray.darken(20%), stroke: black)

    // 4. Node Labels
    content((0, 0.3), text(size: 9pt, "Input"))
    content((2, 3.5), text(size: 9pt, fill: red.darken(10%), "Inhibitory\nInterneuron"))
    content((2, 0.3), text(size: 9pt, "Target\nNeuron"))
    
    // 5. Signal annotations
    content((3.3, 1.9), text(size: 8pt, "Recruits"))
    content((0.6, 1.9), text(size: 8pt, fill: red.darken(10%), "Suppresses"))
  })

  // ------------------------------------------------------------------
  // RIGHT PANEL: (B) Rhythm Generation
  // ------------------------------------------------------------------
  group({
    translate((7.5, 0)) // Shift right panel
    content((2.5, 4.2), text(weight: "bold", size: 10pt, "(B) Rhythm Generation"))

    // 1. Draw Axes
    line((-0.2, 0), (6, 0), mark: (end: ">"), stroke: (thickness: 1pt))
    line((0, -1.8), (0, 3.2), mark: (end: ">"), stroke: (thickness: 1pt))
    content((6.2, 0), "Time")
    content((0, 3.5), "Conductance / Voltage")

    // Threshold line
    let threshold = 1.2
    line((-0.2, threshold), (6, threshold), stroke: (dash: "dashed", paint: gray, thickness: 1pt))
    content((-0.8, threshold), text(size: 8pt, "Threshold"))

    // Constant Input Drive line
    line((0, -1.2), (6, -1.2), stroke: (paint: gray.lighten(20%), thickness: 1.5pt))
    content((-0.8, -1.2), text(size: 8pt, fill: gray.darken(30%), "Constant\nInput"))

    // 2. Procedural Math for Oscillation
    let v_pts = ()
    let g_pts = ()
    
    // Generate 3 rhythmic cycles
    for i in range(3) {
      let t0 = i * 1.8 // Cycle length
      
      // Integration phase (charging up)
      for j in range(21) {
        let dt = j * 0.05
        v_pts.push((t0 + dt, 1.2 * (1.0 - calc.exp(-dt * 2.5))))
        g_pts.push((t0 + dt, 0))
      }
      
      // Spike Peak
      v_pts.push((t0 + 1.025, 3.0))
      g_pts.push((t0 + 1.025, 0))
      
      // Fast Reset
      v_pts.push((t0 + 1.05, -0.2))
      g_pts.push((t0 + 1.05, 0.2))
      
      // Feedback Inhibitory Post-Synaptic Potential (IPSP)
      for j in range(1, 16) {
        let dt = j * 0.05
        let ipsp = -12.0 * dt * calc.exp(-dt * 5.0) 
        let g = 12.0 * dt * calc.exp(-dt * 5.0) // Inhibitory Conductance
        
        v_pts.push((t0 + 1.05 + dt, ipsp))
        g_pts.push((t0 + 1.05 + dt, g))
      }
    }
    
    // Plot the generated traces
    line(..g_pts, stroke: (paint: red.lighten(10%), thickness: 1.5pt, dash: "dashed"))
    line(..v_pts, stroke: (paint: purple.darken(10%), thickness: 2pt))

    // 3. Labels and Callouts
    content((2.9, 1.3), text(fill: red.darken(10%), size: 8pt, "Feedback\nInhibition"))
    
    content((4.8, 3.0), text(fill: purple.darken(10%), size: 8pt, weight: "bold", "Target $V_m$"))
    
    // Highlight the rhythmic interval
    line((1.025, 3.4), (2.825, 3.4), mark: (start: "|", end: "|"), stroke: (thickness: 1pt))
    content((1.925, 3.7), text(size: 8pt, "Regulated Firing Rate"))
  })
})
