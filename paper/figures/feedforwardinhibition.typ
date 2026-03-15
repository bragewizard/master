#import "@preview/cetz:0.4.2"

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  // Global styles
  set-style(
    stroke: (thickness: 1.2pt, cap: "round", join: "round"),
    mark: (fill: black, scale: 0.8)
  )

  // Helper function to draw exact node-to-node connections
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
    content((2, 4.2), text(weight: "bold", size: 10pt, "(A) FFI Circuit"))
    
    let p_in = (0, 1)
    let p_inh = (2, 2.5)
    let p_out = (4, 1)

    // 1. Draw Edges
    // Input -> Target (Direct Excitation)
    connect(p_in, p_out, ">", black, line-thickness: 1.5pt)
    
    // Input -> Inhibitor (Excitation)
    connect(p_in, p_inh, ">", black, line-thickness: 1.5pt)
    
    // Inhibitor -> Target (Delayed Inhibition)
    connect(p_inh, p_out, "|", red.darken(10%), line-thickness: 1.5pt)

    // 2. Draw Nodes
    circle(p_in, radius: 0.4, fill: white, stroke: black)
    circle(p_inh, radius: 0.4, fill: gray.lighten(60%), stroke: red.darken(10%))
    circle(p_out, radius: 0.4, fill: gray.darken(20%), stroke: black)

    // 3. Node Labels
    content((0, 0.3), text(size: 9pt, "Input"))
    content((2, 3.2), text(size: 9pt, fill: red.darken(10%), "Inhibitory\nInterneuron"))
    content((4, 0.3), text(size: 9pt, "Target\nNeuron"))

    // 4. Signal flow annotations
    content((2, 0.7), text(size: 8pt, "Direct Excitation"))
    content((3.3, 2.0), text(size: 8pt, fill: red.darken(10%), "Delayed\nInhibition"))
  })

  // ------------------------------------------------------------------
  // RIGHT PANEL: (B) Temporal Window
  // ------------------------------------------------------------------
  group({
    translate((7.5, 0)) // Shift right panel
    content((2.5, 4.2), text(weight: "bold", size: 10pt, "(B) Integration Window"))

    // 1. Draw Axes
    line((-0.2, 0), (5.5, 0), mark: (end: ">"), stroke: (thickness: 1pt))
    line((0, -2.5), (0, 3.2), mark: (end: ">"), stroke: (thickness: 1pt))
    content((5.7, 0), "Time")
    content((0, 3.5), "Conductance / Voltage")

    // Threshold line
    let threshold = 1.2
    line((-0.2, threshold), (5.5, threshold), stroke: (dash: "dashed", paint: gray, thickness: 1pt))
    content((-0.8, threshold), text(size: 8pt, "Threshold"))

    // 2. Procedural Math for Post-Synaptic Potentials (Alpha functions)
    // EPSP (Blue)
    let epsp-pts = range(51).map(i => {
       let x = i * 0.1
       let t = calc.max(0.0, x - 0.5) // Starts at t=0.5
       let y = 15.0 * t * calc.exp(-t * 2.5)
       (x, y)
    })
    line(..epsp-pts, stroke: (paint: blue.lighten(20%), thickness: 1.5pt, dash: "dashed"))

    // IPSP (Red) - Delayed by 0.4ms
    let ipsp-pts = range(51).map(i => {
       let x = i * 0.1
       let t = calc.max(0.0, x - 0.9) // Starts at t=0.9
       let y = -20.0 * t * calc.exp(-t * 2.0)
       (x, y)
    })
    line(..ipsp-pts, stroke: (paint: red.lighten(20%), thickness: 1.5pt, dash: "dashed"))

    // Combined Membrane Potential (Purple) = EPSP + IPSP
    let combined-pts = range(51).map(i => {
       let x = i * 0.1
       let tE = calc.max(0.0, x - 0.5)
       let yE = 15.0 * tE * calc.exp(-tE * 2.5)
       
       let tI = calc.max(0.0, x - 0.9)
       let yI = -20.0 * tI * calc.exp(-tI * 2.0)
       (x, yE + yI)
    })
    line(..combined-pts, stroke: (paint: purple.darken(10%), thickness: 2pt))

    // 3. Labels and Callouts
    content((2.5, 2.2), text(fill: blue.darken(10%), size: 8pt, "EPSP"))
    content((3.0, -1.8), text(fill: red.darken(10%), size: 8pt, "IPSP"))
    
    // Annotate the combined trace
    content((4.2, 1.0), text(fill: purple.darken(10%), size: 8pt, weight: "bold", "Combined\n$V_m$"))
    line((3.5, 0.8), (2.0, -0.2), mark: (start: ">"), stroke: (paint: purple.darken(10%), thickness: 0.8pt))

    // 4. Highlight the narrow integration window
    // The curve crosses threshold around x=0.7 and drops below around x=1.05
    line((0.7, 1.3), (1.05, 1.3), mark: (start: "|", end: "|"), stroke: (thickness: 1pt))
    content((1.8, 1.6), text(size: 8pt, weight: "bold", "Narrow Window\nfor Spiking"))
    line((1.2, 1.5), (0.9, 1.4), mark: (start: ">"), stroke: (thickness: 0.8pt))
    
    // Show the delay
    line((0.5, -0.2), (0.9, -0.2), mark: (start: "|", end: "|"), stroke: (thickness: 1pt))
    content((0.7, -0.5), text(size: 7pt, "Delay"))
  })
})
