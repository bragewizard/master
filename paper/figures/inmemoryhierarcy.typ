#import "@preview/cetz:0.4.2"

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  // Global aesthetics matching minimalist academic style
  set-style(
    stroke: (thickness: 1.2pt, cap: "round", join: "round"),
    mark: (fill: black, scale: 0.8)
  )

  // define reusable core block logic
  let core_r = 0.5
  let spacing = 2.0

  // 1. Title
  content((5, 12), text(weight: "bold", size: 12pt, "Hierarchical Architecture & Address Event Representation (AER)"))

  // ------------------------------------------------------------------
  // TOP PANEL: (A) Hierarchical System
  // ------------------------------------------------------------------
  group({
    content((2.5, 10.5), text(weight: "bold", size: 10pt, "(A) Hierarchical System"))
    
    // --- 1. Nodes (Neuromorphic Cores & Clusters) ---
    let pts = ()
    for i in range(2) {
      for j in range(2) {
        let pt = (i * spacing + 1.0, j * spacing + 6.0)
        circle(pt, radius: core_r, fill: purple.lighten(80%), stroke: (paint: purple.darken(10%), thickness: 1.2pt))
        content(pt, text(size: 8pt, weight: "bold", fill: purple.darken(10%), "Core"))
        pts.push(pt)
      }
    }
    // defined Cluster boundary
    rect((0.2, 5.2), (3.8, 8.8), radius: 0.3, fill: none, stroke: (paint: gray, thickness: 1.5pt, dash: "dashed"))
    content((2.0, 5.0), text(size: 9pt, weight: "bold", fill: gray.darken(30%), "Neuromorphic Cluster"))

    // --- 2. Connections (Network-on-Chip) ---
    for (i, p1) in pts.enumerate() {
      for (j, p2) in pts.enumerate() {
        if i < j {
          line(p1, p2, stroke: (paint: gray.lighten(60%), thickness: 1.0pt, dash: "dotted"))
        }
      }
    }
    
    // Router nodes and inter-cluster connections
    circle((5.0, 7.0), radius: core_r, fill: white, stroke: black)
    content((5.0, 7.0), text(size: 8pt, weight: "bold", "Router"))
    
    line((3.5, 7.0), (4.5, 7.0), mark: (end: ">"), stroke: (paint: gray, thickness: 1.2pt))
    line((5.5, 7.0), (6.5, 7.0), mark: (end: ">"), stroke: (paint: gray, thickness: 1.2pt))
    
    circle((7.0, 7.0), radius: core_r, fill: gray.lighten(80%), stroke: black)
    content((7.0, 7.0), text(size: 8pt, "Off-Chip"))
    
    // --- 3. Spike annotations (Input/Output) ---
    line((0.8, 7.0), (0.2, 7.0), mark: (end: ">"), stroke: (paint: blue.darken(20%), thickness: 1.5pt))
    content((0.5, 7.3), text(size: 8pt, fill: blue.darken(20%), "AER Spikes"))
    
    line((3.2, 7.0), (3.8, 7.0), mark: (end: ">"), stroke: (paint: red.darken(10%), thickness: 1.5pt))
    content((3.5, 7.3), text(size: 8pt, fill: red.darken(10%), "AER Spikes"))
  })

  // ------------------------------------------------------------------
  // BOTTOM PANEL: (B) Address Event Representation (AER) Mechanism
  // ------------------------------------------------------------------
  group({
    translate((0, -0.5)) 
    content((2.5, 10.5), text(weight: "bold", size: 10pt, "(B) AER Mechanism"))
    
    // --- 1. Operation: Spikes to Digital Packet ---
    circle((1.0, 8.5), radius: 0.4, fill: gray.darken(20%), stroke: black)
    content((1.0, 8.5), text(size: 8pt, "Source\nCore"))
    circle((0.7, 8.2), radius: 0.1, fill: black, stroke: none) 
    
    content((1.8, 8.5), text(size: 8pt, "generate\nSpike"))
    line((1.3, 8.5), (2.0, 8.5), mark: (end: ">"), stroke: (paint: gray, thickness: 0.8pt))
    
    // AER Encoder block logic
    rect((2.5, 8.0), (4.0, 9.0), radius: 0.2, fill: white, stroke: black)
    content((3.25, 8.5), text(size: 8pt, "AER\nEncoder"))
    
    // --- 2. Communication: Asynchronous Bus ---
    // FIXED: Manually defined points instead of using floats in range()
    let p_a = ((4.5, 8.5), (5.3, 8.5), (6.1, 8.5), (6.9, 8.5))
    line(..p_a, stroke: (paint: gray, thickness: 1pt, dash: "dashed"))
    line((4.5, 8.5), (7.0, 8.5), mark: (end: ">"), stroke: (paint: gray, thickness: 1pt))
    
    // Conceptual AER packets
    content((5.3, 8.8), text(size: 7pt, "$(\\text{Addr}_1, t_1)$"))
    content((6.1, 8.8), text(size: 7pt, "$(\\text{Addr}_2, t_2)$"))
    content((6.9, 8.8), text(size: 7pt, "$(\\text{Addr}_3, t_3)$"))
    
    content((5.5, 9.2), text(weight: "bold", fill: red.darken(10%), size: 9pt, "Asynchronous AER Bus"))
    content((5.5, 8.0), text(size: 7pt, fill: gray.darken(30%), "Sparse, Event-Driven Events"))

    // --- 3. Operation: Digital Packet to Spikes ---
    // AER Decoder block logic
    rect((7.5, 8.0), (9.0, 9.0), radius: 0.2, fill: white, stroke: black)
    content((8.25, 8.5), text(size: 8pt, "AER\nDecoder"))
    
    content((9.2, 8.5), text(size: 8pt, "reconstruct\nSpike"))
    line((8.5, 8.5), (9.2, 8.5), mark: (end: ">"), stroke: (paint: gray, thickness: 0.8pt))
    
    // Destination Core logic
    circle((10.0, 8.5), radius: 0.4, fill: white, stroke: black)
    content((10.0, 8.5), text(size: 8pt, "Dest.\nCore"))
    circle((10.3, 8.2), radius: 0.1, fill: black, stroke: none) 
  })

  // ------------------------------------------------------------------
  // SIDE LEGEND: Contrast with Von Neumann Bottleneck (Fig 29)
  // ------------------------------------------------------------------
  group({
    translate((7.5, 1)) 
    content((0, 10.5), text(weight: "bold", size: 9pt, "AER Efficiency (vs Fig 29)"))
    
    rect((-0.5, 9.5), (2.5, 10.2), fill: gray.darken(20%), stroke: (paint: black, thickness: 1.2pt))
    content((1.0, 9.85), text(size: 7pt, weight: "bold", "Von Neumann Bottleneck"))
    
    line((0.0, 9.85), (2.0, 9.85), mark: (end: ">"), stroke: (paint: red.darken(10%), thickness: 1.5pt))
    content((1.0, 9.2), text(size: 7pt, fill: red.darken(10%), "Continuous Data Flow"))
    line((1.0, 8.8), (1.0, 8.3), mark: (end: ">"), stroke: (thickness: 0.8pt, paint: gray))
    
    content((1.0, 8.0), text(size: 7pt, fill: gray.darken(30%), "High Energy transport"))
    line((1.0, 7.6), (1.0, 7.1), mark: (end: ">"), stroke: (thickness: 0.8pt, paint: gray))
    
    line((0.0, 6.7), (2.0, 6.7), mark: (end: ">"), stroke: (paint: blue.darken(20%), thickness: 1.5pt, dash: "dotted"))
    content((1.0, 6.1), text(size: 7pt, fill: blue.darken(20%), "Sparse AER Events"))
    line((1.0, 5.7), (1.0, 5.2), mark: (end: ">"), stroke: (thickness: 0.8pt, paint: gray))
    
    content((1.0, 4.9), text(size: 7pt, fill: gray.darken(30%), "Low Energy transport"))
  })
})
