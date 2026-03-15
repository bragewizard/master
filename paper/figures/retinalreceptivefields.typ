#import "@preview/cetz:0.4.2"

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  // Global aesthetics matching your minimalist academic style
  set-style(
    stroke: (thickness: 1.2pt, cap: "round", join: "round"),
    mark: (fill: black, scale: 0.8)
  )

  // Colors for the spatial stimulus icons
  let light_col = yellow.lighten(30%)
  let dark_col = gray.darken(60%)

  // Helper function to procedurally generate the physiological spike trains
  let draw_trace(x_offset, y_base, stimulus_type, condition) = {
    let center_col = if stimulus_type == "center" or stimulus_type == "diffuse" { light_col } else { dark_col }
    let surround_col = if stimulus_type == "surround" or stimulus_type == "diffuse" { light_col } else { dark_col }

    // 1. Draw mini Stimulus Icon (Spatial layout of the light)
    circle((x_offset + 0.5, y_base + 0.4), radius: 0.6, fill: surround_col, stroke: black)
    circle((x_offset + 0.5, y_base + 0.4), radius: 0.25, fill: center_col, stroke: black)
    
    // Label the stimulus type
    let stim_label = if stimulus_type == "center" { "Center Spot" } 
                     else if stimulus_type == "surround" { "Surround Annulus" } 
                     else { "Diffuse Illumination" }
    content((x_offset + 0.5, y_base - 0.5), text(size: 7pt, stim_label))

    // 2. Draw Stimulus Time Bar (Yellow rectangle above trace)
    let x_start = x_offset + 2.0
    let light_start = 0.5
    let light_end = 3.0
    
    rect((x_start + light_start, y_base + 1.3), (x_start + light_end, y_base + 1.5), fill: light_col, stroke: none)
    content((x_start + 1.75, y_base + 1.7), text(size: 7pt, "Light ON"))

    // 3. Build Voltage Trace points mathematically
    let pts = ((x_start, y_base), (x_start + light_start - 0.1, y_base))

    if condition == "excite" {
        // Sustained depolarization + high frequency spikes
        pts.push((x_start + light_start, y_base + 0.2)) 
        for i in range(10) {
            let sx = x_start + light_start + 0.15 + i*0.23
            pts.push((sx - 0.05, y_base + 0.2))
            pts.push((sx, y_base + 1.1)) // Spike Peak
            pts.push((sx + 0.05, y_base - 0.1)) // Undershoot
        }
        pts.push((x_start + light_end, y_base + 0.2))
        pts.push((x_start + light_end + 0.1, y_base))
        
    } else if condition == "inhibit_rebound" {
        // Hyperpolarization during light, rebound spikes at offset
        pts.push((x_start + light_start, y_base))
        pts.push((x_start + light_start + 0.2, y_base - 0.4))
        pts.push((x_start + light_end - 0.2, y_base - 0.4))
        pts.push((x_start + light_end, y_base))
        
        // Rebound spikes when inhibition is released
        for i in range(3) {
            let sx = x_start + light_end + 0.1 + i*0.25
            pts.push((sx - 0.05, y_base))
            pts.push((sx, y_base + 1.1))
            pts.push((sx + 0.05, y_base - 0.1))
        }
        pts.push((x_start + light_end + 0.8, y_base))
        
    } else if condition == "weak" {
        // Cancelled out signals (Spontaneous background firing only)
        pts.push((x_start + light_start, y_base))
        for i in (0.2, 1.5, 2.7, 4.0) {
            let sx = x_start + i
            pts.push((sx - 0.05, y_base))
            pts.push((sx, y_base + 1.1))
            pts.push((sx + 0.05, y_base - 0.1))
        }
    }
    
    // Finish the trace
    pts.push((x_start + 4.5, y_base))
    
    // Draw the generated trace and baseline reference
    line(..pts, stroke: (paint: blue.darken(20%), thickness: 1.2pt, join: "round"))
    line((x_start, y_base), (x_start + 4.5, y_base), stroke: (paint: gray, thickness: 0.8pt, dash: "dashed"))
  }

  // ------------------------------------------------------------------
  // LEFT PANEL: (A) On-Center
  // ------------------------------------------------------------------
  group({
    content((3.5, 9.5), text(weight: "bold", size: 10pt, "(A) On-Center Cell"))
    
    // Receptive Field Diagram
    circle((3.5, 8.0), radius: 1.2, fill: gray.lighten(60%), stroke: black) // Surround (-)
    circle((3.5, 8.0), radius: 0.5, fill: white, stroke: black) // Center (+)
    
    content((3.5, 8.0), text(weight: "bold", size: 14pt, "+"))
    for (px, py) in ((3.5, 9.0), (3.5, 7.0), (2.5, 8.0), (4.5, 8.0)) {
        content((px, py), text(weight: "bold", size: 14pt, "-"))
    }

    // Firing Logic Traces
    draw_trace(0, 5.0, "center", "excite")
    draw_trace(0, 2.5, "surround", "inhibit_rebound")
    draw_trace(0, 0.0, "diffuse", "weak")

    // Timeline Axis
    line((2.0, -0.8), (6.5, -0.8), mark: (end: ">"), stroke: (thickness: 1pt))
    content((6.8, -0.8), text(size: 9pt, "Time"))
  })

  // ------------------------------------------------------------------
  // RIGHT PANEL: (B) Off-Center
  // ------------------------------------------------------------------
  group({
    let x_off = 8.5
    content((x_off + 3.5, 9.5), text(weight: "bold", size: 10pt, "(B) Off-Center Cell"))
    
    // Receptive Field Diagram
    circle((x_off + 3.5, 8.0), radius: 1.2, fill: white, stroke: black) // Surround (+)
    circle((x_off + 3.5, 8.0), radius: 0.5, fill: gray.lighten(60%), stroke: black) // Center (-)
    
    content((x_off + 3.5, 8.0), text(weight: "bold", size: 16pt, "-"))
    for (px, py) in ((x_off + 3.5, 9.0), (x_off + 3.5, 7.0), (x_off + 2.5, 8.0), (x_off + 4.5, 8.0)) {
        content((px, py), text(weight: "bold", size: 12pt, "+"))
    }

    // Firing Logic Traces
    draw_trace(x_off, 5.0, "center", "inhibit_rebound")
    draw_trace(x_off, 2.5, "surround", "excite")
    draw_trace(x_off, 0.0, "diffuse", "weak")

    // Timeline Axis
    line((x_off + 2.0, -0.8), (x_off + 6.5, -0.8), mark: (end: ">"), stroke: (thickness: 1pt))
    content((x_off + 6.8, -0.8), text(size: 9pt, "Time"))
  })

  // ------------------------------------------------------------------
  // BOTTOM ANNOTATION: The Edge Enhancement Logic
  // ------------------------------------------------------------------
  line((0, -1.8), (15, -1.8), stroke: (paint: gray, dash: "dotted", thickness: 1pt))
  content((7.5, -2.4), text(weight: "bold", fill: red.darken(20%), size: 9pt, "Lateral Inhibition acts as an Edge-Enhancement Filter:"))
  content((7.5, -3.0), text(size: 9pt, "Under diffuse illumination (bottom traces), excitatory and inhibitory regions cancel each other out, resulting in weak baseline firing. \nThe cell only fires strongly when a contrast edge selectively illuminates the excitatory region without triggering the inhibitory surround."))
})
