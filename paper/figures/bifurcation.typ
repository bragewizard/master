#import "@preview/cetz:0.4.2"

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  // Global aesthetics matching your previous style
  set-style(
    stroke: (thickness: 1.5pt, cap: "round", join: "round"),
    mark: (fill: black, scale: 0.8)
  )

  // Helper function to draw consistent axes for all subplots
  let draw-axes(x-label, y-label) = {
    line((-0.2, 0), (4.5, 0), mark: (end: ">", fill: black), stroke: (thickness: 1pt, paint: black))
    line((0, -0.2), (0, 3.2), mark: (end: ">", fill: black), stroke: (thickness: 1pt, paint: black))
    content((4.7, 0), x-label)
    content((0, 3.4), y-label)
  }

  // ------------------------------------------------------------------
  // LEFT PANEL: CLASS I (Integrator / SNIC Bifurcation)
  // ------------------------------------------------------------------
  group({
    // Top Left: f-I Curve
    group({
      content((2.25, 4.0), text(weight: "bold", size: 10pt, "Class I (Integrator)"))
      draw-axes($I$, $f$)
      
      let trace-style = (paint: blue.darken(20%), thickness: 1.5pt)
      line((0,0), (1.5,0), stroke: trace-style) // Zero frequency before threshold
      
      // Calculate a smooth square-root curve for SNIC onset
      let sqrt-pts = range(0, 26).map(i => {
        let x = i * 0.1
        (1.5 + x, 1.8 * calc.sqrt(x))
      })
      line(..sqrt-pts, stroke: trace-style)
      
      circle((1.5, 0), radius: 0.08, fill: blue.darken(20%), stroke: none)
      content((1.5, -0.5), $I_c$)
      
      content((2.8, 1.2), text(size: 8pt, "Smooth onset"))
      line((2.8, 1.0), (1.8, 0.5), mark: (end: ">"), stroke: (thickness: 0.8pt))
    })

    // Bottom Left: Voltage Trace (Sub-threshold dynamics)
    group({
      translate((0, -4.5))
      draw-axes($t$, $V$)
      
      line((-0.2, 2.0), (4.2, 2.0), stroke: (dash: "dashed", paint: gray, thickness: 1pt))
      content((-0.6, 2.0), $theta.alt$)
      
      // Calculate a slow, smooth integration curve
      let v-pts = range(0, 41).map(i => {
        let t = i * 0.1
        (t, 2.0 - 2.0 * calc.exp(-t * 0.8))
      })
      line(..v-pts, stroke: (paint: purple.darken(10%), thickness: 1.5pt))
      
      content((2.5, 0.8), text(size: 8pt, "Slow integration\n(No oscillations)"))
    })
  })

  // ------------------------------------------------------------------
  // RIGHT PANEL: CLASS II (Resonator / Hopf Bifurcation)
  // ------------------------------------------------------------------
  group({
    translate((7.5, 0)) // Offset right panel
    
    // Top Right: f-I Curve
    group({
      content((2.25, 4.0), text(weight: "bold", size: 10pt, "Class II (Resonator)"))
      draw-axes($I$, $f$)
      
      let trace-style = (paint: blue.darken(20%), thickness: 1.5pt)
      line((0,0), (1.5,0), stroke: trace-style)
      
      // The discontinuous jump
      line((1.5,0), (1.5,1.2), stroke: (dash: "dashed", paint: blue.darken(20%), thickness: 1pt))
      
      let curve-pts = range(0, 26).map(i => {
        let x = i * 0.1
        (1.5 + x, 1.2 + 0.8 * calc.sqrt(x)) // f stays bounded away from 0
      })
      line(..curve-pts, stroke: trace-style)
      
      circle((1.5, 0), radius: 0.08, fill: white, stroke: blue.darken(20%)) // Open circle
      circle((1.5, 1.2), radius: 0.08, fill: blue.darken(20%), stroke: none) // Solid circle
      content((1.5, -0.5), $I_c$)
      
      content((0.7, 0.6), text(size: 8pt, "Jump"))
    })

    // Bottom Right: Voltage Trace (Sub-threshold dynamics)
    group({
      translate((0, -4.5))
      draw-axes($t$, $V$)
      
      line((-0.2, 2.0), (4.2, 2.0), stroke: (dash: "dashed", paint: gray, thickness: 1pt))
      content((-0.6, 2.0), $theta.alt$)
      
      // Calculate a mathematically accurate damped/growing oscillator curve
      let v-pts = range(0, 41).map(i => {
        let t = i * 0.1
        let env = 0.04 * calc.exp(t * 0.75) // Exponential growth envelope
        (t, 1.0 + env * calc.sin(t * 400deg))
      })
      line(..v-pts, stroke: (paint: purple.darken(10%), thickness: 1.5pt))
      
      content((2.8, 0.5), text(size: 8pt, "Sub-threshold\noscillations"))
      line((2.8, 0.8), (3.3, 1.4), mark: (end: ">"), stroke: (thickness: 0.8pt))
    })
  })
})
