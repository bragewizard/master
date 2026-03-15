#import "@preview/cetz:0.4.2"

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  // Global aesthetics matching your minimalist academic style
  set-style(
    stroke: (thickness: 1.2pt, cap: "round", join: "round"),
    mark: (fill: black, scale: 0.8)
  )

  // 1. Title and Math Formula
  content((4.25, 5.5), text(weight: "bold", size: 10pt, "Leaky Membrane Potential Dynamics"))
  content((4.25, 4.8), text(size: 9pt, fill: gray.darken(30%), "$V_m(t) = V_\"rest\" + \\sum w \\cdot \\exp\\left(-\\frac{t - t_i}{\\tau}\\right)$"))

  // ------------------------------------------------------------------
  // TOP PANEL: Membrane Potential V_m(t)
  // ------------------------------------------------------------------
  group({
    // Axes
    line((0, 0), (8.5, 0), mark: (end: ">"), stroke: (thickness: 1pt))
    line((0, 0), (0, 4.0), mark: (end: ">"), stroke: (thickness: 1pt))
    content((-0.8, 2.0), text(size: 9pt, "Potential\n$V_m(t)$"), angle: 90deg)
    
    let v_rest = 0.5
    let tau = 1.0
    let amp = 1.2
    let spikes = (1.0, 3.0, 3.8, 4.4, 7.0)

    // Reference Lines
    line((0, v_rest), (8.5, v_rest), stroke: (paint: gray, dash: "dashed", thickness: 1pt))
    content((-0.4, v_rest), text(size: 8pt, fill: gray.darken(30%), "$V_\"rest\"$"))
    
    let v_th = 3.2
    line((0, v_th), (8.5, v_th), stroke: (paint: red.lighten(30%), dash: "dotted", thickness: 1.2pt))
    content((-0.4, v_th), text(size: 8pt, fill: red.darken(10%), "$V_\"th\"$"))

    // Procedural math curve for exponential decay and summation
    let pts = ()
    let dt = 0.02
    for i in range(426) { // 8.5 / 0.02
        let t = i * dt
        let v = v_rest
        for ts in spikes {
            if t >= ts {
                // Exponential decay influence of each past spike
                v += amp * calc.exp(-(t - ts) / tau)
            }
        }
        pts.push((t, v))
    }
    
    // Draw the continuous membrane trace
    line(..pts, stroke: (paint: purple.darken(10%), thickness: 2pt, join: "round"))

    // Callouts
    // 1. Single Exponential Decay
    content((2.3, 1.8), text(size: 8pt, fill: purple.darken(10%), "Exponential\nDecay (Leak)"))
    line((1.9, 1.5), (1.5, 1.2), mark: (end: ">"), stroke: (paint: purple.darken(10%), thickness: 0.8pt))
    
    // 2. Temporal Summation bracket
    line((3.0, 3.5), (4.4, 3.5), mark: (start: "|", end: "|"), stroke: (paint: gray, thickness: 1pt))
    content((3.7, 3.8), text(size: 8pt, weight: "bold", "Temporal Summation"))
    
    // 3. Sub-threshold peak
    circle((4.4, 2.85), radius: 0.1, fill: none, stroke: (paint: red.darken(10%), thickness: 1pt))
    content((5.8, 3.0), text(size: 8pt, fill: red.darken(10%), "Fails to reach\nThreshold ($V_\"th\"$)"))
    line((5.0, 2.9), (4.6, 2.85), mark: (end: ">"), stroke: (paint: red.darken(10%), thickness: 0.8pt))
  })

  // ------------------------------------------------------------------
  // BOTTOM PANEL: Incoming Spikes
  // ------------------------------------------------------------------
  group({
    translate((0, -2.0))
    
    // Axes
    line((0, 0), (8.5, 0), mark: (end: ">"), stroke: (thickness: 1pt))
    content((8.2, -0.4), text(size: 9pt, "Time $t$"))
    content((-0.8, 0.5), text(size: 9pt, "Incoming\nSpikes"), angle: 90deg)

    let spikes = (1.0, 3.0, 3.8, 4.4, 7.0)
    
    // Draw discrete spike events
    for (i, ts) in spikes.enumerate() {
        line((ts, 0), (ts, 1.0), stroke: (paint: blue.darken(20%), thickness: 1.5pt))
        circle((ts, 1.0), radius: 0.08, fill: blue.darken(20%), stroke: none)
        
        // Link bottom trace to top trace
        line((ts, 1.2), (ts, 2.0), stroke: (paint: gray.lighten(30%), thickness: 1pt, dash: "dotted"))
    }
    
    content((1.0, -0.4), text(size: 7pt, "$t_1$"))
    content((3.0, -0.4), text(size: 7pt, "$t_2$"))
    content((3.8, -0.4), text(size: 7pt, "$t_3$"))
    content((4.4, -0.4), text(size: 7pt, "$t_4$"))
    content((7.0, -0.4), text(size: 7pt, "$t_5$"))
  })
})
