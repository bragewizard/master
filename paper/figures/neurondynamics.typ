#import "@preview/cetz:0.4.2"
#import "@preview/cetz-plot:0.1.3": plot, chart

// --- AdEx Parameters ---
#let C = 281.0
#let gL = 30.0
#let EL = -70.6
#let VT = -50.4
#let DeltaT = 2.0
#let a = 4.0
#let tau_w = 144.0
#let b = 80.5
#let I = 800.0 // High current for spiking

// Nullclines
#let v-null(v) = -gL * (v - EL) + gL * DeltaT * calc.exp((v - VT) / DeltaT) + I
#let w-null(v) = a * (v - EL)

// Simulation (Euler)
#let solve-adex(steps, dt, v0, w0) = {
  let pts = ((v0, w0),)
  let v = v0
  let w = w0
  let resets = () 

  for i in range(steps) {
    let exp_term = gL * DeltaT * calc.exp((v - VT) / DeltaT)
    let dv = (-gL * (v - EL) + exp_term - w + I) / C
    let dw = (a * (v - EL) - w) / tau_w
    
    v = v + dv * dt
    w = w + dw * dt
    
    if v >= -30 { // Peak
      let v_old = v
      let w_old = w
      v = -70.6 // Reset
      w = w + b
      resets.push(((v_old, w_old), (v, w)))
      pts.push((v_old, w_old))
      pts.push((v, w))
    } else {
      pts.push((v, w))
    }
  }
  (pts, resets)
}

#let (traj, resets) = solve-adex(3000, 0.05, -70.6, 0.0)

// Data for Nullcline Curves
#let v_range = range(-80, -28).map(x => x * 1.0)
#let v_nc_pts = v_range.map(v => (v, v-null(v)))
#let w_nc_pts = v_range.map(v => (v, w-null(v)))

// Close the shape for filling (V-nullcline area)
#let fill_shape = v_nc_pts + ((-30, -100), (-80, -100))

#cetz.canvas(length: .8cm, {
  import cetz.draw: *

  plot.plot(
    size: (10, 8), 
    x-tick-step: 10, 
    y-tick-step: 50, 
    x-min: -85, x-max: -25,
    y-min: -50, y-max: 500,
    axis-style: "school-book", 
    name: "phase", 
    {    
    plot.add(v_nc_pts, style: (stroke: (paint: gray, thickness: 2pt)), label: "V-Nullcline")
    plot.add(w_nc_pts, style: (stroke: (paint: gray, thickness: 2pt)), label: "w-Nullcline")
  })
  
  content("phase.north", text(weight: "bold", "AdEx Phase Portrait (Colored Regions)"), anchor: "south", padding: .2)
})
