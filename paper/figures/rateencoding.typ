#import "@preview/cetz:0.4.2"  
#import "@preview/cetz-plot:0.1.3": plot

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  // x = time (ms), y = row/neuron index
  let spike-data = (
    // Row 4: High Value (9) -> High firing rate (10 spikes)
    (0.5, 4), (1.5, 4), (2.5, 4), (3.5, 4), (4.5, 4), 
    (5.5, 4), (6.5, 4), (7.5, 4), (8.5, 4), (9.5, 4),
    
    // Row 3: Medium Value (6) -> Medium firing rate (6 spikes)
    (1.2, 3), (2.8, 3), (4.4, 3), (6.0, 3), (7.6, 3), (9.2, 3),
    
    // Row 2: Low Value (3) -> Low firing rate (3 spikes)
    (2.0, 2), (5.0, 2), (8.0, 2),
    
    // Row 1: Very Low Value (1) -> Minimal firing rate (1 spike)
    (4.5, 1)
  )

  // Global styles for the plot axes
  set-style(axes: (
    stroke: (thickness: 1pt, paint: black),
    x: (mark: (end: ">", fill: black, scale: 0.8)),
    y: (mark: (end: ">", fill: black, scale: 0.8)),
    tick: (stroke: black + 1pt),
  ))

  plot.plot(
    size: (8, 4), 
    x-tick-step: 2, 
    x-min: 0, 
    x-max: 10,
    y-min: 0.5, 
    y-max: 4.5,
    x-label: "Time (ms)",
    
    // Custom Y-ticks to show the Stimulus Value on the left
    y-ticks: (
      (1, text(weight: "bold", [Value: 1])), 
      (2, text(weight: "bold", [Value: 3])), 
      (3, text(weight: "bold", [Value: 6])), 
      (4, text(weight: "bold", [Value: 9]))
    ),
    
    axis-style: "left", 
    name: "raster", 
    {    
      plot.add(
        spike-data,
        mark: "|", // Draws vertical lines for spikes
        mark-style: (stroke: 1.5pt + blue.darken(20%)),
        mark-size: 0.4, // Scales the height of the spike mark
        style: (stroke: none) // Hides the connecting lines between points
      )
  })
})
