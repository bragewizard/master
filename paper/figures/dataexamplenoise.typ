#import "@preview/cetz:0.4.2"

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  // Global aesthetics matching your minimalistic style
  set-style(
    stroke: (thickness: 1pt, cap: "round", join: "round"),
    mark: (fill: black, scale: 0.8)
  )

  // 1. A simple pseudo-random hash function to replace calc.random()
  // This guarantees the noise looks random but renders identically every time you compile.
  let hash(x, y) = {
    let v = calc.sin((x * 12.9898 + y * 78.233) * 1rad) * 43758.5453
    return calc.rem(calc.abs(v), 1.0)
  }

  // Title for the entire figure
  content((3.2, 7.5), text(weight: "bold", size: 10pt, "64x64 Dot Matrix with Grayscale Noise"))

  // Draw a large black box for the matrix bounds
  let matrix_side = 6.4
  rect((0, 0), (matrix_side, matrix_side), fill: black, stroke: (paint: gray, thickness: 1.5pt))
  
  let dot_radius = 0.035
  let dot_spacing = 0.1

  // 2. Procedural Matrix Generation with Noise
  for row in range(64) {
    for col in range(64) {
      let x = col * dot_spacing
      let y = (63 - row) * dot_spacing 

      // Set base intensity (0.1 for dark background, 0.9 for bright shapes)
      let base_intensity = 0.1 
      
      // Shape 1: Small Triangle 
      if (row >= 24 and row <= 36) {
        let half_width = (row - 24) * 0.5
        if (calc.abs(col - 16) <= half_width) {
          base_intensity = 0.9 
        }
      }
      
      // Shape 2: Small Square 
      if (col >= 43 and col <= 53 and row >= 25 and row <= 35) {
        base_intensity = 0.9 
      }

      // Apply the pseudo-random noise
      let noise_amp = 0.4 // Adjust this to make it more/less noisy
      let random_val = hash(row, col)
      let noisy_intensity = base_intensity + (random_val - 0.5) * noise_amp
      
      // Clamp to ensure we don't go outside 0.0-1.0 limits
      let final_intensity = calc.max(0.0, calc.min(1.0, noisy_intensity))

      // Draw dot using Typst's native luma() for grayscale
      circle((x, y), radius: dot_radius, fill: luma(final_intensity * 100%), stroke: none)
    }
  }

  // 3. Add callouts and labels
  group({
    translate((7.2, 3.5)) 
    rect((0, 0), (3, 2), fill: black, stroke: (paint: gray, thickness: 1.5pt))
    content((1.5, 2.3), text(weight: "bold", size: 9pt, "Magnified Pixel View"))
    
    let mag_r = 0.2
    
    // Hardcode an example of a dark noisy pixel
    circle((0.7, 1.2), radius: mag_r, fill: luma(20%), stroke: none)
    content((0.7, 0.5), text(size: 8pt, fill: gray.lighten(50%), "Noisy Off Pixel\n(State ~0)"))
    
    // Hardcode an example of a bright noisy pixel
    circle((2.3, 1.2), radius: mag_r, fill: luma(80%), stroke: none)
    content((2.3, 0.5), text(size: 8pt, fill: white, "Noisy On Pixel\n(State ~1)"))
    
    // Callout lines 
    let matrix_pt = (53 * 0.1, (63 - 30) * 0.1) 
    line(matrix_pt, (7.2, 1.5), mark: (start: ">"), stroke: (paint: gray, thickness: 1pt, dash: "dashed"))
    line(matrix_pt, (7.2, 0.8), mark: (start: ">"), stroke: (paint: gray, thickness: 1pt, dash: "dashed"))
  })

  // Shape Labels
  content((16 * 0.1, -0.6), text(weight: "bold", size: 9pt, "(A) Triangle"))
  line((16 * 0.1, -0.3), (16 * 0.1, 0), mark: (end: ">"), stroke: (thickness: 0.8pt))
  
  content((48 * 0.1, -0.6), text(weight: "bold", size: 9pt, "(B) Square"))
  line((48 * 0.1, -0.3), (48 * 0.1, 0), mark: (end: ">"), stroke: (thickness: 0.8pt))

  // Matrix Dimension Markers
  line((-0.3, matrix_side), (-0.3, 0), mark: (start: "|", end: "|"), stroke: (paint: gray, thickness: 1pt))
  content((-0.8, matrix_side / 2), text(size: 8pt, fill: gray, "64 rows"))
  
  line((0, matrix_side + 0.3), (matrix_side, matrix_side + 0.3), mark: (start: "|", end: "|"), stroke: (paint: gray, thickness: 1pt))
  content((matrix_side / 2, matrix_side + 0.6), text(size: 8pt, fill: gray, "64 columns"))
})
