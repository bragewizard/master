#import "@preview/cetz:0.4.2"

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  // Global aesthetics matching your minimalistic style
  set-style(
    stroke: (thickness: 1pt, cap: "round", join: "round"),
    mark: (fill: black, scale: 0.8)
  )

  // Title for the entire figure
  content((3.2, 7.5), text(weight: "bold", size: 10pt, "64x64 Dot Matrix: Triangle and Square"))

  // 1. Draw a large black box for the entire matrix bounds
  // We'll use 0.1 spacing, making a 6.4 x 6.4 cm matrix
  let matrix_side = 6.4
  rect((0, 0), (matrix_side, matrix_side), fill: black, stroke: (paint: gray, thickness: 1.5pt))
  
  // Define dot properties
  let dot_radius = 0.035
  let dot_spacing = 0.1
  let col_on = white
  let col_off = gray.darken(70%) // Dark gray so the grid structure is still subtly visible

  // 2. Procedural Matrix Generation: Loop through 64x64 matrix
  for row in range(64) {
    for col in range(64) {
      let x = col * dot_spacing
      // Inverse y to make Row 0 at the top
      let y = (63 - row) * dot_spacing 

      let is_on = false
      
      // Shape 1: Small Triangle (Centered around col 16)
      // Top tip at row 24, base at row 36.
      // Width expands by 0.5 pixels per row moving downwards.
      if (row >= 24 and row <= 36) {
        let half_width = (row - 24) * 0.5
        if (calc.abs(col - 16) <= half_width) {
          is_on = true
        }
      }
      
      // Shape 2: Small Square (Centered around col 48)
      // 10x10 pixels (reduced from 16x16)
      if (col >= 43 and col <= 53 and row >= 25 and row <= 35) {
        is_on = true
      }

      // Draw the dot
      if (is_on) {
        circle((x, y), radius: dot_radius, fill: col_on, stroke: none)
      } else {
        circle((x, y), radius: dot_radius, fill: col_off, stroke: none)
      }
    }
  }

  // 3. Add callouts and labels for clarity
  
  // Magnified view of pixel definition
  group({
    translate((7.2, 3.5)) // Shift callout view to the right
    rect((0, 0), (3, 2), fill: black, stroke: (paint: gray, thickness: 1.5pt))
    content((1.5, 2.3), text(weight: "bold", size: 9pt, "Magnified Pixel View"))
    
    // Draw magnified pixels and labels
    let mag_r = 0.2
    circle((0.7, 1.2), radius: mag_r, fill: col_off, stroke: none)
    content((0.7, 0.5), text(size: 8pt, fill: col_off, "Off Pixel\n(State 0)"))
    
    circle((2.3, 1.2), radius: mag_r, fill: col_on, stroke: none)
    content((2.3, 0.5), text(size: 8pt, fill: col_on, "On Pixel\n(State 1)"))
    
    // Callout lines mapping the magnified view to the matrix
    let matrix_pt = (53 * 0.1, (63 - 30) * 0.1) // Edge of the square
    line(matrix_pt, (7.2, 1.5), mark: (start: ">"), stroke: (paint: gray, thickness: 1pt, dash: "dashed"))
    line(matrix_pt, (7.2, 0.8), mark: (start: ">"), stroke: (paint: gray, thickness: 1pt, dash: "dashed"))
  })

  // 4. Shape Labels
  // (A) Triangle
  content((16 * 0.1, -0.6), text(weight: "bold", size: 9pt, "(A) Triangle"))
  line((16 * 0.1, -0.3), (16 * 0.1, 0), mark: (end: ">"), stroke: (thickness: 0.8pt))
  
  // (B) Square
  content((48 * 0.1, -0.6), text(weight: "bold", size: 9pt, "(B) Square"))
  line((48 * 0.1, -0.3), (48 * 0.1, 0), mark: (end: ">"), stroke: (thickness: 0.8pt))

  // 5. Matrix Dimension Markers
  line((-0.3, matrix_side), (-0.3, 0), mark: (start: "|", end: "|"), stroke: (paint: gray, thickness: 1pt))
  content((-0.8, matrix_side / 2), text(size: 8pt, fill: gray, "64 rows"))
  
  line((0, matrix_side + 0.3), (matrix_side, matrix_side + 0.3), mark: (start: "|", end: "|"), stroke: (paint: gray, thickness: 1pt))
  content((matrix_side / 2, matrix_side + 0.6), text(size: 8pt, fill: gray, "64 columns"))
})
