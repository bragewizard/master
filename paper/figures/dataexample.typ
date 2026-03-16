#import "@preview/cetz:0.4.2"

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  set-style(
    stroke: (thickness: 1pt, cap: "round", join: "round"),
    mark: (fill: black, scale: 0.8)
  )

  let matrix_side = 8.0
  rect((-0.25, -0.25), (matrix_side, matrix_side), fill: black, radius:4pt)
  
  let dot_radius = 0.1
  let dot_spacing = 0.25
  let col_on = white
  let col_off = gray.darken(70%)
  let size = 32
  for row in range(size) {
    for col in range(size) {
      let x = col * dot_spacing
      let y = (size - 1 - row) * dot_spacing 

      let is_on = false
      
      if (row >= 14 and row <= 20) {
        let half_width = (row - 16) * 0.5
        if (calc.abs(col - 16) <= half_width) {
          is_on = true
        }
      }
      
      if (col >= 10 and col <= 14 and row >= 4 and row <= 8) {
        is_on = true
      }

      if (is_on) {
        circle((x, y), radius: dot_radius, fill: col_on, stroke: none)
      } else {
        circle((x, y), radius: dot_radius, fill: col_off, stroke: none)
      }
    }
  }
})
