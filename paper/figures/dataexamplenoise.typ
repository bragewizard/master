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

  let hash(x, y) = {
    let v = calc.sin((x * 12.9898 + y * 78.233) * 1rad) * 43758.5453
    return calc.rem(calc.abs(v), 1.0)
  }

  for row in range(size) {
    for col in range(size) {
      let x = col * dot_spacing
      let y = (size - 1 - row) * dot_spacing 

      let base_intensity = 0.1 

      if (row >= 14 and row <= 20) {
        let half_width = (row - 16) * 0.5
        if (calc.abs(col - 16) <= half_width) {
          base_intensity = 0.9 
        }
      }
      
      if (col >= 10 and col <= 14 and row >= 4 and row <= 8) {
        base_intensity = 0.9 
      }

      let noise_amp = 0.4 // Adjust this to make it more/less noisy
      let random_val = hash(row, col)
      let noisy_intensity = base_intensity + (random_val - 0.5) * noise_amp
      let final_intensity = calc.max(0.0, calc.min(1.0, noisy_intensity))
      circle((x, y), radius: dot_radius, fill: luma(final_intensity * 100%), stroke: none)
    }
  }
})
