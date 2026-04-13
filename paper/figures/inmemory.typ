#import "@preview/cetz:0.4.2"

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  set-style(
    stroke: (thickness: 1.6pt, cap: "butt", join: "miter"),
    mark: (fill: black, scale: 1.0)
  )

  content((4, 7.5), text(weight: "bold", "In-Memory Computing (Crossbar Array)"))

  let n_lines = 4
  let spacing = 1.2
  let offset_x = 2.0
  let offset_y = 2.0

  for i in range(n_lines) {
    let y = offset_y + i * spacing
    line((0.5, y), (offset_x + (n_lines - 0.5) * spacing, y), stroke: (paint: blue.darken(20%)))

    // Input labels
    content((0, y), text([$V_#str(n_lines - i)$]))
    if i == n_lines - 1 {
      content((-1.2, y - 1.7), text(weight: "bold", fill: blue.darken(20%), [Input\ Voltages]))
    }
  }

  // Draw Vertical Bitlines (Outputs)
  for j in range(n_lines) {
    let x = offset_x + j * spacing
    line((x, offset_y - 0.5), (x, offset_y + (n_lines - 0.5) * spacing), stroke: (paint: green.darken(10%)))
    mark((x, offset_y - 0.5), (x, offset_y - 0.8), mark: (end: ">", fill: green.darken(10%)), stroke: none)

    // Output labels
    content((x, offset_y - 1.2), text([$I_#str(j + 1)$]))
    if j == 0 {
      content((x + 1.9, offset_y - 2.3), text(weight: "bold", fill: red.darken(10%), [Output\ Currents]))
    }
  }

  for i in range(n_lines) {
    for j in range(n_lines) {
      let x = offset_x + j * spacing
      let y = offset_y + i * spacing

      let r = 0.25
      line((x - r, y), (x, y + r), (x + r, y), (x, y - r), (x - r, y), fill: green.lighten(60%))
    }
  }

  let h_x = offset_x + 2 * spacing
  let h_y = offset_y + 1 * spacing

  // Callout circle
  circle((h_x, h_y), radius: 0.45, stroke: (paint: gray, dash: "dashed" ))

  // Callout line
  line((h_x + 0.43, h_y + 0.20), (h_x + 3.2, h_y + 1.8), stroke: (paint: gray ))

  // Physics Math Explanation Box
  group({
    translate((h_x + 2.8, h_y + 0.8))
    rect((0, -0.4), (6.3, 3.0), radius:2pt)

    content((2.50, 2.4), text(weight: "bold", "Physics as Computation"))

    // Multiplication
    content((0.2, 1.7), text( "1. Multiplication (Ohm's Law)"), anchor: "west")
    content((0.5, 1.2), text( fill: green.darken(30%), [$I = V times G_(i j)$]), anchor: "west")

    // Accumulation
    content((0.2, 0.7), text("2. Accumulation (Kirchhoff's Law)"), anchor: "west")
    content((0.5, 0.2), text(fill: blue.darken(10%), [$I_"total" = sum I$]), anchor: "west")
  })

  group({
    translate((6.6, 0.5))
    content((0.5, 1.5), text([Memory ($G$) and Compute ($I=V G$)\ are physically co-located.]), anchor: "west")
    content((0.5, 0.5), text(weight: "bold", fill: green.darken(20%), "Zero data transport cost."), anchor: "west")
  })
})
