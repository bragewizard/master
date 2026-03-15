#import "@preview/cetz:0.4.2"

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  // Global aesthetics matching your minimalist academic style
  set-style(
    stroke: (thickness: 1pt, cap: "round", join: "round"),
    mark: (fill: gray.darken(20%), scale: 0.8)
  )

  // 1. Title
  content((5, 6.5), text(weight: "bold", size: 12pt, "Convolutional Neural Network (CNN) Architecture"))

  // ====================================================================
  // 2. CONFIGURATION DICTIONARY (EDIT THIS TO CHANGE THE NETWORK)
  // ====================================================================
  // c: Thickness (Channels), w: Width, h: Height
  let architecture = (
    (name: "Input",   c: 0.2, w: 4.0, h: 4.0, col: gray,   dims: "32x32x3"),
    (name: "Conv1",   c: 1.0, w: 3.5, h: 3.5, col: blue,   dims: "28x28x16"),
    (name: "Pool1",   c: 1.0, w: 1.7, h: 1.7, col: red,    dims: "14x14x16"),
    (name: "Conv2",   c: 2.0, w: 1.3, h: 1.3, col: blue,   dims: "10x10x32"),
    (name: "Pool2",   c: 2.0, w: 0.6, h: 0.6, col: red,    dims: "5x5x32"),
    (name: "Flatten", c: 0.4, w: 0.4, h: 4.0, col: purple, dims: "800"),
    (name: "Dense1",  c: 0.4, w: 0.4, h: 2.0, col: purple, dims: "128"),
    (name: "Output",  c: 0.4, w: 0.4, h: 0.5, col: green,  dims: "10"),
  )

  let gap = 1.2 // Space between each layer in the 3D projection

  // ====================================================================
  // 3. PROCEDURAL 3D DRAWING ENGINE
  // ====================================================================
  // Shift the whole network down slightly so the title has room
  group({
    translate((0, 1.5))
    
    // Native 3D Orthographic Projection
    ortho(x: 30deg, y: -25deg, {
      let current_x = 0.0

      for (i, layer) in architecture.enumerate() {
        let c = layer.c
        let w = layer.w
        let h = layer.h
        let col = layer.col
        
        // A. Draw connecting flow lines from the previous layer
        if i > 0 {
          line((current_x - gap, 0, 0), (current_x, 0, 0), 
               mark: (end: ">"), stroke: (paint: gray.darken(20%), thickness: 1.5pt, dash: "dashed"))
        }

        // B. Draw the 3D Block (Back-to-Front face rendering)
        // Side face (XZ plane) - Left side of the box
        on-xz(y: -w/2, {
          rect((current_x, -h/2), (current_x + c, h/2), fill: col.darken(15%), stroke: black)
        })
        
        // Top face (XY plane) - Top of the box
        on-xy(z: h/2, {
          rect((current_x, -w/2), (current_x + c, w/2), fill: col.lighten(15%), stroke: black)
        })
        
        // Front face (YZ plane) - Front of the box closest to viewer
        on-yz(x: current_x + c, {
          rect((-w/2, -h/2), (w/2, h/2), fill: col, stroke: black)
          
          // C. Labels (Anchored to the front face so they float in perfect 3D perspective)
          content((0, h/2 + 0.6), text(weight: "bold", size: 8pt, fill: col.darken(30%), layer.name))
          content((0, -h/2 - 0.6), text(size: 7pt, fill: gray.darken(50%), layer.dims))
        })

        // Advance the X coordinate for the next layer
        current_x += c + gap
      }
    })
  })

  // ====================================================================
  // 4. COLOR LEGEND
  // ====================================================================
  group({
    translate((2, -2.5))
    let legend_items = (
      (name: "Input / Map", col: gray),
      (name: "Convolution", col: blue),
      (name: "Pooling / Subsample", col: red),
      (name: "Fully Connected", col: purple),
      (name: "Softmax Output", col: green)
    )
    
    for (i, item) in legend_items.enumerate() {
      let x_pos = i * 2.8
      rect((x_pos, 0), (x_pos + 0.3, 0.3), fill: item.col, stroke: black)
      content((x_pos + 0.4, 0.15), text(size: 8pt, item.name), anchor: "west")
    }
  })
})
