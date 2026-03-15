#import "@preview/cetz:0.4.2"

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  // Global aesthetics matching your minimalist academic style
  set-style(
    stroke: (thickness: 1pt, cap: "round", join: "round"),
    rect: (radius: 0.15) // Subtle rounding for nodes
  )

  // 1. Title
  content((5, 9), text(weight: "bold", size: 14pt, "Evaluation Metrics: Classification vs. Object Detection"))

  // ====================================================================
  // 2. CONFIGURATION DICTIONARY (EDIT THIS TO ADD/CHANGE METRICS)
  // ====================================================================
  let task_groups = (
    (
      name: "Classification",
      col: blue,
      x: 2.0,
      desc: "Predicts global image label",
      metrics: (
        (name: "Accuracy",   formula: "(TP + TN) / Total"),
        (name: "Precision",  formula: "TP / (TP + FP)"),
        (name: "Recall",     formula: "TP / (TP + FN)"),
        (name: "F1-Score",   formula: "Harmonic Mean of P & R")
      )
    ),
    (
      name: "Object Recognition",
      col: red,
      x: 8.0,
      desc: "Predicts label + bounding box",
      metrics: (
        (name: "IoU", formula: "Area of Overlap / Area of Union"),
        (name: "AP",  formula: "Area under PR Curve"),
        (name: "mAP", formula: "Mean AP across all classes")
      )
    )
  )

  // ====================================================================
  // 3. PROCEDURAL DRAWING ENGINE
  // ====================================================================
  let root_y = 7.5
  let box_w = 4.5
  let box_h = 1.0
  let gap_y = 1.4

  // Draw Root Node
  rect((5 - box_w/2, root_y), (5 + box_w/2, root_y + box_h), 
       fill: gray.lighten(70%), stroke: gray.darken(20%), name: "root")
  content("root", text(weight: "bold", size: 11pt, "Model Evaluation Domains"))

  // Iterate over domains
  for group in task_groups {
    let gx = group.x
    let gcol = group.col
    let header_y = 5.5

    // Connecting lines from root (Orthogonal routing)
    line("root.bottom", (5, 6.7), (gx, 6.7), (gx, header_y + box_h),
         mark: (end: ">"), stroke: (paint: gray.darken(20%), thickness: 1.2pt))

    // Group Header Box
    rect((gx - box_w/2, header_y), (gx + box_w/2, header_y + box_h), 
         fill: gcol.lighten(70%), stroke: gcol, name: "h_" + group.name)
    content("h_" + group.name, [
      #text(weight: "bold", fill: gcol.darken(30%), group.name) \
      #text(size: 7pt, fill: gray.darken(50%), group.desc)
    ])

    // Metrics Nodes
    let current_y = header_y - gap_y
    let prev_node = "h_" + group.name

    for (i, metric) in group.metrics.enumerate() {
      let node_name = "m_" + group.name + str(i)

      // Metric Box (Slightly narrower than header)
      rect((gx - box_w/2 + 0.3, current_y), (gx + box_w/2 - 0.3, current_y + box_h*0.8),
           fill: gcol.lighten(90%), stroke: (paint: gcol.lighten(20%), dash: "solid"), name: node_name)

      // Metric Content
      content(node_name, [
        #text(weight: "bold", size: 9pt, metric.name) \
        #text(size: 7pt, fill: gray.darken(30%), metric.formula)
      ])

      // Directional arrow from previous node
      line(prev_node + ".bottom", node_name + ".top", 
           mark: (end: ">"), stroke: (paint: gcol.lighten(20%)))

      prev_node = node_name
      current_y -= gap_y
    }
  }

  // ====================================================================
  // 4. LEGEND / FOOTNOTE
  // ====================================================================
  content((5, -1.0), text(size: 8pt, fill: gray.darken(20%), style: "italic",
    "TP: True Positive   |   TN: True Negative   |   FP: False Positive   |   FN: False Negative"
  ))
})
