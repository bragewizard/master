
#import "@preview/cetz:0.4.2"
// #show math.equation : set text(font:"TeX Gyre Schola Math", size: 11pt)
// #set page(width: auto, height: auto, margin: 0.6pt)

#cetz.canvas({
  import cetz.draw: *
  set-style(
    stroke:(thickness:2pt),
    cap: "round", join: "round"
  )

  let radius = 1
  let point_size = 2pt
  let xprev = 0
  let yprev = 0
  let x = 0
  let y = 0
  let xc = 0
  let yc = 0
  for i in range(7) {
    // Calculate the angle for each point.
    // 360deg / 6 points = 60deg per point.
    let angle = i * 60deg
    x = radius * calc.cos(angle)
    y = radius * calc.sin(angle)
    let anglec = i * 60deg - 30deg
    xc = radius * calc.cos(anglec) * 0.5
    yc = radius * calc.sin(anglec) * 0.5
    // Place a solid black circle at the (x, y) position.
    // circle((x,y),radius:.1)
    // circle((xprev,yprev),radius:.1, fill:blue)
    // circle((xc,yc),radius:.1)
    if i > 0 {
      bezier((x,y),(xprev,yprev),(xc,yc))
    }
    xprev = x
    yprev = y
      // circle(radius: point_size, fill: black)
  }
  circle((0,0),radius:.4)
  let startx = radius * calc.cos(180deg)
  let endx = -4
  let offset = 1
  let startx2 = radius * calc.cos(0deg)
  let end = (startx - 2, 0)
  let end2 = (startx2,0)

  let i = (endx,1.5)
  let j = (endx,.5)
  let k = (endx,0)
  let l = (endx,-.5)
  let m = (endx,-1.5)
  let n = (endx - offset,1.5)
  let o = (endx - offset,.5)
  let p = (endx - offset,0)
  let q = (endx - offset,-.5)
  let r = (endx - offset,-1.5)
  let i2 = (2,1.5)
  let j2 = (2,.5)
  let k2 = (2,0)
  let l2 = (2,-.5)
  let m2 = (2,-1.5)
  let n2 = (2.5,1.5)
  let o2 = (2.5,.5)
  let p2 = (2.5,0)
  let q2 = (2.5,-.5)
  let r2 = (2.5,-1.5)
  // line((startx,0),end)
  line(end,(startx,0), stroke:5pt)
  bezier(end, n, k, i)
  bezier(end, o, k, j)
  bezier(end, q, k, l)
  bezier(end, r, k, m)
})
