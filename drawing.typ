#import "@preview/cetz:0.3.1"
#import "@preview/suiji:0.3.0": *

#set text(font: "Source Serif 4")
#show math.equation: set text(font:"STIX Two Math")

#set page(fill:none,paper:"presentation-16-9")

#align(center + horizon,
cetz.canvas({
  import cetz.draw: *
  let v = ()
  let first_four = ()
  let others = ()
  let rng = gen-rng(42)
  let letters = ("A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z")
  for i in range(26) {
    (rng, v) = uniform(rng, low: 2, high: 18.0, size: 1)
    let x = int(v.at(0))
    let y = i*0.6
    v = (x,y)
    if first_four.len() < 5 {
      first_four.push(v)
    }
    else if v.at(0) > first_four.first().at(0) {
      others.push(first_four.remove(0))
      first_four.push(v)
    }
    else { others.push(v) }
    circle((0,y),radius:.2, fill:black) 
    line((0,y),(20,y))
    content((-0.6,(25-i)*0.6), letters.at(i))
    first_four = first_four.sorted()
  }
  for i in range(21) {
    circle((others.at(i)),radius:.2, fill:red,stroke:none) 
  }
  for i in range(4) {
    circle((first_four.at(i)),radius:.2, fill:green,stroke:none) 
    content((rel:(0,.4)),str(4-i))
  }
}))

#pagebreak()


#align(center + horizon,
cetz.canvas({
  import cetz.draw: *
  let val = (250,230,200,180,100,70,50,40)
  content((-0.4,3),text(size:60pt,${$))
  line((1,5),(4,7.5),stroke:4pt, mark:(end:">"))
  content((0.8,4.4),text(size:60pt,${$),angle:-90deg)
  content((5,7.8),text(size:120pt,${$))

  rect((2,1),(rel:(1,1)),fill:luma(val.at(0)))
  rect((3,1),(rel:(1,1)),fill:luma(val.at(5)))
  rect((3,0),(rel:(1,1)),fill:luma(val.at(1)))
  rect((2,0),(rel:(1,1)),fill:luma(val.at(6)))
  rect((0,1),(rel:(1,1)),fill:luma(val.at(7)))
  rect((1,1),(rel:(1,1)),fill:luma(val.at(6)))
  rect((1,0),(rel:(1,1)),fill:luma(val.at(6)))
  rect((0,0),(rel:(1,1)),fill:luma(val.at(7)))
  rect((2,3),(rel:(1,1)),fill:luma(val.at(7)))
  rect((3,3),(rel:(1,1)),fill:luma(val.at(7)))
  rect((3,2),(rel:(1,1)),fill:luma(val.at(7)))
  rect((2,2),(rel:(1,1)),fill:luma(val.at(5)))
  rect((0,3),(rel:(1,1)),fill:luma(val.at(1)))
  rect((1,3),(rel:(1,1)),fill:luma(val.at(6)))
  rect((1,2),(rel:(1,1)),fill:luma(val.at(0)))
  rect((0,2),(rel:(1,1)),fill:luma(val.at(6)))
  let order = (0,5,1,6,7,6,6,7,7,7,7,5,1,6,0,6)

  let v = ()
  let first_four = ()
  let others = ()

  for i in range(16) {
    rect((6,i -6),(rel:(1,1)),fill:luma(val.at(order.at(i))))
    circle(((val.at(order.at(i)) * 0.07)+6,i -5.5),radius:.2, fill:black) 
    line((7,i -5.5),(26,i -5.5))
    first_four = first_four.sorted()
  }

}))
