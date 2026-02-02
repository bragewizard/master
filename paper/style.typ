#let style(it) = {
  set text(font: "Geist", size: 11pt, weight: "regular", top-edge:.7em)
  // show math.equation : set text(font:"New Computer Modern Math", size: 12pt, weight: "medium")
  // show math.equation : set text(font:"STIX Two Math", size: 12pt, weight: "medium")
  show math.equation : set text(font:"Latin Modern Math", size: 12pt, weight: "medium")
  show raw : set text(font:"GeistMono NF", weight: "medium", size:10pt)
  set list(marker: sym.bullet, indent: 1em)
  show heading: set text(font:"Geist",weight: "bold", style:"normal")
  show heading.where( level: 1 ): it => block(width: 100%)[
    #set align(left + horizon)
    #set text(28pt)
    #upper(it)
    #v(.8em)
  ]
  show heading.where( level: 2 ): it => block(width: 100%)[
    #set align(left + horizon)
    #set text(18pt)
    #upper(it)
    #v(.8em)
  ]
  show heading.where( level: 3 ): it => block(width: 100%)[
    #set align(left + horizon)
    #set text(14pt)
    #upper(it)
    #v(0.4em)
  ]
  show figure.caption: it => {
    set align(left)
    set par(justify: true)
    it
  }
  set page(fill:none, margin:auto, numbering: "1")
  set page(footer: align(center, text(size:10pt, weight: "medium", font:"Geist",
  context { counter(page).display("1") })))
  set par(justify: true)
  set heading(numbering: "1.1 \u{2022}")
  it
}

#let serif-text(body) = {
  set text(font: "Source Serif 4 18pt", size: 12pt, weight: "regular", top-edge:.7em)
  // set text(font: "STIX Two Text", size: 12pt, weight: "regular", top-edge:.7em)
  // set text(font: "Source Serif 4 18pt", size: 12pt, weight: "medium", top-edge:.7em)
  body
}

#let mono-text(body) = {
  set text(font: "GeistMono NF", size: 10pt, weight: "medium")
  body
}

#let box-text(body) = {
block(stroke:(thickness:0pt, paint:luma(0)), inset: 10pt, radius: 0pt, fill: luma(220),
  width: 100%)[#body]
}

#let mini-header(body) = {
  set text(font:"Geist",weight: "bold", size: 11pt)
  upper(body)
}
