#let strings = (
  eng: (
    faculty: "Faculty of Mathematics and Natural Sciences",
    dept: "Department of Informatics",
    thesis-kind: "Master's thesis",
    bachelor-kind: "Bachelor's thesis",
    sp-name: "ECTS study points",
    words: "words",
    supervisor: "Supervisor",
    supervisors: "Supervisors",
    spring: "Spring",
    autumn: "Autumn",
  ),
  bm: (
    faculty: "Det matematisk-naturvitenskapelige fakultet",
    dept: "Institutt for informatikk",
    thesis-kind: "Masteroppgave",
    bachelor-kind: "Bacheloroppgave",
    sp-name: "studiepoeng",
    words: "ord",
    supervisor: "Veileder",
    supervisors: "Veiledere",
    spring: "Våren",
    autumn: "Høsten",
  ),
  nn: (
    faculty: "Det matematisk-naturvitenskaplege fakultet",
    dept: "Institutt for informatikk",
    thesis-kind: "Masteroppgåve",
    bachelor-kind: "Bacheloroppgåve",
    sp-name: "studiepoeng",
    words: "ord",
    supervisor: "Rettleiar",
    supervisors: "Rettleiarar",
    spring: "Våren",
    autumn: "Hausten",
  ),
)

// -------------------------------------------------
// %% Font Sizes
// -------------------------------------------------
#let small-size = 12pt
#let std-size = 14pt
#let big-size = 18pt
#let head-size = 32pt
// -------------------------------------------------
// %% Colours
// -------------------------------------------------
#let colors = (
  blue: (
    main: rgb(52.5%, 64.2%, 96.9%),
    light: rgb(90.2%, 92.5%, 100%),
  ),
  orange: (
    main: rgb(99.2%, 79.6%, 52.9%),
    light: rgb(100%, 91.0%, 83.1%)
  ),
  pink: (
    main: rgb(98.4%, 40.0%, 40.0%),
    light: rgb(99.6%, 87.8%, 87.8%),
  ),
  green: (
    main: rgb(48.4%, 88.2%, 67.1%),
    light: rgb(80.8%, 100%, 87.5%),
  ),
  gray: (
    main: rgb(70.0%, 70.0%, 70.0%),
    light: rgb(91.2%, 91.2%, 91.2%),
  ),
  grey: ( // Alias
    main: rgb(70.0%, 70.0%, 70.0%),
    light: rgb(91.2%, 91.2%, 91.2%),
  ),
)

#let cover-text(
  title,
  author,
  subtitle: none,
  program: none,
  sp: 2,
  sp-name: "?",
  co-sp-name: "?",
  dept: "Departement of Informatics",
  fac: "Faculty of Mathematics and Natural Sciences",
) = {
  block(width:210mm, [
    #text(size: head-size, weight: "bold",title)\
    #if subtitle != none {[
      #v(-.2em)
      #text(size: big-size, subtitle)\
    ]}
    #v(.8em)
    #text(size: std-size, weight: "semibold", author)
    #v(2em)
    #grid(columns: (38%,1fr), column-gutter: 0pt,
    [
      #block(width: 90%,[
      #text(size: std-size, dept)\
      #text(size: std-size, fac)])
    ],
    [
      #block(width: 100%,[
      #text(size: std-size, sp-name + " - " + "Supervisor")\
      #if sp == 2 {[
        #text(size: std-size, co-sp-name + " - " + "Co-Supervisor")
      ]}
      ])
    ]
    )
    #v(2cm)
    #if program != none {
      [
        #text(size: std-size, program)
        #parbreak()
      ]
    }
    #v(1.2em)
  ])
}

#let cover(
  uiosign: "04_uio_naventrekk_eng_pos.svg",
  uiologo: "04_UiO_segl_pos.svg",
  date: datetime.today().display(),
  body-font: "Geist",
  mono-font: "GeistMono NF",
) = {
  set page(margin: 0cm)
  page(
    grid(rows:2, columns:1,
    block(height: 92.7mm, width: 100%,[
      #place(top + left, rect(width: 100%, height: 100%, fill: colors.gray.main))
      #place(top + left, dx: -14mm, dy: -14mm, image(uiosign, height: 6.2cm))
      #place(bottom + left, dx: 14mm, dy: - 14mm,
        rect(
          fill: black,
          outset: 3mm,
          text(white, weight: "bold",size:small-size, "Master's Thesis"),
        )
      )
    ]),
    block(height: 204.3mm, width: 100%,[
      #place(bottom + left, rect(width: 100%,height: 100%, fill: colors.gray.light))
      #place(bottom + right, dx: 14mm, dy: 14mm, image(uiologo, width: 8.2cm))
      #place(bottom + left, dx: 11mm, dy: -14mm, text(size: small-size,font:body-font, date))
      #place(top + left, dx: 11mm, dy: 14mm,
        cover-text(
          "NEUROMORPHIC COMPUTING",
          "Brage Wiseth",
          subtitle: "With Spiking Neural Networks",
          sp-name: "Philip Haflinger",
          co-sp-name: "Yngve Hafting",
        )
      )
    ])
  ))
}
