#import "@preview/mannot:0.2.2": markrect

#set text(
  font: "New Computer Modern",
  size: 12pt
)

#set page(
  paper: "us-letter",
  margin: (x: 1in, y: 1in),
  header: context {
    [
      #context document.title
      #h(1fr)
      Rajeev Atla
      #line(length: 100%, stroke: 0.5pt)
    ]
  },
  footer: context {
    [
      #line(length: 100%, stroke: 0.5pt)
      #set align(center)
      — Page
      #counter(page).display(
        "1 of 1 —",
        both: true,
      )
    ]
  }
)

#set document(
  title: "Exam 1 Questions Review",
  author: ("Rajeev Atla"),
  date: auto
)

#set par(
  justify: true
)

#set text(
  hyphenate: false
)

#show link: underline

#let codeblock(filename) = raw(read(filename), block: true, lang: filename.split(".").at(-1))


#let Cov(value) = $text("Cov")(#value)$


#align(center, text(16pt)[
  * #context document.title *
])
