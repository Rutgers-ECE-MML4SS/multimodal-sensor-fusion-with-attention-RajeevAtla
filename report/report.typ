#import "@preview/mannot:0.2.2": markrect

#set text(
  font: "New Computer Modern",
  size: 11pt
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
  title: "Multimodal A2: Sensor Fusion with Attention",
  author: ("Rajeev Atla"),
  date: auto
)

#set par(
  justify: true
)

#set text(
  hyphenate: false
)

#set heading(
  numbering: "1.",
)

#show link: underline

#let codeblock(filename) = raw(read(filename), block: true, lang: filename.split(".").at(-1))


#let Cov(value) = $text("Cov")(#value)$


#align(center, text(16pt)[
  * #context document.title *
])

#align(center)[
  Rajeev Atla
]


= Introduction

Human activity recognition underpins many wellness applications,
from rehabilitation and fall detection to everyday fitness tracking.
Inertial measurement units are a natural starting point for motion understanding,
but they struggle during low-motion or stationary states such as sleep or quiet rest.
Physiological signals like heart rate supply complementary context that can stabilize predictions in those edge cases.
Fusing physiological and kinematic data can therefore lift robustness and overall accuracy,
especially when sensors are noisy or degraded.

This report studies multimodal fusion on the PAMAP2 Physical Activity Monitoring dataset @pamap2_paper @pamap2_uci,
which aligns heart rate with IMU streams from the accelerometer,
gyroscope,
and magnetometer across 18 activities.
The task is challenging for three reasons:
different sampling rates between modalities,
missing or unreliable measurements that include IMU drift and noisy heart rate,
and temporal offsets such as delayed heart rate changes after a subject sits down.
Fusion models also risk overfitting to the denser modality,
which hurts validation performance and transfer when inputs go missing.

= Approach

We compare three fusion strategies for activity recognition: early, late, and hybrid fusion.

*Early fusion*: Signals are temporally aligned and concatenated at the input level,
then processed by a shared encoder. 
This captures joint correlations but can overfit to the most informative or highest-rate stream and absorb its noise.

*Late fusion*:
Each modality is encoded by a separate transformer,
and predictions are combined near the output.
This preserves modularity and can improve calibration,
but it may miss fine-grained cross-modal interactions that arise earlier in the pipeline.

*Hybrid fusion*:
Separate encoders are linked with cross-attention so that features exchange information at multiple depths.
Each encoder continues to learn modality-specific structure while cross-attention adjusts the relative influence of signals in context.

= Results

== Fusion Comparison

== Attention 

== Uncertainty Calibration

== Ablation Studies

= Discussion

= Conclusion

#show bibliography: set heading(numbering: "1.")
#bibliography("citations.bib", title: "References", style: "ieee")