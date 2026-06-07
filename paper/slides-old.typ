#import "@preview/touying:0.7.3": *
#import themes.simple: *
#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge

#show: simple-theme.with(
  aspect-ratio: "16-9",
  footer: [Neuromorphics],
  primary: rgb("#1b1b1b"),
)

#set page(fill:rgb("#F8F5EF"))
#set text(font: "Geist", size: 16pt, weight: "medium", top-edge:.7em, fill:rgb("#202020"))
#show math.equation : set text(font:"Latin Modern Math", size: 18pt, weight: "medium")
#show raw : set text(font:"GeistMono NF", weight: "medium", size:14pt)
#set list(marker: sym.bullet, indent: 1em)
#show heading: set text(font:"Geist",weight: "bold", style:"normal")
#show heading.where( level: 1 ): it => block(width: 100%)[ #set text(28pt); #upper(it) ]
#show heading.where( level: 2 ): it => block(width: 100%)[ #set text(18pt); #upper(it) ]
#show heading.where( level: 3 ): it => block(width: 100%)[ #set text(14pt); #upper(it) ]
#set par(justify: true)

#let box-text(body) = {
block(stroke:(thickness:1pt, paint:white), inset: 12pt, radius: 0pt, fill:rgb("#E4E1DB"),
  width: 100%)[#body]
}

#title-slide[
  = ON NEUROMORPHIC COMPUTING\
    WITH SPIKING NEURAL NETWORKS
  #v(1em)
  08.06.2026
]

= INTRODUCTION AND MOTIVATION

- AI has done cool things [cite]
- AI is not effecient enough [cite]
- The brain is effecien [cite]

#box-text()[ lets draw more inspiration from biological inteligence -> neuromorphic computing ]

#pagebreak()
Techniques to learn from the brain (will go into detail later):

- *Sparse:* Only sends and computes when needed
- *Event driven:* Highly effecient neuron communication (binary spikes)
- *Local:* Neurons are independent on global state and compute and memory is unified (physically in the same place)


== RESEARCH OBJECTIVES

#box-text()[
*Sparse Efficient Computing:* Investigate whether biologically inspired, event-driven algorithms reduce the computational footprint of visual classification when simulated on standard hardware.

*Neuron Model Evaluation:* Identify which temporal integration dynamics are compatible with Time-To-First-Spike (TTFS) rank-order decoding across systematic threshold regimes.

*Inference Via Weight Transfer:* Quantify the accuracy penalty of zero-shot Artificial-Neural-Network (ANN) to Spiking-Neural-Network (SNN) weight transfer under TTFS encoding, isolating the cost of transitioning to event-driven integration.

*Native Unsupervised Learning:* Determine if local Spike-Time-Dependent-Plasticity (STDP) can autonomously extract meaningful geometric features from visual input without global error signals.
]

= BIOLOCOGAL PRINCIPLES


== NEURON STRUCTURE & FUNCTION


== ACTION POTENTIAL & SPIKE TRAINS

#align(center + horizon)[
#include "figures/spiketrain.typ"
]

== NEURON MODELS


== ENCODING INFORMATION IN SPIKES

== RATE ENCODING
- Unlike Rate Coding, which requires extended time windows, TTFS encodes stimulus intensity inversely to response latency[cite: 1].
- *Mechanism:* A high-intensity (bright) pixel triggers an early spike, compressing spatial information into a priority-driven queue[cite: 1].
- *Advantages:*
  - *Sparsity:* Sub-threshold noise (background pixels) is aggressively discarded and never fires[cite: 1].
  - *Latency:* Eliminates the need to wait for a time window to close; processing begins as soon as salient features arrive[cite: 1].
- Phase ambiguity is resolved by simulating a biological saccade, establishing a global temporal reference frame ($t_0$)[cite: 1].

== TIME-TO-FIRST-SPIKE (TTFS) ENCODING
- Unlike Rate Coding, which requires extended time windows, TTFS encodes stimulus intensity inversely to response latency[cite: 1].
- *Mechanism:* A high-intensity (bright) pixel triggers an early spike, compressing spatial information into a priority-driven queue[cite: 1].
- *Advantages:*
  - *Sparsity:* Sub-threshold noise (background pixels) is aggressively discarded and never fires[cite: 1].
  - *Latency:* Eliminates the need to wait for a time window to close; processing begins as soon as salient features arrive[cite: 1].
- Phase ambiguity is resolved by simulating a biological saccade, establishing a global temporal reference frame ($t_0$)[cite: 1].


#grid(columns: 2,inset: 20pt, include "figures/rateencoding.typ" ,include "figures/temporalcoding.typ")
#grid(columns: 2,inset: 20pt, [#h(3.5cm) Rate encoded #h(7cm)], [TTFS encoded])




== BIOLOGICAL NEURAL NETWORKS

- Inhibition patterns



== BIOLOGICAL LEARNING

- STDP




= OPTIMIZATION


= METHOD

== DECODING TEMPORAL SEQUENCES
- Evaluated four architectures (IF, LIF, Linear Ramp, State Discount) across saturation, critical, and deficit threshold regimes[cite: 1].
- *The LIF Misalignment:* Under critical constraints, the standard Leaky Integrate-and-Fire model actively penalized early, high-salience spikes due to its exponential decay[cite: 1].
- *The Solution (Model C):* Developed a Current-Accumulating Linear Ramp model combining integration momentum with a strict 10-tick coincidence window[cite: 1].
- Model C successfully preserved rank-order sequence priority without the computational overhead of continuous exponentials[cite: 1].


== EXPERIMENT SETUP

- *Dataset:* MNIST (28x28 normalized grayscale images), chosen for its high degree of spatial sparsity[cite: 1].
- *Topology:* Fully Connected Network (FCN) $784 arrow.r 128 arrow.r 10$[cite: 1].
  - An FCN allows mathematically transparent, one-to-one parameter mapping without the structural overhead of convolutional unrolling[cite: 1].
- *SNN Simulator:* Custom PyTorch discrete-time engine operating over a $T_{max} = 64$ tick saccade window[cite: 1].
- *Metrics:* Top-1 Accuracy, Temporal Latency (Time-to-Decision), and Synaptic Operations (SyOPs) as a hardware proxy[cite: 1].

== PHASE I: NEURON MODEL EVALUATION
== PHASE II: ZERO-SHOT WEIGHT TRANSFER
== PHASE III: LEARNING

= RESULTS

== PHASE I: NEURON MODEL EVALUATION
== PHASE II: ZERO-SHOT WEIGHT TRANSFER
== PHASE III: LEARNING



= DISCUSSION

== FUTURE WORK

= CONCLUSION

#pagebreak()
#bibliography("references.bib")
