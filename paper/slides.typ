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

== INTRODUCTION AND MOTIVATION

- Deep learning has exceeded human performance in some domains and specific problems — AlphaFold, GPT, AlphaGo
  @jumper_highly_2021, @silver_mastering_2016
- Yet training state-of-the-art models consumes vast energy and is rapidly approaching physical and economic limits
  @strubell_energy_2019, @geirhos_shortcut_2020, @horowitz_11_2014, @martin_synaptic_2000, @friedman_clock_2001
- The human brain operates on ~20 W and still outperforms AI on adaptation, few-shot learning, and common-sense
  reasoning @laughlin_energy_2001

#box-text()[Let's draw more inspiration from biological intelligence → *neuromorphic computing*

Key takeaway lessons from the brain and what we should base our neuromorphic architecture on:]

- *Sparse:* Only sends and computes when needed — silence is free
- *Event driven:* Highly efficient neuron communication via binary spikes; information encoded in timing, not magnitude
- *Local:* Neurons hold their own weights; compute and memory are co-located, physically in the same place (eliminating the von Neumann bottleneck)


== RESEARCH OBJECTIVES

#box-text()[
*Sparse Efficient Computing:* Investigate whether biologically inspired, event-driven algorithms reduce the computational footprint of visual classification when simulated on standard hardware.

*Neuron Model Evaluation:* Identify which temporal integration dynamics are compatible with Time-To-First-Spike (TTFS) rank-order decoding across systematic threshold regimes.

*Inference Via Weight Transfer:* Quantify the accuracy penalty of zero-shot Artificial-Neural-Network (ANN) to Spiking-Neural-Network (SNN) weight transfer under TTFS encoding, isolating the cost of transitioning to event-driven integration.

*Native Unsupervised Learning:* Determine if local Spike-Time-Dependent-Plasticity (STDP) can autonomously extract meaningful geometric features from visual input without global error signals.
]

#v(2cm)
First let's look at how the brain works...

== NEURON STRUCTURE & FUNCTION

#v(1cm)
A biological neuron is a highly specialized cell that is the foundational building block in the nervous system. Functionally it is a event-driven processor with three functional zones:

- *Dendrites (Input):* Branching tree collecting signals from thousands of upstream neurons via synaptic terminals
- *Soma (Integration):* Cell body summing competing excitatory and inhibitory inputs; acts as a biological capacitor
- *Axon (Output):* Long cable transmitting the neuron's output spike to downstream targets; insulated by myelin for fast conduction

@kandel_principles_2021,  @gerstner_neuronal_2014

#box-text()[
*All-or-nothing principle:* If membrane potential exceeds the threshold, voltage-gated channels cascade open → rapid depolarisation spike. Sub-threshold noise is silently discarded. The neuron then enters a *refractory period*, resetting the gradient.
]


== ACTION POTENTIAL & SPIKE TRAINS

#align(center)[
#include "figures/actionpotential.typ"
]

#box-text()[Information is encoded entirely in the *precise timing* of spikes — abstracted as a sum of Dirac delta functions: $S(t) = sum_f delta(t - t_f)$]

#align(center)[
#include "figures/spiketrain.typ"
]
Because the spike waveform is stereotypical and invariant across neurons, *the waveform itself carries no information.*
@gerstner_neuronal_2014, @metcalfe_action_2020

== NEURON MODELS

A fundamental trade-off: *biological realism* vs *computational efficiency*

- *Hodgkin-Huxley:* Full ionic channel dynamics — accurate, but requires supercomputing for large networks
- *Integrate-and-Fire (IF):* Pure arithmetic accumulator; fires when $u > theta.alt$, then resets. Minimal overhead — $O(1)$ per spike.
- *Leaky Integrate-and-Fire (LIF):* Adds exponential membrane decay $tau_m$:
- *Generalised LIF (GLIF):* Adds an adaptation variable $w(t)$ enabling bursting, frequency adaptation, and richer firing patterns

@hodgkin_quantitative_1952, @gerstner_neuronal_2014

== LIF

#grid(columns: 2, inset: 20pt,  include "figures/lifcircuit.typ", include "figures/lifdynamics.typ")

#box-text()[ The LIF forms the foundation of the other neuron models in this thesis. The main properties of the LIF are:

*integration*,
*firing when reaching threshold*,
*leaking*
]

== ENCODING INFORMATION IN SPIKES

The brain occupies a unique middle ground between digital and analog:

- Spike output is *binary* — discrete, stereotypical events.
- Spikes can arrive and emit at any time, so it is *continuous in the time domain*.

This creates a *quantised continuous signal*: discrete amplitude, continuous time.

#box-text()[
Requires a suitable codec to be useful as a communication medium.

Two primary coding schemes are hypothesised...
]


== RATE ENCODING

- Stimulus intensity is encoded as *mean firing frequency* over a time window — stronger stimulus → more spikes per second
- *Advantages:*
  - Simple and robust; directly observed in motor and sensory neurons
- *Limitations:*
  - latency barrier: The post-synaptic neuron must integrate spikes over tens–hundreds of milliseconds to estimate a reliable rate
  - This contradicts biological reaction times often under 100 ms, suggesting rate coding alone cannot account for time-critical processing @thorpe_speed_1996
  - On digital hardware, high firing rates cause rapid transistor switching — costly for power draw and bus congestion

@adrian_discharge_1929, @gerstner_neuronal_2014

== TIME-TO-FIRST-SPIKE (TTFS) ENCODING

- Unlike Rate Coding, which requires extended time windows, TTFS encodes stimulus intensity inversely to response latency
- *Mechanism:* A high-intensity (bright) pixel triggers an early spike, compressing spatial information into a priority-driven queue
- *Advantages:*
  - *Sparsity:* Sub-threshold noise (background pixels) is aggressively discarded and never fires
  - *Early Exit:* Salient features arrive first, enabling confident decisions within the first ~15% of the temporal window
  - *Latency:* Eliminates the need to wait for a time window to close; processing terminates as soon as the threshold is reached
- *Limitations:*
  - *Phase ambiguity* -- cannot determine whether a spike represents a delayed response to a previous stimulus or an early response to a new one without a reference signal.
  - *Sensitive to noise*

@rullen_rate_2001, @gerstner_neuronal_2014

#v(2cm)

#grid(columns: 2, inset: 20pt, include "figures/rateencoding.typ", include "figures/temporalcoding.typ")
#grid(columns: 2, inset: 20pt, [#h(3.5cm) Rate encoded #h(7cm)], [TTFS encoded])


== TTFS PHASE AMBIGUITY
#align(center + horizon)[
#include "figures/phaseambiguity.typ"
]

Spikes occurring at the same relative phase ($phi_1$ and $phi_2$) across different oscillation cycles are mathematically indistinguishable ($phi_1 = phi_2 (mod 2pi)$). Without a mechanism to track the global cycle count, downstream neurons cannot determine whether a spike represents a delayed response to a previous stimulus or an early response to a new one.

== BIOLOGICAL NEURAL NETWORKS

*Lateral Inhibition and Winner-Takes-All (WTA):*

- An active excitatory neuron stimulates nearby inhibitory neurons, which in turn suppress competing excitatory neighbours
- This creates a *WTA dynamic* — the most active neuron silences its competitors, providing a physical mechanism for categorical decisions without a central processor

#box-text()[
*Excitation-Inhibition balance is critical:*
- Excess excitation → runaway feedback loops (analogous to seizures)
- Excess inhibition → signal extinction (quiescence)

The brain operates at the critical point between these two extremes. Achieving this balance is essential in engineered SNNs.
]


== BIOLOGICAL LEARNING

*Spike-Timing-Dependent Plasticity (STDP):*

A causal, millisecond-scale refinement of Hebbian learning. Weight update depends on the timing difference $Delta t = t_"post" - t_"pre"$:

#box-text()[
- *Long-Term Potentiation (LTP):* Pre fires *before* post ($Delta t > 0$) — input contributed to firing → synapse *strengthened*
- *Long-Term Depression (LTD):* Pre fires *after* post ($Delta t < 0$) — input was irrelevant → synapse *weakened*
]

- Fully local: no global error signal required — only pre- and post-synaptic activity at the cleft
- *Homeostatic plasticity* prevents runaway growth: if a neuron's average rate exceeds a target, all incoming weights are globally scaled down


#set text(size:12pt)
#align(center)[
#include "figures/stdpcurve.typ"
]
#set text(size:16pt)



== BACK TO WHY DEEP LEARNING IS INEFFICIENT

Deep learning shares many core ideas with neuroscience but has diverged into its own discipline over time.

=== SIMILARITIES

- Deep learning has a version of neurons (the perceptron); it integrates and fires when reaching a threshold.
- It has neural networks—neurons connected together in hierarchical topologies to extract features.

=== DIFFERENCES

- *Signals:* ANNs use continuous, synchronous floating-point values (activations). The brain uses discrete, asynchronous events (spikes).
- *Optimization:* ANNs rely on backpropagation (global error gradients and exact weight transport). The brain relies on local plasticity (STDP).
- *Hardware Mapping:* ANNs execute dense matrix multiplications regardless of data sparsity. Biological networks communicate sparsely, saving massive amounts of energy.

== BOTTLENECKS OF DEEP LEARNING

Standard deep learning imposes four compounding hardware bottlenecks:

- *Von Neumann Bottleneck:* Weights shuttle between memory and compute each step. Moving 32 bits from DRAM costs ~640 pJ; computing with them costs ~0.1 pJ — *Energy is wasted on transport, not computation*
- *Dense Processing of Sparse Data:* GPUs execute $0 times w = 0$ multiplications for zero activations — structurally blind to sparsity, even when ReLU produces mostly-zero activation maps
- *Clock Synchrony Tax:* The global clock distribution network alone consumes 30–40% of chip power, even when circuits are idle
- *Backpropagation Locking:* All intermediate activations must remain in VRAM for the full backward pass; local synapses cannot adapt until the global error loop completes

== NEUROMORPHIC PRINCIPLES

Three architectural pillars that directly address each bottleneck:

#box-text()[
*Co-location of Memory and Compute:*
Synaptic weights live with the neuron — zero data transport cost. Learning and inference are co-located.

*Event-Driven Asynchrony:*
No global clock. Energy scales with task activity, not network size. Silent neurons consume nothing.

*Sparse Binary Communication:*
Information is encoded in spike timing, not magnitude. Bandwidth scales with information content, not data dimensionality.

*Local learning rules:* Synapses can update independent of eachother avoiding the backpropagation lock.
]

== METHOD

To answer the research questions:
#box-text()[
*Sparse Efficient Computing:* Investigate whether biologically inspired, event-driven algorithms reduce the computational footprint of visual classification when simulated on standard hardware.

*Neuron Model Evaluation:* Identify which temporal integration dynamics are compatible with Time-To-First-Spike (TTFS) rank-order decoding across systematic threshold regimes.

*Inference Via Weight Transfer:* Quantify the accuracy penalty of zero-shot Artificial-Neural-Network (ANN) to Spiking-Neural-Network (SNN) weight transfer under TTFS encoding, isolating the cost of transitioning to event-driven integration.

*Native Unsupervised Learning:* Determine if local Spike-Time-Dependent-Plasticity (STDP) can autonomously extract meaningful geometric features from visual input without global error signals.
]

We propose the following experimental setup...

== EXPERIMENT SETUP

The experiments are divided into three phases:

- *Phase I:* Neuron model evaluation — finding a suitable and efficient neuron model to use for Phase II and Phase III.
- *Phase II:* Weight transfer — using a pre-trained network via established methods and transferring weights to an SNN to investigate computational savings.
- *Phase III:* Training an SNN from scratch with zero dependence on traditional frameworks for potentially maximum efficiency.

== DATASET

MNIST (28×28 normalized grayscale images), chosen for its simplicity and high degree of spatial sparsity.
Need to encode image as TTFS spike train ($t_i = (1 - p_i) dot 64$, $p_i in [0,1]$, background suppressed)



#figure( image("figures/mnist_grid.png", width:60%), caption: [Sample of the MNIST dataset. The 28x28 images are normalized and flattened into 1D vectors before being translated into temporal spike events.]) <fig:mnist_grid>



== NETWORK ARCHITECTURE

- *Topology:* Fully Connected Network (FCN) $784 arrow.r 128 arrow.r 10$
  - An FCN allows mathematically transparent, one-to-one parameter mapping without the structural overhead of convolutional unrolling
  - *No bias terms* — to preserve pure event-driven sparsity and simplify ANN→SNN transfer
- *SNN Simulator:* Custom PyTorch discrete-time engine operating over a $T_"max" = 64$ tick saccade window (arbitrary choice)
- *WTA:* Winner-Takes-All (WTA) — first output neuron to cross threshold claims the class label
  WTA is also used in the hidden layer durin learning with STDP

#set text(size:12pt)

#align(center + horizon)[
  #include("figures/softwarearch.typ")
]
#set text(size:16pt)

== PHASE I: NEURON MODEL EVALUATION

A *unit test* for isolated neuron temporal dynamics. Two synthetic spike trains are constructed:

- *Concordant:* Strongest synaptic weights arrive *first* — aligned with TTFS rank-order priority
- *Discordant:* Strongest weights arrive *last* — inverted temporal order

Each model is then evaluated across three threshold regimes:

#box-text()[
- *Saturation (low $theta.alt$):* Does the model fire early when evidence is abundant?
- *Critical (balanced $theta.alt$):* Can the model discriminate *temporal order*, not just total magnitude?
- *Deficit (high $theta.alt$):* Does the model fail gracefully when total weight is insufficient to cross the threshold?
]

No MNIST data is used — this phase isolates the neuron's mathematical response to controlled spike patterns.

#align(center + horizon)[
#include "figures/phase1pattern.typ"
]


== IF MODEL (A)
#align(center)[
#include "figures/ifmodel.typ"
]

== LIF MODEL (B)
#align(center)[
#include "figures/lifdynamics.typ"
]

== LINEAR ACUMULATING RAMP MODEL (C)
#align(center)[
#include "figures/rampmodel.typ"
]

== STATE DISCOUNT MODEL (D)
#align(center)[
#include "figures/discountmodel.typ"
]

== PHASE II: ZERO-SHOT WEIGHT TRANSFER

1. Train a standard ANN baseline ($784 arrow.r 128 arrow.r 10$, Adam optimizer, 1000 epochs, no bias terms)
2. Copy FP32 weights directly to a structurally identical SNN — scaled by $times 64$ to fit with the neuruon models (arbitrary choice)
3. Run SNN inference using Model C over $T_"max" = 64$ ticks with WTA output

#box-text()[
*Goal:* Measure the accuracy lost purely by switching from static continuous dot-product activations to discrete TTFS momentum integration, with *no retraining*.
]

Efficiency is tracked via SyOPs vs MACs and mean time-to-decision latency across the 10,000-image test set.

== PHASE III: LEARNING

The SNN is trained *from scratch* — randomized weights, no labels, no backpropagation.
Learning loop per image (one saccade):
1. Forward pass — all 128 hidden neurons integrate independently via Model C
2. *Post-hoc Hard WTA:* Only the single earliest-firing neuron is permitted to update
3. *Vectorized STDP:* Pre-spikes before the winner's time → LTP ($+A_+$); after or silent → LTD ($-A_-$)
4. *Homeostatic adaptation:* Winner receives threshold penalty ($+600$); all thresholds decay ($times 0.90$)

#box-text()[
Post-hoc labeling after training: pass labeled data through the frozen network; assign each output neuron the digit class it most frequently wins for. Then measure accuracy on the test set.
]


== METRICS

- Top-1 Accuracy
- Temporal Latency (Time-to-Decision)
- Synaptic Operations (SyOPs) as a hardware proxy


== PHASE I: NEURON MODEL EVALUATION -- RESULTS


#align(center)[
#image("figures/phase1_composite_sweep.png")
]

== PHASE II: ZERO-SHOT WEIGHT TRANSFER -- RESULTS

#v(0.5em)
#align(center)[
#image("figures/phase2_cumulative_accuracy.png", width: 55%)
]

#v(1cm)


#figure(
  table(
    columns: (1.2fr, 1fr),
    inset: 10pt,
    align: center,
    [*Model Configuration*], [*Accuracy (%)*],
    [ANN (Static FP32, Bias=False)], [98.40%],
    [SNN (Spiking Model C, FP32)], [94.50%],
  ),
  caption: [Performance degradation breakdown. The Temporal Penalty isolates the loss of TTFS encoding without any weight alteration besides scaling.],
  kind: "table",
  supplement: [Table]
) <tbl:phase2_accuracy>

#figure(
  table(
    columns: (1.2fr, 1.0fr, 0.8fr, 0.8fr, 0.6fr),
    inset: 10pt,
    align: center,
    [*Architecture*], [*Metric Target*], [*Avg. Latency*], [*Avg. Operations / Image*],[*Compute Reduction*],
    [ANN (Static FP32)], [Dense MACs], [N/A (Static)], [101,632 MACs],[0%],
    [SNN (Spiking FP32)], [Sparse SynOps], [8.4 Ticks], [15,021 SynOps],[85.2%],
  ),
  caption: [Computational cost comparison. The SNN significantly reduces operations by leveraging temporal early-exit sparsity.],
  kind: "table",
  supplement: [Table]
) <tbl:phase2_efficiency>

The S-curve profile confirms the *early-exit hypothesis*: >80% of correct decisions are locked in between ticks 5–15.
Mean time-to-decision: *8.4 ticks* out of 64 → *85.2% reduction* in operations (15,021 SyOPs vs 101,632 MACs).

== PHASE III: LOCAL SNN LEARNING -- RESULTS

Unsupervised STDP + Hard WTA + homeostasis: *44.3% top-1 accuracy*

#grid(columns: 2, gutter: 20pt,
  image("figures/phase3_02_receptive_after.png"),
  image("figures/phase3_04_confusion_matrix.png")
)

#v(1cm)

#align(center)[
#image("figures/phase3_03_baseline_fcn_weights.png", width:50%),
]

- Mean latency: *8.3 ticks* — comparable to supervised transfer
- SyOPs: *12,954* → *87.2% reduction* (marginally sparser than Phase II)

#box-text()[
Hidden neurons self-organized into holistic digit templates without any labels. Visually simple digits ('0', '1') cluster reliably; morphologically ambiguous pairs ('5'/'0', '4'/'9') cause the most confusion.
]


== DISCUSSION: KEY FINDINGS & LIMITATIONS



*Neuron Model Evaluation:*
- Classical LIF may be misaligned with TTFS — biological fidelity must be selectively adapted for engineering objectives
- Model C achieves temporal discriminability through purely linear mechanics, avoiding expensive exponential kernels

*Weight Transfer Viability:*
- 94.5% zero-shot accuracy proves ANN spatial hierarchies are recoverable from spike timing alone — hybrid pipelines are viable for edge deployment
- *Sparsity Paradox:* GPUs still execute boolean-masked zero multiplications; physical efficiency gains require native neuromorphic ASICs or FPGAs or smarter GPU implementations

*Unsupervised Learning:*
- STDP clusters geometric features without labels — the 44.3% result is not a failure of the learning rule but of the single-layer fully-connected WTA topology
- Holistic template matching breaks down on morphologically overlapping digit classes

== FUTURE WORK

- *Complex datasets:* Imagenet, N-MNIST, DVS Gesture sensors — output spike trains natively, need to handle TTFS contrast

- *Spiking CNN architectures:* Local receptive fields + local WTA → neurons learn edge and curve primitives rather than rigid full-digit templates, enabling translation invariance

- *Structural plasticity:* Prune consistently depressed synapses → block-sparse tensor formats → real memory bandwidth reductions on FPGAs and ASICs

- *Hardware deployment:* Port verified TTFS algorithms to Intel Loihi, IBM TrueNorth or custom accelerators to measure true joule-per-inference energy consumption

== CONCLUSION

#box-text()[
The brain runs on *~20 W*. Modern AI clusters run on *megawatts*. Closing this gap is a paradigmatic challenge.
]

This thesis demonstrated:

- *TTFS encoding* reduces theoretical compute by 85–87% vs. dense ANN inference via temporal early-exit sparsity
- *Standard LIF may be incompatible with TTFS* — Model C (current-accumulating linear ramp) is a suitable model if we need to decode rank order
- *Zero-shot weight transfer* recovers 94.5% accuracy with only a 3.9 pp temporal penalty — no retraining required
- *Unsupervised STDP* successfully extracts geometric features (44.3%) without labels or backpropagation


GPUs still perform masked zero multiplications: *realizing these efficiency gains requires native neuromorphic substrates* or smarter GPU implementations.

#align(horizon)[
#box-text()[
The majority of the teoretical savings come from the TTFS encoding and the challange is building systems
around this.
]]

#pagebreak()
#bibliography("references.bib")
