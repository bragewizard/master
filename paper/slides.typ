#import "@preview/touying:0.7.3": *
#import themes.simple: *
#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge

#show: simple-theme.with(
  aspect-ratio: "16-9",
  footer: [Neuromorphics],
  primary: black,
)

#set text(font: "Geist", size: 16pt, weight: "medium", top-edge:.7em)
#show math.equation : set text(font:"Latin Modern Math", size: 18pt, weight: "medium")
#show raw : set text(font:"GeistMono NF", weight: "medium", size:14pt)
#set list(marker: sym.bullet, indent: 1em)
#show heading: set text(font:"Geist",weight: "bold", style:"normal")
#show heading.where( level: 1 ): it => block(width: 100%)[ #set text(28pt); #upper(it) ]
#show heading.where( level: 2 ): it => block(width: 100%)[ #set text(18pt); #upper(it) ]
#show heading.where( level: 3 ): it => block(width: 100%)[ #set text(14pt); #upper(it) ]
#set par(justify: true)

#let box-text(body) = {
block(stroke:(thickness:0pt, paint:luma(0)), inset: 10pt, radius: 0pt, fill: luma(220),
  width: 100%)[#body]
}

#title-slide[
  = ON NEUROMORPHIC COMPUTING\
    WITH SPIKING NEURAL NETWORKS
  #v(1em)
  08.06.2026
]

= INTRODUCTION AND MOTIVATION

==
- *The von Neumann Bottleneck:* The physical separation of processing and memory forces massive energy expenditure on data transport[cite: 1].
- Fetching weights from DRAM consumes orders of magnitude more energy than the computation itself[cite: 1].
- *Dense Processing of Sparse Data:* Deep learning hardware executes dense matrix multiplications regardless of activation sparsity, wasting cycles on null results[cite: 1].
- *The Cost of Synchrony:* Global clock networks consume up to 40% of power budgets and enforce worst-case latencies[cite: 1].
- *Biological Contrast:* The brain operates on ~20W, performing real-time multi-sensory reasoning through asynchronous, sparse events[cite: 1].

#box-text()[ lets draw more inspiration from biological inteligence -> neuromorphic computing ]

= WHY IS NEUROMORPHIC MORE EFFECIENT?

== SHIFTING THE COMPUTING PARADIGM
#box-text()[ Neuromorphic engineering replaces rigid, clock-driven logic with the adaptive, event-driven dynamics of neural tissue[cite: 1]. ]
- *The Synaptic Principle:* Eliminates the von Neumann bottleneck by physically co-locating memory and compute across the silicon die[cite: 1].
- *The Action Potential Principle:* Operates asynchronously, driven strictly by the arrival of data, ensuring energy scales linearly with task complexity[cite: 1].
- *The Spike Principle:* Communicates via discrete binary events, drastically reducing bandwidth by encoding information in precise timing rather than complex magnitudes[cite: 1].

= Research Objectives

== Core Questions Addressed in this Thesis
- *Sparse Efficient Computing:* Do biologically inspired, event-driven algorithms reduce the computational footprint of visual classification?[cite: 1]
- *Neuron Model Evaluation:* Which temporal integration dynamics are compatible with Time-to-First-Spike (TTFS) decoding?[cite: 1]
- *Inference Via Weight Transfer:* What is the accuracy penalty of zero-shot ANN-to-SNN weight transfer under TTFS?[cite: 1]
- *Native Unsupervised Learning:* Can local Spike-Timing-Dependent Plasticity (STDP) autonomously extract meaningful geometric features?[cite: 1]

= ENCODING INFORMATION IN SPIKES

#align(center + horizon)[
#include "figures/spiketrain.typ"
]

#align(center + horizon)[
#include "figures/rateencoding.typ"
]

== Time-to-First-Spike (TTFS) Encoding
- Unlike Rate Coding, which requires extended time windows, TTFS encodes stimulus intensity inversely to response latency[cite: 1].
- *Mechanism:* A high-intensity (bright) pixel triggers an early spike, compressing spatial information into a priority-driven queue[cite: 1].
- *Advantages:*
  - *Sparsity:* Sub-threshold noise (background pixels) is aggressively discarded and never fires[cite: 1].
  - *Latency:* Eliminates the need to wait for a time window to close; processing begins as soon as salient features arrive[cite: 1].
- Phase ambiguity is resolved by simulating a biological saccade, establishing a global temporal reference frame ($t_0$)[cite: 1].

#align(center + horizon)[
#include "figures/temporalcoding.typ"
]

= Phase I: Neuron Model Dynamics

== Decoding Temporal Sequences
- Evaluated four architectures (IF, LIF, Linear Ramp, State Discount) across saturation, critical, and deficit threshold regimes[cite: 1].
- *The LIF Misalignment:* Under critical constraints, the standard Leaky Integrate-and-Fire model actively penalized early, high-salience spikes due to its exponential decay[cite: 1].
- *The Solution (Model C):* Developed a Current-Accumulating Linear Ramp model combining integration momentum with a strict 10-tick coincidence window[cite: 1].
- Model C successfully preserved rank-order sequence priority without the computational overhead of continuous exponentials[cite: 1].

= Methodology & Setup

== Experimental Framework
- *Dataset:* MNIST (28x28 normalized grayscale images), chosen for its high degree of spatial sparsity[cite: 1].
- *Topology:* Fully Connected Network (FCN) $784 arrow.r 128 arrow.r 10$[cite: 1].
  - An FCN allows mathematically transparent, one-to-one parameter mapping without the structural overhead of convolutional unrolling[cite: 1].
- *SNN Simulator:* Custom PyTorch discrete-time engine operating over a $T_{max} = 64$ tick saccade window[cite: 1].
- *Metrics:* Top-1 Accuracy, Temporal Latency (Time-to-Decision), and Synaptic Operations (SyOPs) as a hardware proxy[cite: 1].

= Phase II: Zero-Shot Weight Transfer

== Translating ANN to SNN
- A baseline ANN was trained without bias terms to 98.40% accuracy, establishing an ideal ceiling[cite: 1].
- Continuous FP32 weights were transferred *directly* to the SNN (Model C) without any intermediate retraining or fine-tuning[cite: 1].
- *Results:*
  - *Accuracy:* Maintained 94.50%, proving momentum-based TTFS successfully preserves spatial hierarchies[cite: 1].
  - *Latency:* Reached decisions rapidly, with a mean latency of 8.4 ticks[cite: 1].
  - *Efficiency:* Achieved an 85.2% reduction in operations (SyOPs) due to the early-exit S-curve profile[cite: 1].

= Phase III: Native Unsupervised Learning

== Local STDP & Self-Organization
- Discarded pre-trained weights; initialized randomly and trained strictly via a discrete, vectorized STDP rule[cite: 1].
- *Competitive Specialization:*
  - *Winner-Takes-All (WTA):* The first output neuron to fire suppresses all competitors, forcing distinct geometric clustering[cite: 1].
  - *Homeostasis:* Implemented adaptive threshold decay and winner penalties to prevent single-neuron dominance (dead neurons)[cite: 1].

= Phase III: Results & Limitations

== Representational Boundaries
- *Accuracy:* Achieved 44.3% top-1 accuracy via unsupervised self-organization[cite: 1].
- Successfully clustered geometric features without global error signals, maintaining the early-exit sparsity (87.2% compute reduction)[cite: 1].
- *The Limitation:* Single-layer WTA forces rigid, holistic template matching[cite: 1].
- The network struggled with morphological overlaps (e.g., '5' vs. '0') because the global WTA discards complementary spatial evidence in favor of the single most salient feature[cite: 1].

= Discussion

== The Sparsity Paradox
#box-text()[ While theoretical SynOps were reduced by $>85\%$, simulating this on von Neumann hardware highlights the central engineering bottleneck. ]
- Deep learning frameworks and GPUs are optimized for dense matrix multiplications and will execute floating-point cycles even for boolean-masked zeros[cite: 1].
- Combined with temporal loop overhead, the SNN simulator inherently consumes more absolute clock cycles on a GPU than the continuous ANN baseline[cite: 1].
- *Conclusion:* Realizing calculated energy gains requires deployment on native event-driven substrates (e.g., neuromorphic ASICs or FPGAs) that utilize Address Event Representation (AER) to physically halt processing[cite: 1].

= Future Work

== Bridging the Engineering Gap
- *Native Sensors:* Transitioning to Dynamic Vision Sensors (DVS) to encode contrast changes directly, bypassing the latency bottlenecks of processing dark pixels under TTFS[cite: 1].
- *Spiking CNNs:* Replacing global WTA with localized receptive fields and local lateral inhibition to build deep, translation-invariant hierarchies[cite: 1].
- *Structural Plasticity:* Pruning depressed synapses to create block-sparse tensors, yielding physical memory bandwidth reductions even prior to ASIC deployment[cite: 1].

= Conclusion
- The standard deep learning scaling paradigm is approaching strict physical limits[cite: 1].
- This work demonstrates that TTFS temporal encoding drastically reduces computational workloads via early-exit, momentum-based integration[cite: 1].
- While local STDP is viable for unsupervised feature extraction, unlocking the true potential of these algorithms requires bridging the gap between event-driven software and specialized physical hardware[cite: 1].
