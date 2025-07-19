#import "@preview/lovelace:0.3.0": pseudocode-list
#import "@preview/wrap-it:0.1.1": wrap-content

// ================================================================================================
// CONFIG
// ================================================================================================

#set text(font: "Source Serif 4 18pt", size: 11pt)
#show math.equation : set text(font:"TeX Gyre Schola Math", size: 10.5pt)
#show raw : set text(font:"CommitMono", weight: "medium", size:9pt)
#set list(marker: sym.square.filled.small, indent: 1em)
#show heading: set text(font:"Source Serif 4",weight: "semibold", style:"italic")
#show heading.where( level: 1 ): it => block(width: 100%)[
  #set align(center); #set text(20pt); #v(1em); #upper(it) #v(1em)
]
#show heading.where( level: 2 ): it => block(width: 100%)[
  #set text(16pt); #v(1em); #upper(it) #v(0.5em)
]
#show heading.where( level: 3 ): it => block(width: 100%)[
  #set text(12pt); #v(1em); #upper(it) #v(0.3em)
]
#show heading.where( level: 4 ): it => block(width: 100%)[
  #set text(11pt); #v(1em); #upper(it) #v(0.5em)
]
#set heading(numbering: "1.1 ~ ")

// ================================================================================================
// FRONTPAGE
// ================================================================================================

#set page(fill: gradient.linear(rgb("003049"),rgb("B2FFA8"),angle:45deg), margin: (left: 2in))
#line(start: (0%, 5%), end: (8.5in, 5%), stroke: (thickness: 2pt))
#align(horizon + left)[
  #text(size: 30pt, weight: "semibold", style: "italic", font: "Source Serif 4" )[
    MACHINE LEARNING WITH SPIKES
  ]
  #v(1em)
  #text(font: "CommitMono", weight: "medium")[
    Brage Wiseth\ Universitiy of Oslo\
    #link("mailto:bragewi@uio.no")\
    #datetime.today().display()
  ]
]
#pagebreak()

// ================================================================================================
// ABSTRACT, ACKNOWLEDGEMENTS AND OUTLINE
// ================================================================================================

#set page(fill:none, margin:auto,numbering: "1")
#set par(justify: true)
#counter(page).update(1)

#heading(level: 1,outlined: false,numbering: none)[ACKNOWLEDGEMENTS]

#lorem(40)

#v(4em)
#heading(level: 1,outlined: false,numbering: none)[ABSTRACT]

#lorem(300)
#pagebreak()

#{
  // set text(font: "CommitMono")
  outline(depth:3, indent: auto)
}

// ================================================================================================
// BODY
// ================================================================================================

= Introduction And Theory

// TODO: needs citation
The concept of intelligence, how it arises and what needs to be in place for it to occur, is
probably been some of the longest standing questions in human history. How and if it can be
reproduced artificially is a particuarly hot topic today. Getting answers to these questions will
not only help us understand our own minds but also brings the promise of unlocking new technology
discovering new drugs or materials, it may be the last invention humans ever need to make. In recent
years we have crept ever closer to answer some of these questions. New state of the art artificial
inteligence systems have achieved remarkable success like the sophisticated language capabilities of
GPT models and the protein-folding predictions of AlphaFold.

Despite these triumphs, a significant gap persists between artificial systems and their biological
counterparts. Evidently, these AI systems might posses superhuman capabilities in one or a few
domains but none of them surpass humans in all, what we call Artificial General Inteligence (AGI).
Also more relevant to this thesis is that current state-of-the-art ANNs, require vast amount of
data, computatuon and energy resources. This demand stands in stark contrast to the biological
brain---an extraordinarily complex and efficient organ estimated to operate on merely 20-30 Watts
while also sitting comfortably in the AGI category. This profound difference in efficiency and
capability suggests that contemporary ANN paradigms, might be missing or oversimplifying fundamental
principles crucial for truly intelligent and scalable computation.

In this thesis we explore new approaches that first and foremost might solve the critical
limitations of scalability and energy efficiency in artificial intelligence. But also hopefully
lay the foundation for systems that might eventually unlock true AGI. This likely requires moving
beyond current mainstream ANN architectures. We will explore the potential of incorporating
more sophisticated biological principles into AI design. This involves investigating alternative
computational paradigms, inspired by mechanisms such as sparse, event-driven processing observed
in Spiking Neural Networks (SNNs), the role of temporal dynamics in neural coding, or the potential
computational advantages of systems operating near critical states. The central challenge lies
in identifying and abstracting the truly essential biological mechanisms for intelligence and
efficiency, distinguishing core principles from intricate biological details that may not be
necessary for artificial implementation. Concretly this thesis wants to
// TODO: add or remove research questions

  - Explore how information-flow based on sparse event might be implemented in a network, encoding
  - Explore learning algorithms suitable for such a network

In the succeeding sections we will lay down the theoretical foundations that we base our methods on

_Established Methods_. We will get familiar with the current AI methods

_Neuroscience 101_. Review neuroscience literature

_Method_. Outline the methods and how you can recreate the work 

_Results_. bla bla bla

_Discussions_. blabla bla future work bla bla

== Estasblished Methods

// TODO: add explanation and ilustrations
The term Aritifical Inteligence forms an umbrella over many different techniques that make use
of machines to do some intelligent task. The most promising way to acheive AI to day is trough
deep neural networks. The neural networks of today are almost exclusivly based on the simple
perceptron neuron model. It is a fairly old idea based on a simple model on how the brain processes
information. The model of the neuron that the is based on has synapses just like the biological
one, the synapses functions as inputs which when firing will exite the reciving neuron more or less
depending on the strenth of the connection. If the reciving neuron get exited above a threshold it
will fire and pass the signal downstream to another reciving neuron. Which is conceptually similar
to how real neurons operate. $ z = a(sum x w) $
#figure(caption:"perceptron")[#include("figures/perceptron.typ")]
This simple
model is called a perceptron, which introduced a learning rule for a single computational neuron
capable of classifying linearly separable patterns. However, to the MLP was the understanding that
stacking multiple layers of these perceptron-like units could overcome these limitations by creating
more complex decision boundaries. The critical breakthrough enabling the practical use of MLPs
was the independent development and subsequent popularization of the backpropagation algorithm.
Backpropagation provided an efficient method to calculate the gradient of the error function with
respect to the network's weights, allowing for effective training of these deeper, multi-layered
architectures. This combination---multiple layers of interconnected units
#footnote[
  While often conceptualized in layers (e.g., layers of the neocortex), the brain's connectivity
  is vastly more complex than typical feedforward ANNs, featuring extensive recurrent connections,
  feedback loops, and long-range projections that make a simple 'unrolling' into discrete layers
  an oversimplification
],
typically using non-linear activation functions, trained via backpropagation---defines the MLP,
which became a foundational architecture for neural networks and paved the way for the deep
learning revolution. GPT, alphafold, etc. all use these fundamentals with differetn variations of
architechtures which boils down to how many layers how large layers how dense layers and how they
should be connected (attention, RNN, CNN, resnet)

=== Problems With The Established Methods

We mentioned in the introduction that the deep learning technique is ineficient compared to the
brain. The reason why is not clear, from a hardware standpoint the brain simply has better hardware
much more connections per area and the computation is baked into the hardware. From an algorithmic
standpoint there may also be room for imporovement, In order to compute with deep learning and
perceptron networks we need to compute all the entries even tho they might not contribute or are
zero. Take an image for example, the human visual system is really good at ignoring unimportant
details and we only have a tiny area of focus. even then we dont porcess much unless something
interesting happens like movement. In deep learning we have to process the entire image. The status
quo needs global synchronization, every previous layer need to finish computing before the next
can start, this can be hard to scale for large systems where multiple proccesors need to talk to
eachother. The same applies to backpropagation it requires freezing the entire network and separates
computation and learning into two separate stages, local connectetions that should be independent of
eachother have to wait extreme quantization models (1bit) also highlight the ineficiency
// TODO: citations needed

== Neuroscience 101

// TODO: add relevant theory here that we refrence to later, do not add stuff that
// does not add important context nor future reference
Altough the perceptron captures common key aspects of biologial neuron models A lot is left on the
table. A lot of progress and new ideas has surfaced since the invention of the perceptron. The simple
neuron previously though to be simple like the perceptron model turns out to be more complex, the
information encoding is also a key research topic not explored by older models. How to brain learn
is also entirly different than what deep learning uses, changing the models and information encoding
forces us to rethink how the learning algorithms in the brain works. Network architechture, fully
asynchronus

=== Neuron Models

// ressonator neurons and integrator neurons ???
// Integrate-and-Fire Neurons (IF):  
// Output neurons accumulate spikes from their connected synapses within a short time window. If the
// accumulated input exceeds a threshold (e.g., 4/4 synapses fire), the neuron fires. This process
// ensures that only significant patterns propagate further. They need to reset after, either leaky
// or instant, also dependng on wether they fired or not
The neuron is the fundamental bulding block of the brain. Comprised of an axon synapses dendrites.
When presynaptic neurons fire the postsynaptic neuron increaes in potential if it reaches a
threshold it will itself fire. Neurons communicate with neurotransmitters such as dopmine and
glutamate. There are ion channels and some calsium idk.

=== Encoding

// rate encoding
// time to first spike
// neurons might be intensity to delay converters
// temporal coding
// Allow only the first n of m spikes to pass through ($N$ of $M$ encoding) Alternativly use Rank Order
// Coding or N of M Rank Order
// Alternativly use a dynamic $N$. This could be a thresold per region of an image use relative
// threshold so that dark spots still get their information trough
// For images, divide the input into spatial chunks and apply n-of-m coding within each chunk to
// preserve spatial information.
// For sequences like text, audio, or video, apply n-of-m coding to time frames. For video, combine
// n-of-m coding across both spatial chunks and temporal frames. Can be linked to brain waves.
It is observed that neurons fire in short bursts called spikes. Experiments show that neurons fire
repetably. A sequence of spikes is called a spike train, and exactly how information is encoded in a
spike train is a topic of hot debate in neuroscience. A popular idea is that information is encoded
in the average value of spikes per time called rate encoding

=== Learning

// The brain does not use backpropagation
// Use lateral inhibition
// Use Homeostatic Plasticity
// Use Synaptic Competition
// Grow Synapses: If a neuron is close to firing (e.g., 3/4 synapses activate), connect its final
// synapse to the most recent active input neuron. This mimics biological synapse growth
// Move Synapses: Adjust existing synapses toward frequently active input neurons to refine
// connections.
// Prune Synapses: Remove inactive synapses over time to maintain efficiency and sparsity.
_Spikes Do Not Play Nice With Gradients_. While models like Spiking Neural Networks (SNNs) offer
greater biological plausibility and potential advantages in processing temporal information and
energy efficiency, their adoption faces significant challenges, primarily stemming from the nature
of their core computational element: the discrete spike.

A cornerstone of the success of modern deep learning, particularly with Multi-Layer Perceptrons
(MLPs) and related architectures, is the backpropagation algorithm. Backpropagation relies
fundamentally on the network's components being differentiable; specifically, the activation
functions mapping a neuron's weighted input sum to its output must have a well-defined gradient.
This allows the chain rule of calculus to efficiently compute how small changes in network weights
affect the final output error, enabling effective gradient-based optimization (like Stochastic
Gradient Descent and its variants). These techniques have proven exceptionally powerful for training
deep networks on large datasets.

However, when we transition from the continuous-valued, rate-coded signals typical of MLPs to the
binary, event-based spikes used in SNNs, this differentiability is lost. The spiking mechanism
itself—where a neuron fires an all-or-none spike only when its internal state (e.g., membrane
potential) crosses a threshold—is inherently discontinuous. Mathematically, this firing decision is
often represented by a step function (like the Heaviside step function), whose derivative is zero
almost everywhere and undefined (or infinite) at the threshold.

Consequently, standard backpropagation cannot be directly applied to SNNs. Gradients calculated
using the chain rule become zero or undefined at the spiking neurons, preventing error signals
from flowing backward through the network to update the weights effectively. This incompatibility
represents a substantial obstacle, as it seemingly precludes the use of the highly successful and
well-understood gradient-based optimization toolkit that underpins much of modern AI.

Surrogate Gradients: A popular approach involves using a "surrogate"
function during the backward pass of training. While the forward pass uses the discontinuous spike
generation, the backward pass replaces the step function's derivative with a smooth, differentiable
approximation (e.g., a fast sigmoid or a clipped linear function). This allows backpropagation-like
algorithms (often termed "spatio-temporal backpropagation" or similar) to estimate gradients and
train deep SNNs, albeit with approximations.

=== Network

// Adress event representaion (AER)
// critical brain theory
// inhibitory connections (maybe more relevant in the neuron section)
// Binary weights (also maybe more relevant in the neuron section)
// only check if there is and outgoing/incomming spike or not

However, this abstraction, while powerful, significantly simplifies the underlying neurobiology.
Decades of rigorous neuroscience research reveal that brain function emerges from complex
electro-chemical and molecular dynamics far richer than the simple weighted sum and static
activation. While it's crucial to discern which biological details are fundamental to computation
versus those that are merely implementation specifics
#footnote[
  Disentangling core computational mechanisms from biological implementation details is a major
  ongoing challenge in neuroscience and neuromorphic engineering. Some complex molecular processes
  might be essential for learning or adaptation, while others might primarily serve metabolic or
  structural roles not directly involved in the instantaneous computation being modeled.
],
moving beyond the standard MLP model is necessary to capture more sophisticated aspects of neural
processing.

A primary departure lies in the nature of neural communication. Unlike the continuous-valued
activations typically passed between layers in an MLP (often interpreted as representing average
firing rates), biological neurons communicate primarily through discrete, stereotyped, all-or-none
electrical events known as action potentials, or 'spikes'. Information in the brain is encoded
not just in the rate of these spikes (rate coding), but critically also in their precise timing,
relative delays, and synchronous firing across populations (temporal coding) (Gerstner et al.,
2014). For instance, the relative timing of spikes arriving at a neuron can determine its response,
allowing the brain to process temporal patterns with high fidelity – a capability less naturally
captured by standard MLPs. Spikes can thus be seen as event-based signals carrying rich temporal
information.

Furthermore, neural systems exhibit complex dynamics beyond simple feedforward processing. Evidence
suggests that cortical networks may operate near a critical state, balanced at the 'edge of chaos,'
a regime potentially optimal for information transmission, storage capacity, and computational
power. Systems like the visual cortex demonstrate this complexity, where intricate patterns of
spatio-temporal spiking activity underlie feature detection, object recognition, and dynamic
processing. These biologically observed principles—event-based communication, temporal coding,
and complex network dynamics—motivate the exploration of Spiking Neural Networks (SNNs), which
explicitly model individual spike events and their timing, offering a potentially more powerful and
biologically plausible framework for computation than traditional MLPs.

= Methodology

#figure(caption:"spike train")[#image("figures/spiketrain.svg")]
// Step: explore learning algorithms
// experiment 1 set up random pattern across 1 dimension, inject a repeating pattern
// for the network to learn
// Step: explore edge detection
// can be achived by duplicating and inverting the signal
// Step: explore different neuronal models
// Step: prove mathematical equivalence to other ANNs Prove that the network satisties the universal
// approximation theorem
// Step: expand to 2d
// Step: expand network to more layers figure out how to connect (AER)
// Step: explore how learning algos work with multi-layer
// Step: try to learn MNIST
// NOTETOSELF: Steps either build on eachother or are independent, this way we can do
// bite size research and stop at any point if we run out of time, we dont have to
// make an entire system in order to have something to write about

== Neuron Models


== Learning


== Network

= Results

= Discussion

#bibliography("references.bib")
