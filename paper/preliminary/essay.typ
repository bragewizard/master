/* Configurations */

#set text(font: "Source Serif 4 18pt", size: 11pt)
#show math.equation : set text(font:"TeX Gyre Schola Math", size: 10.5pt)
#show heading: set text(font:"Source Serif 4",weight: "semibold", style: "italic")
#set heading(numbering: "1.1 ~ ")
#show heading.where(
  level: 1
): it => block(width: 100%)[
  #set align(center)
  #set text(20pt)
  #v(1em)
  #counter(heading).display() #smallcaps(it.body) 
  #v(1em)
]

#show heading.where(
  level: 2
): it => block(width: 100%)[
  #set text(16pt)
  #v(1em)
  #counter(heading).display() #smallcaps(it.body)
  #v(0.5em)
]

#set page(numbering: "1")
#set par(justify: true)


#block(width:100%)[#set text(size: 27pt, weight: "semibold", style: "normal", font: "Source Serif 4")
  #set align(center)
  #smallcaps[Machine Learning With Spiking Neural Networks]
  #v(1em)
]

#align(center)[ Brage Wiseth \ Universitiy of Oslo \ #link("mailto:bragewi@uio.no")]

#align(center)[#datetime.today().display()]

#v(3em)
= Introduction

The quest to create intelligent machines represents a long-standing ambition, one that has gained
significant momentum in recent decades with the advent of Artificial Neural Networks (ANNs).
Drawing high-level inspiration from the computational principles of the mammalian brain, these
models, particularly deep learning architectures like Multilayer Perceptrons (MLPs), have achieved
remarkable success. They underpin many transformative technologies, exemplified by breakthroughs
like the sophisticated language capabilities of GPT models and the protein-folding predictions of
AlphaFold.

Despite these triumphs, a significant gap persists between artificial systems and their biological
counterparts. Current state-of-the-art ANNs, while functionally powerful, require vast computational
resources and energy for both training and operation. This demand stands in stark contrast
to the biological brain---an extraordinarily complex and efficient organ estimated to operate
on merely 20-30 Watts while performing tasks far beyond the capabilities of current AI. This
profound difference in efficiency and capability suggests that contemporary ANN paradigms, often
characterized by dense matrix multiplications and trained via backpropagation, might be missing or
oversimplifying fundamental principles crucial for truly intelligent and scalable computation.

While one might hypothesize that further progress simply requires more computational power and
incremental architectural refinements, the energy and resource costs associated with scaling current
models pose significant practical limitations. This necessitates a re-evaluation of our approach. If
the goal remains to create machines with brain-like capabilities and efficiency, it may be essential
to draw deeper and more nuanced inspiration from neuroscience.

This essay argues that overcoming the critical limitations of scalability and energy efficiency in
artificial intelligence likely requires moving beyond current mainstream ANN architectures. It will
explore the potential of incorporating more sophisticated biological principles into AI design.
This involves investigating alternative computational paradigms, potentially inspired by mechanisms
such as sparse, event-driven processing observed in Spiking Neural Networks (SNNs), the role of
temporal dynamics in neural coding, or the potential computational advantages of systems operating
near critical states. The central challenge lies in identifying and abstracting the truly essential
biological mechanisms for intelligence and efficiency, distinguishing core principles from intricate
biological details that may not be necessary for artificial implementation.


= The Current Level of Brain Inspiration

When we talk about AI today almost all models use some variation of the Multi Layer Perceptron (MLP)
concept. It is a fairly old idea based on a simple model on how the brain processes information.
The MLP evolved from early attempts to create computational models inspired
by biological neurons. Its roots lie in the foundational work of McCulloch and Pitts (1943), who
proposed a simplified binary threshold model of a neuron, and Frank Rosenblatt's Perceptron (late
1950s), which introduced a learning rule for a single computational neuron capable of classifying
linearly separable patterns. However, progress stalled significantly after Minsky and Papert's
1969 book Perceptrons, which rigorously demonstrated the limitations of these single-layer models,
famously highlighting their inability to solve non-linearly separable problems like the XOR
function. The key insight leading to the MLP was the understanding that stacking multiple layers
of these perceptron-like units could overcome these limitations by creating more complex decision
boundaries. The critical breakthrough enabling the practical use of MLPs was the independent
development and subsequent popularization of the backpropagation algorithm in the 1970s and 1980s
(with key work by Werbos, Parker, LeCun, and notably Rumelhart, Hinton, and Williams in 1986).
Backpropagation provided an efficient method to calculate the gradient of the error function with
respect to the network's weights, allowing for effective training of these deeper, multi-layered
architectures. This combination---multiple layers of interconnected units, typically using non-linear
activation functions, trained via backpropagation---defines the MLP, which became a foundational
architecture for neural networks and paved the way for the deep learning revolution.




= More advanced brain models

The perceptron, and its evolution into Multi-Layer Perceptrons (MLPs), represent foundational
models in artificial intelligence inspired by early concepts of neural computation. Indeed, certain
core principles resonate with biological observations: the brain comprises interconnected neurons,
often organized in broadly hierarchical structures or layers#footnote[
While often conceptualized in layers (e.g., layers of the neocortex), the brain's connectivity
is vastly more complex than typical feedforward ANNs, featuring extensive recurrent connections,
feedback loops, and long-range projections that make a simple 'unrolling' into discrete layers an
oversimplification (Felleman & Van Essen, 1991).]
that process information sequentially
from sensory input to higher cognitive areas. Furthermore, individual neurons integrate incoming
signals—analogous to a weighted sum in MLPs—and generate an output spike or 'fire' only when a
certain threshold is exceeded, a mechanism abstracted by the activation functions used in artificial
neurons (McCulloch & Pitts, 1943).

However, this abstraction, while powerful, significantly simplifies the underlying neurobiology.
Decades of rigorous neuroscience research reveal that brain function emerges from complex
electro-chemical and molecular dynamics far richer than the simple weighted sum and static
activation. While it's crucial to discern which biological details are fundamental to computation
versus those that are merely implementation specifics#footnote[
Disentangling core computational mechanisms from biological implementation details is a major
ongoing challenge in neuroscience and neuromorphic engineering. Some complex molecular processes
might be essential for learning or adaptation, while others might primarily serve metabolic or
structural roles not directly involved in the instantaneous computation being modeled.]
, moving beyond the standard MLP model is
necessary to capture more sophisticated aspects of neural processing.

A primary departure lies in the nature of neural communication. Unlike the continuous-valued
activations typically passed between layers in an MLP (often interpreted as representing average
firing rates), biological neurons communicate primarily through discrete, stereotyped, all-or-none
electrical events known as action potentials, or 'spikes' (Hodgkin & Huxley, 1952). Information
in the brain is encoded not just in the rate of these spikes (rate coding), but critically also
in their precise timing, relative delays, and synchronous firing across populations (temporal
coding) (Gerstner et al., 2014). For instance, the relative timing of spikes arriving at a neuron
can determine its response, allowing the brain to process temporal patterns with high fidelity – a
capability less naturally captured by standard MLPs. Spikes can thus be seen as event-based signals
carrying rich temporal information.

Furthermore, neural systems exhibit complex dynamics beyond simple feedforward processing. Evidence
suggests that cortical networks may operate near a critical state, balanced at the 'edge of chaos,'
a regime potentially optimal for information transmission, storage capacity, and computational
power (Beggs & Plenz, 2003; Chialvo, 2010). Systems like the visual cortex demonstrate this
complexity, where intricate patterns of spatio-temporal spiking activity underlie feature detection,
object recognition, and dynamic processing (Hubel & Wiesel, 1962; Thorpe et al., 1996). These
biologically observed principles—event-based communication, temporal coding, and complex network
dynamics—motivate the exploration of Spiking Neural Networks (SNNs), which explicitly model
individual spike events and their timing, offering a potentially more powerful and biologically
plausible framework for computation than traditional MLPs.




= Challenges in Training Advanced Neural Models: The Problem of Discontinuity

While models like Spiking Neural Networks (SNNs) offer greater biological plausibility and
potential advantages in processing temporal information and energy efficiency, their adoption faces
significant challenges, primarily stemming from the nature of their core computational element: the
discrete spike.

A cornerstone of the success of modern deep learning, particularly with Multi-Layer Perceptrons
(MLPs) and related architectures, is the backpropagation algorithm (Rumelhart et al., 1986).
Backpropagation relies fundamentally on the network's components being differentiable;
specifically, the activation functions mapping a neuron's weighted input sum to its output must
have a well-defined gradient. This allows the chain rule of calculus to efficiently compute how
small changes in network weights affect the final output error, enabling effective gradient-based
optimization (like Stochastic Gradient Descent and its variants). These techniques have proven
exceptionally powerful for training deep networks on large datasets.

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

This challenge has spurred significant research into alternative training methodologies for SNNs:

Surrogate Gradients: A popular approach involves using a "surrogate" function during the
backward pass of training. While the forward pass uses the discontinuous spike generation, the
backward pass replaces the step function's derivative with a smooth, differentiable approximation
(e.g., a fast sigmoid or a clipped linear function) (Neftci et al., 2019; Zenke & Ganguli, 2018).
This allows backpropagation-like algorithms (often termed "spatio-temporal backpropagation" or
similar) to estimate gradients and train deep SNNs, albeit with approximations.

Bio-Inspired Local Learning Rules: Drawing inspiration from neuroscience, researchers explore
learning rules based on local activity, such as Spike-Timing-Dependent Plasticity (STDP). STDP
adjusts synaptic weights based on the relative timing of pre- and post-synaptic spikes (Gerstner
et al., 1996; Bi & Poo, 1998). While biologically plausible and inherently suited to spike timing,
purely local rules like STDP often struggle to match the performance of gradient-based methods on
complex supervised learning tasks and can be harder to scale or direct towards a specific global
objective. Hybrid approaches combining STDP with other mechanisms are also being investigated.

Conversion Methods: Another strategy involves training a conventional ANN (like an MLP or CNN)
using standard backpropagation and then converting the trained network into an SNN (Cao et al.,
2015; Diehl et al., 2015). This leverages the power of gradient-based training but may not fully
exploit the unique temporal dynamics SNNs offer, and often requires careful parameter tuning during
conversion.

Gradient-Free Optimization: Techniques like evolutionary algorithms or reinforcement learning
can optimize SNN parameters without requiring explicit gradients, but they often suffer from lower
sample efficiency and scalability issues compared to gradient descent, particularly for very large
networks.

Therefore, while moving towards more biologically realistic, event-driven models like SNNs is
conceptually appealing, overcoming the fundamental incompatibility with standard gradient-based
optimization remains a critical area of active research and development. The success of SNNs in
practice hinges significantly on the effectiveness and scalability of these alternative or adapted
training techniques.




#bibliography("citations.bib")
