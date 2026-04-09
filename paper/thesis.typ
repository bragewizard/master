#import "@preview/droplet:0.3.1": dropcap
#import "@preview/wordometer:0.1.5": word-count, total-words
#import "@preview/lovelace:0.3.0": pseudocode-list
#import "@preview/glossarium:0.5.9": make-glossary, register-glossary, print-glossary, gls, glspl
#import "frontpage/frontpage.typ": cover, colors
#import "glossary.typ": entry-list
#import "style.typ": style, serif-text, mono-text, box-text, mini-header

#show: style
#show: make-glossary
#show: word-count

// *TODO*

// - [ ] proof read, stavefeil, flow, struktur, ordlegging akademisk tone. formler, gjenta til fornøyd
// - [ ] figurer
// - [ ] referanser
// - [ ] metode og resultater handcrafted/kopierte vekter
// - [ ] metode og resultater neuromorfisk læring
// - [ ] discusion
// - [ ] conclusion
// - [ ] abstract
// - [ ] siste proof read

#v(.5em)
#text(size: 9pt, weight: "medium")[
#h(1fr) Wordcount = #total-words / 25000
]

// FRONTPAGE
#cover()

#pagebreak()

// ABSTRACT, ACKNOWLEDGEMENTS AND OUTLINE
#counter(page).update(1)

#v(1.2cm)
#align(center)[
#block(width:90%, inset: 2em)[
#align(left)[
  #text(weight:"semibold",size:16pt,[ABSTRACT])

  #serif-text()[ The development of modern Deep Learning has achieved unprecedented performance across various domains, yet it remains fundamentally bottlenecked by the energy and memory inefficiencies of the von Neumann architecture. To address these limitations, this thesis investigates Neuromorphic Computing with Spiking Neural Networks (SNNs) as a biologically plausible, highly energy-efficient alternative. By shifting from synchronous, continuous-value matrix multiplications to asynchronous, event-driven sparse computations, neuromorphic systems emulate the physical principles of the biological brain.

This work explores the implementation of these principles on standard CPU/GPU hardware. Two primary methodologies are developed and evaluated on visual classification tasks: (1) an inference-optimized SNN that translates weights from a conventionally trained Fully Connected Network (FCN) using Time-To-First-Spike (TTFS) temporal encoding, and (2) an unsupervised, biologically inspired learning simulator incorporating structural plasticity (dynamic synaptogenesis and pruning). The results demonstrate the viability of temporal coding and local learning rules in extracting meaningful features from visual stimuli, highlighting the potential of neuromorphic algorithms to drastically reduce the computational footprint of artificial intelligence.]]
]]

#pagebreak()

#align(center)[
#block(width:100%)[
  #align(left)[
  #text(weight:"semibold",size:16pt,[ACKNOWLEDGEMENTS & DECLARATIONS])

  #serif-text()[
I would like to thank my supervisor and the very kind and helpful comunity at ROBIN and NANO labratories at the departemnt of informatics. I would like to that professor Farad at SDU

#v(1em)
#mini-header()[Declaration of the use of generative artificial intelligence]

In this scientific work, generative artificial intelligence (AI) has been used. All data and personal information have been processed in accordance with the University of Oslo's regulations, and I, as the author of the document, take full responsibility for its content, claims, and references. An overview of the use of generative AI is provided below.

The service GPT UiO, developed by UiO IT department, has been used to improve the content of the report/assignment. A first version of the work was pasted in its entirety, and the model was given the prompt [rewrite this text to make the language more lively.] The text was then iterated a few times through the model where new prompts were used to get the correct structure of the text. The final result was cut out, fact-checked, and partly rewritten by the author(s).
]]
]]

#pagebreak()

#{
  set text(font: "Geist", weight: "medium", size: 10pt)
  outline(depth:3, indent: auto)
}


#pagebreak()

#heading(outlined: false, numbering: none, level: 1)[GLOSSARY]
#{
set text(font: "Geist", weight: "medium", size: 10pt)
register-glossary(entry-list)
print-glossary(
 entry-list
)
}

#pagebreak()

= Introduction <intro>

#serif-text()[ The development of intelligent machines is a significant objective in modern science and engineering. While the concept has historical roots in philosophy and early automata, the field has transitioned from speculative theory to practical application. Currently, artificial intelligence is central to technological and economic development. Understanding the mechanisms of intelligence and reproducing them in synthetic systems offers the potential for improved analysis of biological minds and the creation of tools for applications ranging from personalized medicine to automated scientific discovery.

In recent years, great strides has been made towards this goal. Deep Learning, which utilizes multilayered neural networks, has exceeded previous performance benchmarks. These systems have demonstrated high proficiency in tasks previously limited to human capability. For example, AlphaFold has addressed complex problems in protein folding @Placeholder, reinforcement learning agents have mastered the complexity of games such as Go @Placeholder, and Large Language Models have shown capabilities in text generation that approach human fluency. Consequently, @ai is increasingly viewed as a general-purpose technology that may influence societal infrastructure.

However, despite these advances, there are significant limitations to the current approach. The success of modern deep learning relies heavily on scaling, which involves increasing data volume and computational power. This strategy is approaching physical and economic boundaries. Training state-of-the-art models consumes substantial energy and results in a large carbon footprint @Placeholder. Although specialized hardware allows for more efficient computations, the underlying architecture and algorithms imposes an intrinsic limit on scalability independent of the underlying hardware. Furthermore, the requirement for massive datasets presents challenges in sourcing and curation. Additionally, evidence suggests that this scaling approach yields diminishing returns. Models often function as statistical correlation engines; they lack common-sense reasoning, struggle with out-of-distribution generalization, and are prone to brittle failure modes @Placeholder.

These limitations are evident when comparing artificial systems to biological intelligence. The human brain demonstrates that high-level intelligence is possible without massive energy consumption or dataset sizes. The brain operates on approximately 20 watts @Placeholder. With this limited energy budget, it manages biological functions, processes real-time multi-sensory data, and performs abstract reasoning. In contrast, deep learning models require @gpu clusters with significantly higher power requirements to match a fraction of these capabilities. There is also a discrepancy in learning efficiency. Deep learning models are sample-inefficient, often requiring vast numbers of examples to learn a representation. Biological systems, however, are capable of "one-shot" or "few-shot" learning and can acquire new information without catastrophic forgetting. This suggests the inefficiency of current @ai is a paradigmatic issue rather than just an engineering problem.

The proposed direction for addressing these issues involves biological inspiration in both hardware and algorithm design, specifically Neuromorphic Computing. This field attempts to engineer computer architectures that mimic the biological structure of the nervous system. Unlike traditional @ai, which runs as software on general-purpose hardware, neuromorphic engineering aims to align the algorithm with the physical substrate. It moves away from clock-driven processing toward asynchronous, event-driven systems. In this paradigm, information is encoded as sparse, discrete events or "spikes." Similar to biological neurons, a neuromorphic processor consumes minimal energy when inactive, processing information only when triggered. This approach is being pursued by both industrial groups, such as Intel’s Loihi @Placeholder, and academic projects like SpiNNaker @Placeholder. These systems represent a shift from calculation-based machines to those capable of real-time adaptation.

Although neuromorphic systems achieve optimal performance on co-designed platforms---where the algorithm is embedded directly into the hardware---there is significant value in executing neuromorphic algorithms on traditional von Neumann architectures. In this thesis, we explore biologically inspired algorithms deployed on traditional @cpu and @gpu hardware. We examine how event-driven, biologically plausible computation can address limitations in scalability, data efficiency, and energy consumption, even when simulated on standard processors. We present approaches for efficient information coding and learning algorithms inspired by neural mechanisms. Concretely, this thesis aims to: ]

#box-text()[
- *Investigate Sparse Information Flow*: Explore how information can be encoded and processed using sparse, asynchronous events (spikes) within a neural network.

- *Develop Biologically Inspired Learning*: Explore and evaluate learning algorithms that are suitable for such networks, adhering to the constraints of locality and efficiency.
]

#serif-text()[ The succeeding chapter lays the historical and theoretical foundation, covering early neuroscience and the development of artificial neural networks based on simple models of the brain. Following this, we review relevant modern neuroscience literature, extracting key concepts that will inform the methodology. We also provide consise overview on machine learning concepts and frameworks. Finally, we detail the implementation of these principles in a neuromorphic context and evaluate their performance against standard benchmarks. ]

#pagebreak()

// = Background <background>

// #serif-text()[ This section outlines the historical and theoretical evolution of @ai, reviewing key concepts in modern neuroscience that motivate the methodology used in this thesis.

// We begin at a shared origin point, a time when @ai research and neuroscience were intertwined. We then trace the diverging path that led to modern Deep Learning, examining why it has drifted from biological plausibility. Subsequently, we explore the "neuromorphic path". In @biologicalprinciples, we detail the specific physical principles and neuroscientific insights upon which the neuromorphic methods in this thesis is built. In @mltechnicalities, we contrast the architectural mechanics of deep learning and neuromorphic systems, specifically addressing why the former is computationally powerful yet energetically inefficient. We conclude with a review of existing frameworks, identifying their strengths and weaknesses to contextualize the contributions of this work. ]

= History & Developments <historyanddevelopments>

#serif-text()[ Historically, the understanding of neural tissue was dominated by the reticular theory, which claimed that the brain consisted of a continuous, fused network of nerve fibers. This paradigm was fundamentally challenged by the work of Santiago Ramón y Cajal. Through the application of novel staining techniques, Cajal established the neuron doctrine, demonstrating that the nervous system is composed of discrete, individual cells @Placeholder. Building on these findings, Heinrich Wilhelm Gottfried Von Waldeyer-Hartz proposed the "Neuron Doctrine" and coined the term "neurons" to describe these dicrete cells. Subsequent analysis using electron microscopy has provided irrefutable validation of this discrete cellular structure.

The conceptualization of the brain as a collection of discrete units facilitated the development of mathematical models describing neural function. In 1943, Warren McCulloch and Walter Pitts published A Logical Calculus of the Ideas Immanent in Nervous Activity, introducing the first formal model of the neuron.

The McCulloch-Pitts (M-P) neuron abstracted biological complexity into a binary decision device governed by the following logic: ]

#box-text()[
- *Inputs*: The neuron receives multiple binary inputs, weighted as either excitatory or inhibitory.
- *Summation*: The unit calculates the linear sum of these weighted inputs.
- *Thresholding*: If the aggregate sum exceeds a fixed threshold, the neuron outputs a 1 (firing); otherwise, it outputs a 0 (silence).
]

#serif-text()[ McCulloch and Pitts demonstrated that networks of these units could theoretically compute any logical operation (AND, OR, NOT) @Placeholder. This abstraction established the foundational link between biological processes and digital logic, suggesting that neural function could be replicated in electronic hardware. Consequently, the M-P neuron serves as the common ancestor for both computational neuroscience and artificial intelligence.

However, despite its theoretical utility, the original M-P model presented significant functional limitations. The connectivity was static, requiring circuits to be manually designed rather than learned. Furthermore, the restriction to binary weights precluded the modeling of graded signal intensity, preventing the system from capturing the nuance of real-world sensory input.

In 1949, Donald Hebb addressed the critical issue of plasticity in his work The Organization of Behavior. He proposed a theoretical mechanism for synaptic modification, now known as Hebbian learning, which provided a biological basis for how neural networks could adapt over time. Hebb postulated: ]

#box-text()[ "Let us assume that the persistence or repetition of a reverberatory activity (or "trace") tends to induce lasting cellular changes that add to its stability. ... When an axon of cell A is near enough to excite a cell B and repeatedly or persistently takes part in firing it, some growth process or metabolic change takes place in one or both cells such that A’s efficiency, as one of the cells firing B, is increased" @Placeholder. ]

#serif-text()[ This principle is colloquially summarized as "neurons that fire together, wire together" @Placeholder. Crucially, this describes a local and decentralized learning rule; a synapse does not require a global error signal or external supervision to adjust. It requires only the correlation between the pre-synaptic input and the post-synaptic output. The convergence of the M-P architectural model and the Hebbian plasticity framework established the prerequisite conditions for the development of modern neural networks. ]

#v(2em)
== The Perceptron

#serif-text()[ In 1957, Frank Rosenblatt advanced these theoretical concepts by engineering the Perceptron. The "Mark I Perceptron" was a hardware implementation of the neural model, distinguished by a crucial innovation: a weight-adjustment mechanism based on Hebbian principles. Rosenblatt introduced the perceptron learning rule, an iterative algorithm capable of minimizing error automatically. The system processed an input pattern (e.g., a pixelated character) and produced a binary classification. When the output deviated from the target, the algorithm adjusted the weights proportional to the error: strengthening connections that should have contributed to a correct firing and weakening those that led to false positives. ]

#figure(include("figures/perceptron.typ"),caption:[The perceptron model. Inputs $x_i$ are multiplied by weights $w_i$ and summed. If the linear combination $sum x_i w_i$ exceeds the bias $b$, the neuron activates. ])

#serif-text()[ Consequently, the Perceptron was capable of converging on a solution for any problem where the data was linearly separable. This success generated significant enthusiasm, with contemporary reports suggesting that such machines would soon mimic human consciousness @Placeholder.

These expectations were abruptly tempered by theoretical limitations. In 1969, Marvin Minsky and Seymour Papert published Perceptrons, a rigorous mathematical analysis of the architecture. They demonstrated that a single-layer perceptron is fundamentally a linear classifier. While capable of learning operations like AND or OR, it is mathematically incapable of solving the XOR (Exclusive OR) problem. In the XOR case, the classes cannot be separated by a single hyperplane. This proof highlighted a severe boundary on the utility of single-layer networks for complex, non-linear tasks. ]

#figure(include("figures/gates.typ"),caption:[The XOR problem. Unlike AND/OR, the data points for XOR cannot be separated by a single linear boundary.])

#serif-text()[ The publication of Perceptrons coincided with a significant reduction in neural network research funding, a period retrospectively termed the "First AI Winter". It is worth noting that Minsky and Papert acknowledged that a @mlp, a network stacking multiple layers of neurons, could theoretically solve the XOR problem by creating complex, non-linear decision boundaries.

However, a critical algorithmic gap remained: the "credit assignment problem". While researchers knew that hidden layers could represent complex features, there was no known method to propagate error signals back through the layers to adjust the weights of hidden neurons effectively. Rosenblatt’s rule was mathematically valid only for the output layer. The field remained stagnant until a method for training multi-layer networks could be formalized. ]

#v(2em)
== Deep Learning

#serif-text()[ The critique presented by Minsky and Papert precipitated a contraction in funding; despite this, theoretical inquiry persisted. It was widely hypothesized that the limitations of the single perceptron could be overcome by a @mlp. By organizing neurons (single perceptrons) into hierarchical layers, the network could theoretically perform successive non-linear transformations on the input space, enabling the formation of complex decision boundaries. The primary impediment was not the architecture itself, but the absence of a viable learning algorithm.

In a single-layer perceptron, error attribution is immediate: if the output deviates from the target, the error is directly derived from the weights of the output layer. However, in a multi-layer architecture, quantifying the contribution of a specific neuron within the "hidden" layers to the final output error presents a significant challenge. This is formally known as the Credit Assignment Problem @Placeholder, and it remained the central theoretical obstacle for over a decade. ]

#figure(include("figures/network.typ"),caption:[A @mlp. By inserting "hidden layers" between input and output, the network can approximate non-linear functions such as XOR. The historical challenge lay in deriving a method to train these intermediate layers.])

#serif-text()[ The solution to this theoretical impasse was popularized in 1986 by Rumelhart, Hinton, and Williams in their seminal paper _Learning representations by back-propagating errors_ @Placeholder. They demonstrated that the Chain Rule of calculus could be applied recursively to propagate the error signal from the output layer backwards through the hidden layers. This algorithm, known as Backpropagation, allowed the network to calculate the gradient of the loss function with respect to every weight in the system. Effectively, it provided a mathematical method to tell each hidden neuron exactly how much it contributed to the total error, finally solving the credit assignment problem.

Unlike Hebbian plasticity, which is local and biological, Backpropagation relies on global error signals and precise backward data flow—mechanisms effectively absent in organic tissue. Consequently, the field of @ann effectively decoupled from neuroscience. It transitioned into a branch of engineering and applied mathematics, prioritizing statistical optimization over biological realism. Paradoxically, it was this abandonment of biological fidelity that enabled the rapid scaling and performance breakthroughs that followed. ]

#v(1em)
=== Achievements

#serif-text()[ With the training mechanism solved, the field exploded. The combination of Backpropagation, massive datasets, and @gpu hardware led to a "Cambrian Explosion" of neural architectures, each solving domains previously thought impossible for computers.

The revolution began in earnest with computer vision. @cnn:pl, such as AlexNet (2012) @Placeholder and later ResNet @Placeholder, introduced the idea of learning hierarchical features---detecting edges, then shapes, then objects---much like the human visual cortex. This allowed machines to classify images with superhuman accuracy.

Soon after, the focus shifted to sequence data. @rnn:pl and @lstm architectures gave machines a short-term memory, enabling breakthroughs in speech recognition and machine translation. However, the true paradigm shift occurred with the introduction of the Transformer architecture in 2017. By utilizing an "attention mechanism" to parallelize the processing of language, Transformers allowed for the training of massive @llm:pl like the @gpt.

These techniques have even transcended media generation. Deep Learning has solved fundamental scientific problems; notably, DeepMind's AlphaFold utilized these architectures to predict the 3D structure of proteins from their amino acid sequences, a 50-year-old grand challenge in biology @Placeholder. ]

#v(1em)
=== Shortcomings

#serif-text()[ Deep learning's reliance on computational scaling masks fundamental inefficiencies in both its hardware implementation and underlying algorithms. By simulating biological concepts on digital architectures not designed for them, the current paradigm is approaching physical and economic limits.

A primary limitation is the Von Neumann architecture, which physically separates processing units from memory. Deep neural networks, defined by massive matrices of synaptic weights, necessitate constant data transfer. For every inference step, billions of parameters must be fetched from off-chip DRAM, processed, and written back. This creates a severe memory bottleneck where system performance is bounded by bandwidth rather than processing speed @Placeholder.

Consequently, the energy cost of moving data significantly exceeds the cost of computation itself. Retrieving a single byte from DRAM consumes approximately three orders of magnitude more energy than performing a floating-point operation @Placeholder. Compounding this hardware friction, the dense matrix multiplications required for training scale quadratically with network size, making the pursuit of trillion-parameter models increasingly unsustainable.

Furthermore, the optimization algorithms driving this scale are fundamentally incompatible with physical biological systems. Backpropagation, while mathematically elegant, relies on a global error signal and suffers from the "weight transport problem"—the requirement that the backward pass utilizes the exact same synaptic weights as the forward pass. In organic tissue, synapses are unidirectional, and there is no known mechanism for a neuron to access the exact weight of a downstream synapse to calculate a gradient.

While a detailed technical analysis of these inefficiencies is presented in @mltechnicalities, the central issue is clear: modern AI prioritizes statistical optimization over physical realism. Overcoming the limitations of the Von Neumann bottleneck and backpropagation requires a paradigm shift toward architectures that inherently co-locate memory and computation. ]

#v(2em)
== Birth Of Neuromorphic

#serif-text()[ While the artificial intelligence community debated symbolic logic versus connectionism during the "AI Winter," significant developments were occurring in hardware physics. In the late 1980s at Caltech, physicist Carver Mead—a pioneer of @vlsi design—began to question the trajectory of digital computing.

Mead observed that while digital computers were becoming exponentially faster, they were also becoming less efficient in terms of energy per operation. He noted that using transistors as rigid, high-power switches to perform boolean logic was energetically wasteful compared to the biological systems they aimed to emulate.

In 1990, Mead published his seminal paper, _Neuromorphic Electronic Systems_ @Placeholder, coining the term "neuromorphic" to describe hardware that mimics the biological structure of the nervous system. His thesis proposed that rather than simulating neural equations via software on digital computers, engineers should construct physical hardware that exploits the same physical laws as the biological nervous system.

The foundational insight of the field was the physical analogy between silicon physics and ion-channel physics. In standard digital electronics, transistors are operated in "strong inversion," driven by high voltages to act as binary switches. Mead realized that a single transistor, operating in its "subthreshold" region, follows the same exponential Boltzmann statistics that govern the flow of ions through biological channels.

This realization implied that a single transistor could physically compute the non-linear functions used by biological neurons, but with significantly higher speed and lower power consumption. Consequently, synaptic functions could be implemented by single transistors rather than complex arrangements of logic gates.

To demonstrate this concept, Mead and his doctoral student Misha Mahowald developed the _Silicon Retina_ in 1991 @Placeholder. Unlike a standard camera, which captures full frames at fixed intervals (generating redundant data), the Silicon Retina operated asynchronously. It utilized analog circuits to compute spatial and temporal derivatives directly on-chip, outputting discrete "events" only when local light intensity changed.

This event-driven approach solved the redundancy problem inherent in frame-based sampling. If the scene remained static, the system transmitted no data and consumed negligible energy. This demonstrated that by aligning the hardware physics with the computational task, sensory information could be processed with a fraction of the power required by conventional digital systems. ]

#v(3em)
#line(length:100%)
#v(3em)
#serif-text()[ Since the inception of neuromorphic computing, neuroscience has also advanced significantly. While Mead’s early work was based on the physical intuition of the transistor, modern neuromorphic engineering now incorporates a richer understanding of neuronal dynamics, synaptic plasticity, and network architecture. To advance the field, we must combine these foundational hardware insights with the principles of modern mechanistic neuroscience. ]

#pagebreak()

= Biological Principles <biologicalprinciples>

#serif-text()[ The biological brain remains the gold standard for energy-efficient, robust, and adaptive computation. Since the establishment of the Neuron Doctrine, modern neuroscience has uncovered the specific physical mechanisms that underpin this efficiency. To engineer systems that truly rival biological performance, we must transcend the "spherical cow" abstractions of early cybernetics. We cannot simply mimic the brain's output; we must emulate its internal dynamics. This requires viewing the neuron not as a static summing unit, but as it functions in reality: a complex, time-dependent, and event-driven processor.

This section provides a mechanistic overview of the nervous system, translating biological observations into the computational primitives required for neuromorphic engineering. It explores the structural hierarchy of the neuron, the physics of the action potential, and the mathematical models used to capture these dynamics in silicon. ]

#v(2em)
== Neuron Structure & Function

#serif-text()[ In @historyanddevelopments we established the neuron as the fundamental computational unit of the brain. While it shares standard cellular machinery like mitochondria and a nucleus with other cells, it is morphologically specialized for information transmission. A neuron consists of three functional zones: ]

#box-text()[
- *The Input (Dendrites)*: A branching tree structure that collects signals from thousands of upstream neurons. This is where inputs are integrated.
- *The Integration Zone (Soma)*: The cell body where electrical potentials from the dendrites summate.
- *The Output (Axon)*: A long, cable-like structure that transmits the neuron's own signal to downstream targets.
]

#serif-text()[ The neuron exhibits a distinct morphological polarization that dictates the direction of information flow. The process begins at the "dendritic arbor", a complex branching structure that maximizes the surface area for synaptic connectivity. These dendrites serve as the primary receptor sites, where neurotransmitters binding to post-synaptic terminals induce local conductance changes. These signals propagate passively toward the soma (cell body), the neuron's central processing unit. The soma acts as an integrator, spatially and temporally summing the incoming synaptic currents. Finally, the processed signal is transmitted via the axon, a singular, elongated projection. In many vertebrate neurons, the axon is insulated by a myelin sheath, which facilitates saltatory conduction—a mechanism that allows high-speed signal propagation over long distances with minimal signal degradation. ]

#figure( image("figures/neuron.png", width:60%), caption: [The morphological structure of a biological neuron, illustrating the directional flow of information from dendritic input to axonal output.
© Angela Getz, Mathieu Ducros, Daniel Choquet / IINS et BIC / CNRS-Université de Bordeaux-Inserm.])

#serif-text()[ Functionally, the neuron operates as an electrochemical system enclosed by a cell membrane, known as the "lipid bilayer". This membrane is a thin, fatty structure that is impermeable to ions, acting as an electrical insulator. However, the fluids inside and outside the cell are conductive electrolytes. Consequently, the interaction between the insulating membrane and the conductive fluids creates a biological capacitor, capable of storing charge.

By actively pumping sodium ($"Na"^+$) out and potassium ($"K"^+$) in via the $"Na"^+$-$"K"^+$ ATPase pump, the cell maintains an electrochemical gradient across this capacitor, resulting in a stable "resting potential" of approximately $-70$ mV.

Computation occurs through the modulation of this voltage by competing synaptic inputs. Excitatory inputs cause ion channels to open, allowing positive ions to influx; this reduces the negative charge (depolarization) and pushes the potential toward the firing threshold. Conversely, inhibitory inputs activate channels for negative ions (like Chloride, $"Cl"^-$), driving the potential away from the threshold (hyperpolarization). The soma integrates these opposing push and pull signals. If the aggregate membrane potential surpasses a critical threshold (approximately $-55$ mV), the system undergoes a bifurcating phase transition. Voltage-gated sodium channels cascade open, triggering an @ap—a rapid, non-linear depolarization spike that propagates down the axon. This mechanism is governed by the "all-or-nothing" principle: the output is discrete and binary, effectively filtering out sub-threshold noise. ]

#serif-text()[ Immediately following a spike, the neuron enters a "refractory period" during which ion gradients are restored, imposing a hard limit on the maximum firing frequency and ensuring the temporal separation of events.

It is important to acknowledge that the biological brain exhibits significant cellular diversity beyond this idealized model. The nervous system contains non-neuronal cells known as "glia", which provide structural support and manage energy delivery, though they are generally not considered direct participants in fast information transmission. Additionally, while the vast majority of cortical neurons communicate via uniform action potentials (spikes), certain sensory neurons utilize "graded potentials", where the signal amplitude varies continuously. However, as spiking neurons represent the dominant computational paradigm for information processing in the cortex, this thesis focuses exclusively on the spiking model as the basis for neuromorphic emulation. ]

#v(2em)
== Action Potential & Spike Trains <actionpotentialandspiketrains>

#serif-text()[ As established in the previous section, the action potential is an "all-or-nothing" event. It serves as the fundamental mechanism by which neurons transmit information. Crucially, the waveform of this event is stereotypical: for a given neuron, every spike exhibits a nearly identical amplitude and duration (typically 1–2 ms), independent of the input intensity that triggered it. ]

#figure(include("figures/actionpotential.typ"),caption:[The phases of a typical neuronal action potential. (A) An incoming stimulus depolarizes the membrane past the threshold (−55 mV), triggering a rapid spike. (B) The membrane potential reaches a peak overshoot (+30 mV) before repolarizing. (C) A temporary undershoot (hyperpolarization) occurs before returning to the resting state (−70 mV). The neuron cannot fire during the absolute refractory period (D) and requires a stronger stimulus to fire during the relative refractory period (E).])

#serif-text()[ This biological invariance permits a fundamental simplification in neuromorphic modeling: ]

#box-text()[ If the spike waveform is invariant across neurons and time, the waveform itself carries no information. Consequently, the information content of the signal is encoded entirely in the precise timing of the spike. ]

#serif-text()[ To model this mathematically, we abstract the continuous biophysical voltage trace into a dimensionless point process. We treat the action potential not as a function of voltage over time, but as a singular event occurring at a precise instant, $t_f$, with negligible duration. The standard tool for this abstraction is the Dirac delta function denoted as, $delta(t)$.

The Dirac delta is a generalized distribution defined by the property that it is zero everywhere except at the origin, yet integrates to unity. This represents an idealized pulse of infinite height and zero width, containing a finite unit of effect. ]

#figure( kind: "eq", supplement: [Equation], caption: [The defining properties of the Dirac delta function.],[
$ delta(t) = cases(infinity "if" t = 0, 0 "if" t != 0), quad integral_(-infinity)^(+infinity) delta (t) dif t = 1 $
])<dirac_def>

#serif-text()[ Under this formalism, the output of a neuron is modeled not as a continuous signal, but as a "spike train"—a temporal sequence of these Dirac impulses. For a neuron emitting $N$ spikes at times ${t^((1)), t^((2)), ..., t^((N))}$, the output signal $S(t)$ is defined as: ]

#figure( kind: "eq", supplement: [Equation], caption: [A spike train represented as a sum of Dirac delta functions.], [ $ S(t) = sum_(f=1)^(N) delta(t - t^((f))) $
])<spike_train>

#serif-text()[ This abstraction allows the post-synaptic effect to be modeled using linear systems theory. In neuron models that use this framework, the interaction is treated as instantaneous charge deposition: the arrival of a delta function $delta(t-t_f)$ imparts a discrete step-change to the post-synaptic current. This mimics the rapid opening of ion channels without requiring the computational overhead of simulating the complex voltage trajectory. The shift from continuous values to discrete spike trains fundamentally alters the computational paradigm, moving from spatial representations (magnitude-based) to spatio-temporal representations (time-based). ]

#figure(include("figures/spiketrain.typ"),caption:[Transformation of continuous membrane voltage (top) into a discrete spike train (bottom).], placement: auto)

#v(2em)
== Neuron Models <neuronmodels>

#serif-text()[ In the quest to simulate the brain, there exists a fundamental trade-off between biological realism and computational efficiency. At the high end of the spectrum lie conductance-based models, most notably the Hodgkin-Huxley model. This formalism describes the neuron not as a simple computational unit, but as an electrical circuit with variable resistors representing the precise, non-linear opening and closing dynamics of specific ion channels (sodium, potassium, leak) @Placeholder.

Large-scale initiatives, such as the Blue Brain Project, utilize even more granular "multi-compartment" models. These simulations treat the neuron as a complex 3D structure, discretizing the dendritic arbor and axon into hundreds of segments to model how current flows through the specific morphology of the cell @Placeholder. While invaluable for pharmacological research, these models are computationally prohibitive for large-scale neuromorphic engineering. Simulating a mere second of biological time for a small network using these equations requires supercomputing resources.

To build practical, scalable neuromorphic hardware, we must abstract these biophysical details into a phenomenological model. We seek a mathematical framework that captures the essential computational properties—integration, leakage, and thresholding—without simulating the underlying molecular physics. ]

#v(1em)
=== The Leaky Integrate-and-Fire (LIF) Model

#serif-text()[ The standard approximation used in neuromorphic engineering is the @lif model. This framework aligns perfectly with the "point process" abstraction established in the previous section, as it treats action potentials as instantaneous, discrete events. Its state is defined by a single scalar variable, the membrane potential $u(t)$. The sub-threshold dynamics are governed by a linear differential equation analogous to a simple $R C$ (Resistor-Capacitor) circuit: ]

#figure(include("figures/lifcircuit.typ"), caption:[])

#figure( kind: "eq", supplement: [Equation], caption: [The Leaky Integrate-and-Fire (LIF) differential equation. The change in voltage is driven by the leak (decay to rest) and the input current.], $ tau_m​(dif u)/(dif t)=−(u−u_"rest")+R I(t) $)<lif_eq>

#serif-text()[ Where $tau_m$ is the membrane time constant (determining how fast the neuron "forgets"), $u_"rest"$ is the resting potential, $R$ is the membrane resistance, and $I(t)$ is the input current.

Connecting this to the spike train abstraction derived in the previous section, the input current I(t) is not continuous. It is a sequence of discrete events arriving from pre-synaptic neurons $j$ with weight $w_j$. Mathematically, this is modeled as a sum of Dirac delta functions: ]

#figure( kind: "eq", supplement: [Equation], caption: [Synaptic input modeled as a weighted sum of Dirac delta functions.], $ I(t)=sum j w_j sum f delta(t−t_j(f)) $)<lif_input>

#serif-text()[ Because the differential equation is linear below the threshold, we can solve it analytically. The membrane potential $u(t)$ becomes a convolution of the input spike train with the system's impulse response (a decaying exponential kernel). This means the potential at any moment is simply the sum of the decaying traces of all past spikes: ]

#figure( kind: "eq", supplement: [Equation], caption: [The analytical solution for the membrane potential. The current voltage is the superposition of all past inputs, decayed by time constant $tau_m$.], $ u(t)=u_"rest"+sum j w j sum f exp(−(t−t_j(f))/tau_m) $)<lif_sol>

#serif-text()[ The differential equation above describes the continuous sub-threshold dynamics. To complete the model, we must define the discrete "Fire" mechanism. The @lif neuron operates as a hybrid dynamical system: it integrates continuously until a discontinuity is triggered.

When the membrane potential $u(t)$ exceeds a specific threshold voltage $theta.alt$, the neuron emits a spike (a Dirac delta event). Immediately following this event, the membrane potential is not governed by the differential equation but is forced to a reset value, $u_"reset"$.

To mimic the biological limit on firing frequency, the model enforces an absolute refractory period, $Delta_"ref"$. After a spike occurs at time $t_f$, the neuron is clamped to the resting potential for a duration of $Delta_"ref"$. During this interval, the differential equation is suspended; the neuron ignores all inputs and cannot fire, regardless of the stimulus intensity. ]

#figure(include("figures/lifdynamics.typ"), caption:[The dynamics of an @lif neuron. (A) The membrane potential integrates inputs. (B) Upon crossing the threshold $theta.alt$, a spike is emitted and the voltage is reset. (C) The voltage is clamped during the refractory period.])

#serif-text()[ Mathematically, this firing condition is expressed as: ]

#v(1em)

#figure( kind: "eq", supplement: [Equation], caption: [The discrete firing and reset condition.], [
$ "If " u(t) > theta.alt arrow cases( "Emit spike: " S(t) arrow S(t) + delta(t), "Reset voltage: " u(t) arrow u_"reset", "Pause integration for " t in (t_f, t_f + Delta_"ref"] ) $
])<lif_reset>

#v(1em)

#serif-text()[ This equation represents the engine of most neuromorphic algorithms. It defines a system that integrates information over time and leaks it to ensure temporal relevance. However, once $u(t)$ crosses a threshold $theta.alt$, the linearity breaks and the neuron emits a spike and $u(t)$ is manually reset to $u_"reset"$. ]

#v(1em)
=== The Generalized (Adaptive) LIF Model

#serif-text()[ While the standard @lif model is computationally efficient, its one-dimensional nature limits it primarily to tonic spiking (regular firing under constant input). It struggles to replicate the complex, non-linear behaviors observed in the cortex, such as bursting (clusters of rapid spikes) or spike-frequency adaptation (slowing down after sustained activity).

To capture these dynamics without reverting to the computationally heavy Hodgkin-Huxley equations, we employ the @glif model. This extends the system by introducing a second state variable, $w(t)$, representing cellular adaptation. ]

#figure( kind: "eq", supplement: [Equation], caption: [The Adaptive GLIF system. The adaptation variable $w$ provides negative feedback, enabling complex dynamics like bursting and adaptation.], [
$ tau_m (dif u)/(dif t) &= -(u - u_"rest") + R I(t) - w \
  tau_w (dif w)/(dif t) &= a(u - u_"rest") - w $
])<glif_eq>

#serif-text()[ In this coupled system, $w$ provides a negative feedback loop. Every time the neuron spikes, $w$ increments by a constant $b$, acting as a physiological drag on the membrane potential. By adjusting the coupling parameters between $u$ and $w$, this two-dimensional system can be tuned to emulate the full spectrum of biological firing patterns.

It is natural to question whether such a mathematically reduced model can genuinely capture the behavior of biological neurons. While the @glif model discards the specific ionic mechanisms of the Hodgkin-Huxley equations, empirical validation demonstrates that it retains superior computational dynamics for large-scale modeling.

In the 2008 _Quantitative Single-Neuron Modeling Competition_ @Placeholder organized by the INCF, phenomenological models like the Generalized LIF (specifically the Adaptive Exponential Integrate-and-Fire) unexpectedly outperformed highly detailed biophysical models in predicting the precise spike times of real cortical neurons.

This counter-intuitive success is due to parameter sensitivity. Complex conductance-based models have dozens of unobservable parameters that are difficult to tune. In contrast, the GLIF model captures the "net effect" of these mechanisms using macroscopic parameters that can be robustly fitted to data. As demonstrated by Izhikevich (2003), this simple system of two differential equations is capable of reproducing all known firing patterns observed in the mammalian cortex @Placeholder. ]

#figure(include("figures/izhikevichpatterns.typ"), caption:[The Generalized LIF model is capable of reproducing the diverse firing patterns of biological cortical neurons, as categorized by Izhikevich (2003) @Placeholder.], placement: auto)

#serif-text()[ Consequently, for the purpose of neuromorphic engineering, the GLIF model represents the optimal trade-off between biological fidelity and computational efficiency. ]

#v(2em)
== Neural Coding <neuralcoding>

#serif-text()[ In classical digital computing, information is represented by combining bits into richer structures, such as floating-point or integer numbers. For instance, the luminance of a pixel is typically stored as a discrete 8-bit or 32-bit integer. Conversely, analog electronics represent values as continuous currents or voltages, offering infinite resolution within the dynamic range of the hardware. ]

#figure(include("figures/digitalanalogrepresentation.typ"), caption:[ Digital left analog right representation])

#serif-text()[ The biological brain occupies a unique middle ground. While neurons operate using analog membrane potentials, their communication output—the action potential—is discrete and binary. As established in @actionpotentialandspiketrains, the waveform of a spike is stereotypical; it looks like a "digital bit" in amplitude. However, unlike a digital computer which is synchronized to a rigid clock, these spikes occur in continuous time. Therefore, the information in the nervous system is not stored in the shape of the signal, but in the structure of the spike train itself.

Deciphering the "Neural Code"—the set of rules by which sensory stimuli are translated into these spike sequences—remains one of the central challenges in neuroscience. Currently, several coding schemes are hypothesized to coexist, each offering different trade-offs between latency, information density, and robustness. ]

#v(1em)
=== Rate Coding

#serif-text()[ The most traditional interpretation of neural activity is rate coding. In this paradigm, information is conveyed by the mean firing frequency of a neuron over a specific temporal window. A strong stimulus (e.g., high pressure on skin) elicits a high firing rate, while a weak stimulus results in sparse activity.

This model effectively treats the neuron as an Analog-to-Digital converter where the precise timing of individual spikes is treated as noise; only the average count carries the signal. While rate coding is robust and easily observed in motor neurons, it suffers from a fundamental latency barrier. To estimate a rate with reasonable precision, the post-synaptic neuron must integrate spikes over a significant duration (tens or hundreds of milliseconds). This contradicts the rapid reaction times (often $<100$ ms) observed in biological agents, suggesting that rate coding alone cannot account for time-critical processing. ]

#figure(include("figures/rateencoding.typ"), caption:[Rate Coding: The stimulus intensity is encoded in the frequency of the spike train. Stronger stimuli elicit more spikes per second.])

#v(1em)
=== Temporal Coding

#serif-text()[ To explain the speed of biological processing, neuromorphic engineering emphasizes temporal coding. In this regime, the precise timing of a spike carries significant information. A primary example is @ttfs coding.

In a @ttfs scheme, the intensity of a stimulus is inversely mapped to the latency of the response relative to a stimulus onset. A stronger input causes the neuron to integrate and cross the threshold faster, firing earlier than neurons receiving weak inputs. This shifts the computational model from counting spikes to a "race" between spikes.

In a network utilizing lateral inhibition (@wta), the first neuron to fire inhibits its neighbors, allowing a decision to be made as soon as the first meaningful bit of data arrives. This eliminates the need to wait for a time window to close, drastically reducing latency. Furthermore, since @ttfs coding prioritizes the strongest signals, it acts as a natural filter: the most prominent features arrive first, allowing the system to process signal over noise. ]

#figure(include("figures/temporalcoding.typ"), caption:[Temporal Coding (@ttfs): Stimulus intensity is encoded in the latency of the response. Stronger inputs ($I_1$) trigger an earlier spike ($t_1$) compared to weaker inputs ($I_2$).])

#v(1em)
=== The Phase Ambiguity Problem

#serif-text()[ A critical challenge in temporal coding is the need for a temporal reference frame. In Rate Coding, the "phase" (absolute timing) is irrelevant. However, in Temporal Coding, a spike at time $t$ only has meaning relative to a reference $t_0$. If the receiver does not know when the stimulus started, it cannot decode the latency.

In engineering, this is solved by a global clock or a "frame start" signal. In the brain, evidence suggests that background oscillatory rhythms (brain waves, such as theta or gamma cycles) may serve as this global reference, allowing populations of neurons to synchronize their "clocks" and decode relative timings accurately. ]

#figure(include("figures/phaseambiguity.typ"), caption:[The phase ambiguity problem in temporal encoding. Spikes occurring at the same relative phase ($phi_1$ and $phi_2$) across different oscillation cycles are mathematically indistinguishable ($phi_1 = phi_2 (mod 2pi)$). Without a mechanism to track the global cycle count, downstream neurons cannot determine whether a spike represents a delayed response to a previous stimulus or an early response to a new one.])

#v(1em)
=== Population & Sparse Coding

#serif-text()[
While single-neuron codes provide the basic signaling mechanism, the brain employs ensemble strategies to ensure robustness and precision. In population coding, variables are represented by the joint activity of a large group of neurons, each with broad, overlapping tuning curves. A classic example is found in the Primary Visual Cortex (V1), where orientation-selective neurons each respond maximally to a preferred angle but also fire weakly for nearby orientations. By reading the weighted population vector across the group, the network reconstructs the stimulus with far greater precision than any individual cell could provide alone.
The brain further optimizes for metabolic efficiency through sparse coding, where only a small fraction of neurons are active at any moment. This strikes a mathematical balance between representational capacity and energy cost, and is naturally enforced by lateral inhibition circuits that suppress weaker, competing responses. ]

#v(1em)
=== Coexistence of Codes

#serif-text()[ These schemes are not mutually exclusive but complementary. A circuit may use TTFS for a rapid initial response — alerting the system to a salient change — before transitioning to rate-based activity for sustained processing. Neuromorphic systems often adopt this hybrid approach, using temporal codes for energy-efficient sparse event transmission and rate-based readouts for interfacing with downstream control systems. This thesis follows the same principle, using TTFS encoding for the transmission of visual features combined with a population-level representation at the hidden layer. ]

#v(2em)
== Neural Networks <networks>

#serif-text()[ Having established the mathematical description of the individual neuron, we now turn to the collective behavior of these units. A single neuron, regardless of its dynamical complexity, is of limited computational utility in isolation. Functional intelligence emerges only when these units are organized into specific structural topologies.

The brain is not a random mesh of connections; it is constructed from recurring architectural "motifs" that appear across various cortical areas. Understanding these motifs is essential for designing neuromorphic systems that transcend simple feed-forward processing. ]

#v(1em)
=== Synaptic Efficacy & Weights

#serif-text()[ Before analyzing the structural topology of networks, we must define the fundamental unit of connectivity: the synapse. In the biological brain, neurons do not touch; they are separated by a microscopic gap known as the synaptic cleft. Communication across this gap is chemical, mediated by the release of neurotransmitters.

The efficiency of this transmission—determined by factors such as the amount of neurotransmitter released and the number of post-synaptic receptors—is abstracted in mathematical models as the synaptic weight ($w$).

In the @snn formalism, the weight represents a scaling factor for the incoming spike. When a pre-synaptic neuron $j$ fires a spike at time $t_j$, it induces a @psc in neuron $i$ scaled by the weight $w_(i j)$. Mathematically, the total synaptic input $I(t)$ is the weighted sum of all incoming spike trains: ]

#figure( kind: "eq", supplement: [Equation], caption: [The synaptic input current as a weighted sum of incoming impulses.], [
$ I_i(t) = sum_j w_(i j) dot S_j(t) $
])<synaptic_input>

#serif-text()[ Synaptic weights determine not just the magnitude but also if the synapse is excitatory or inhibitory. ]
#box-text()[
- *Excitatory Synapses ($w > 0$):* These depolarize the target neuron, pushing its membrane potential closer to the firing threshold (e.g., Glutamate synapses).
- *Inhibitory Synapses ($w < 0$):* These hyperpolarize the target neuron, pushing the potential away from the threshold (e.g., GABA synapses). ]

#serif-text()[ A fundamental constraint in biological networks, known as Dale's Principle, states that a neuron performs the same chemical action at all of its synaptic outputs. This means a neuron is strictly excitatory or strictly inhibitory; it cannot send positive signals to one neighbor and negative signals to another. While standard @ann:pl often violate this rule for mathematical convenience (allowing weights to flip signs during training), bio-plausible neuromorphic architectures often enforce this constraint to mimic the distinct populations of Pyramidal (excitatory) and Interneuron (inhibitory) cells found in the cortex. ]

#serif-text()[ The network must maintain a precise Excitation-Inhibition (E/I) Balance. The brain operates at a critical point of instability: ]

#box-text()[
- *Excess Excitation* leads to runaway feedback loops (analogous to epileptic seizures).
- *Excess Inhibition* leads to signal extinction (quiescence). ]

#v(1em)
=== Directionality

#serif-text()[ Structurally, neural topologies can be categorized by the flow of information.

In sensory peripheries (such as the retina) and early processing stages, information flows unidirectionally from input to output. This topology supports rapid, reflex-like feature extraction. This configuration is known as a feed-forward network, which is mathematically equivalent to a Directed Acyclic Graph (@dag) and serves as the standard architecture for most Deep Learning @cnn:pl.

In higher cognitive areas, the dominant topology is recurrence. Neurons form feedback loops, connecting back to themselves or to distinct layers. This recurrence introduces a time component to the computation, transforming the network into a dynamical system where the current output depends not only on the input but on the network's previous state (history). ]

#figure(include("figures/connectivity.typ"), caption:[Network topologies. (A) Feed-Forward. (B) Recurrent.])

#v(1em)
=== Synaptic Hypothesis: Structure As Function

#serif-text()[ A foundational premise in neuromorphic engineering, derived from biological observation, is that the neuron operates largely as a generic processing unit. The functional identity of a neural circuit—whether it processes visual stimuli or governs motor control—is determined principally by the topology and efficacy of its synaptic interconnections.

This paradigm, known as the Synaptic Hypothesis, posits that the physical configuration of synaptic weights constitutes the substrate for all computation and memory. Unlike traditional Von Neumann architectures, where data is retrieved from a distinct memory module and processed in a central CPU, biological systems eliminate the distinction between "data" and "program." Memory is not a static artifact, but a latent computational potential distributed across the network's structural graph. Consequently, learning in a neuromorphic system is realized through the physical alteration of these synaptic weights, ensuring robust, decentralized processing that is inherently resistant to localized hardware failure (graceful degradation). ]

#v(1em)
=== Inhibition Patterns

#serif-text()[ A ubiquitous micro-circuit motif in the cortex is lateral inhibition. In this configuration, an active excitatory neuron stimulates distinct inhibitory interneurons, which in turn suppress the activity of neighboring excitatory neurons. This competition engenders a @wta dynamic: as one neuron—representing a specific feature or decision—becomes active, it effectively silences its competitors. In the context of neuromorphic engineering, @wta circuits are indispensable; they provide a physical mechanism for both noise reduction, by actively suppressing weak, sub-threshold signals, and categorical decision making, enabling the circuit to autonomously select the most salient option without the need for a central processor to sort or compare values. ]

#figure(include("figures/lateralinhibition.typ"), caption:[The mechanism of lateral inhibition. (A) A highly stimulated neuron in the input layer strongly excites its corresponding output neuron while simultaneously sending lateral inhibitory signals to its immediate neighbors. (B) This architectural motif acts as a spatial filter, producing a contrast enhancement effect. A broad input stimulus (dashed blue line) is transformed into a sharper output response (solid purple line) characterized by an amplified center and suppressed surroundings (a "Mexican hat" profile), thereby sharpening signal boundaries.])

#serif-text()[ While lateral inhibition processes information in the spatial domain, Feed-Forward Inhibition (FFI) operates in the temporal domain. Structurally, this motif bifurcates an input signal into two parallel pathways: a direct excitatory route to the target neuron, and a disynaptic inhibitory route that reaches the same target with a slight synaptic delay. This architecture creates a narrow "temporal window of opportunity." Because the excitation triggers the neuron immediately before the delayed inhibition abruptly truncates the response, the neuron is prevented from integrating noise over extended durations. Consequently, FFI forces the neuron to function as a precise Coincidence Detector rather than a sluggish integrator, a dynamic that is fundamental to sound localization in the auditory cortex and fine-grain timing in the somatosensory system. ]

#v(2em)
== Biological Learning <bio_learning>

#serif-text()[ As previously established, the functional identity of a neural circuit is not defined by a transient software state, but by its physical hardware configuration. Consequently, "learning" in a biological substrate cannot be viewed as a simple parameter optimization; it is a fundamental morphological process. If structure dictates function, then the acquisition of new skills or memories necessitates the physical restructuring of the connectome itself.

Because the brain lacks a central supervisor or global communication bus, this restructuring must be driven by Locality. A synapse can only change based on information physically available at the cleft: the activity of the pre-synaptic axon, the voltage of the post-synaptic dendrite, and the immediate neurochemical environment. Despite this constraint, the brain successfully credits specific synaptic events with outcomes that occur seconds or minutes later.

This adaptation occurs across multiple timescales and spatial resolutions via two distinct mechanisms: Structural Plasticity (the rewiring of the network topology) and Synaptic Plasticity (the modulation of connection strength). ]

#v(1em)
=== Structural Plasticity

#serif-text()[ While synaptic weight adjustment accounts for rapid learning and pattern recognition, the long-term storage of information and the optimization of energy efficiency are governed by structural plasticity. This mechanism involves the physical genesis (synaptogenesis) and removal (pruning) of synapses and even entire axonal branches. ]

#box-text()[
- *Synaptogenesis*: When neurons are repeatedly co-active but lack a direct connection, the brain can physically grow new dendritic spines and axonal boutons to bridge the gap. This effectively alters the network's topology, creating new pathways for information flow where none existed before.
- *Pruning*: Equally critical is the removal of redundant or noisy connections. During sleep and developmental critical periods, the brain aggressively prunes weak synapses. This "sparsification" reduces metabolic cost and increases the signal-to-noise ratio of the circuit by removing irrelevant pathways. ]

#serif-text()[In the context of the Synaptic Hypothesis, structural plasticity represents the "compiling" of temporary associations into permanent hardware architecture. ]

#v(1em)
=== Synaptic Plasticity

#serif-text()[ Once a structural connection exists, its efficacy—the magnitude of the post-synaptic response to a pre-synaptic spike—must be tuned. In biological terms, this "weight" corresponds to the amount of neurotransmitter released and the density of receptors on the receiving side. This fine-grained adjustment is governed by local learning rules. ]

#v(1em)
=== Hebbian Learning: Rate-Based Correlation

#serif-text()[ The foundational axiom of biological learning was postulated by Donald Hebb in 1949. Hebb proposed that synaptic efficiency is a function of the correlated activity between two neurons. Colloquially summarized as "Neurons that fire together, wire together," this rule implies that the brain learns by detecting statistical regularities in sensory input.

Mathematically, if neuron $A$ consistently takes part in firing neuron $B$, the connection from $A$ to $B$ is strengthened. This mechanism allows the brain to perform unsupervised clustering, physically encoding associations between features that occur simultaneously in the environment (e.g., the smell of smoke and the sight of fire). ]

#v(1em)
=== Spike-Timing-Dependent Plasticity (STDP)

#serif-text()[ Modern neuroscience has refined Hebb’s macroscopic theory into a precise, millisecond-scale mechanism known as @stdp. Unlike rate-based models, @stdp operates on the precise timing of individual action potentials, introducing the critical element of causality.

The @stdp rule adjusts the synaptic weight based on the relative timing difference ($Delta t$) between the pre-synaptic input and the post-synaptic output: ]

#box-text()[
- *@ltp*: If the input spike arrives *before* the output spike ($Delta t > 0$), it implies the input contributed to the firing. The synapse is strengthened to reinforce this causal link.
- *@ltd*: If the input spike arrives *after* the output spike ($Delta t < 0$), the input was irrelevant to the decision. The synapse is weakened. ]

#serif-text()[ This asymmetry allows the network to self-organize, naturally filtering out random noise while reinforcing specific spatiotemporal patterns. ]

#figure(include("figures/stdpcurve.typ"), caption:[The @stdp Learning Curve. Synaptic weight change is plotted against spike timing difference. Pre-before-post timing triggers strengthening (@ltp), while post-before-pre triggers weakening (@ltd).],placement: auto)

#v(1em)
=== Homeostatic Plasticity

#serif-text()[ If Hebbian mechanisms (@ltp) were the sole drivers of plasticity, neural networks would be inherently unstable. A positive feedback loop would emerge where strengthened synapses drive higher firing rates, which in turn induce further strengthening, leading to runaway excitation (seizures). Conversely, unchecked LTD could silence a network entirely.

To maintain stability, the brain employs Homeostatic Plasticity (or Synaptic Scaling). This is a global regulatory mechanism that operates on a slower timescale (minutes to hours). It functions as a negative feedback loop: if a neuron's average firing rate exceeds a target set-point, the cell chemically downscales the strength of all its incoming synapses. This ensures that neurons remain within a sensitive dynamic range, preventing saturation regardless of how strong the inputs become. ]

#v(3em)
#line(length:100%)
#v(3em)
#serif-text()[ The following chapter shifts perspective---from biology to engineering. We examine the mathematical framework of modern Deep Learning, to identify where its abstractions diverge from biological reality and what computational cost those divergences impose. This analysis will make explicit the bottlenecks that neuromorphic architectures are designed to resolve, grounding the methodological choices of this thesis in a concrete technical rationale. ]

#pagebreak()

= Technical Details Of Machine Learning <mltechnicalities>

#serif-text()[ This chapter delineates the technical foundations of modern artificial intelligence, contrasting the established paradigms of @dl with the emerging principles of Neuromorphic Engineering. We begin by analyzing the algorithmic architecture of standard Deep Learning, identifying the computational bottlenecks inherent in its reliance on dense matrix multiplication and backpropagation.

A critical distinction must be drawn between biological plausibility and bio-inspired engineering. From an engineering perspective, the primary objective is functional utility. An engineer may treat the brain merely as a source of heuristic inspiration rather than a blueprint to be copied dogmatically. However, the pursuit of biologically plausible systems remains vital; it offers potential advantages in robustness and energy efficiency while serving as a verification tool for neuroscience. ]

#v(2em)
== Optimization

#serif-text()[ Optimization is the selection of a "best candidate" with regard to defined criteria. Biological learning fits this description, where the optimal candidate is the configuration of synaptic weights that performs well for a specific task. Therefore, it is useful to establish a mathematical framework for this process.

Fundamentally, a deep learning model operates as a function approximator. We assume the existence of an unknown underlying function $f: X arrow Y$ that perfectly maps inputs to their target outputs. Since this true function is unknown, we construct a family of hypothesis functions $f_bold(theta)(bold(x))$ to approximate it. Here, $bold(theta) in RR^d$ represents the state of the system—a vector containing all tunable parameters, such as synaptic weights or biases. The dimensionality $d$ corresponds to the degrees of freedom of the model.

The key problems in optimization are defining the objective goal (the loss function) and finding the parameter configuration $bold(theta)$ that achieves that goal. ]

#v(1em)
=== Supervised Learning

#serif-text()[ To guide the search for optimal parameters $hat(bold(theta))$, we must quantify the divergence between the model's predictions and the ground truth. We define a scalar Loss Function $cal(L)(hat(bold(y)), bold(y))$ that evaluates the error on a single data point. To ensure generalization, we seek to minimize the Empirical Cost Function $J(bold(theta))$, defined as the average loss over a dataset of size $N$:

$ J(bold(theta)) = 1/N sum_(i=1)^N cal(L)( f_bold(theta)(bold(x)_i), bold(y)_i) $

Geometrically, the cost function $J(bold(theta))$ induces an Optimization Landscape. Finding a low-energy state in this non-convex topology is the central challenge of AI training. We rely on iterative optimization algorithms, principally Gradient Descent. This method updates the system state in the direction opposite to the gradient vector $nabla_(bold(theta)) J(bold(theta))$ (the steepest ascent). The update rule for iteration $t$ is:

$ bold(theta)_(t+1) arrow.l bold(theta)_t - eta nabla_(bold(theta)) J(bold(theta)_t) $

Here, $eta$ represents the Learning Rate. Because computing the gradient over the entire dataset $N$ is computationally prohibitive, modern AI employs Stochastic Gradient Descent (SGD), approximating the gradient using small random subsets (mini-batches). This introduces beneficial noise, preventing the system from getting trapped in shallow local minima.

Crucially, gradient descent requires the loss function to be differentiable. As will be discussed later, this presents a significant challenge for optimizing neuromorphic systems utilizing discrete, non-differentiable spike trains. ]

#figure(include("figures/gradientdecent.typ"), caption:[The Optimization Landscape. The system seeks to traverse the high-dimensional surface defined by $J(bold(theta))$ to find the global minimum $bold(theta)^*$, using the gradient $nabla J$ as a navigational compass.])

#serif-text()[ Strictly minimizing the empirical cost carries the risk of overfitting — the model memorizes training data including noise rather than learning the underlying function. In biological systems this is naturally regulated by metabolic constraints; the brain prunes weak connections to maintain a sparse topology, effectively trading model complexity for generalization. In artificial systems this is managed via explicit regularization penalties added to the cost function. ]

#v(1em)
=== Unsupervised Learning

#serif-text()[ While supervised learning relies on labeled targets, biological systems predominantly learn via Unsupervised Learning. In this regime, the dataset consists only of input vectors $X = {bold(x)_1, ..., bold(x)_N}$. The optimization objective shifts from minimizing prediction error to minimizing representation error.

Mathematically, the goal is often to discover a lower-dimensional manifold that efficiently captures the structure of the data. A common formulation is the minimization of Reconstruction Loss, where the system attempts to compress the input into a latent code and reconstruct it:

$ J(bold(theta)) = 1/N sum_(i=1)^N || bold(x)_i - f_"decode"(f_"encode"(bold(x)_i; bold(theta))) ||^2 $

Alternatively, the system may optimize for clustering density or distances between feature centroids. The distinction between supervised and unsupervised learing is critical for Neuromorphic Engineering, as biological plasticity rules (like STDP) are unsupervised, functioning by detecting statistical correlations in the input stream to build internal representations without external labels. ]

#v(2em)
== Deep Learning Framework

#serif-text()[ Modern Deep Learning aggregates simple units into high-dimensional layers. A deep network with $L$ layers is expressed as a composite function mapping input $bold(x)$ to output $bold(y)$ through nested operations:

$ bold(y) = f_L ( ... f_2 ( f_1 ( bold(x) ) ) ) $

During the Forward Pass, each layer performs an Affine Transformation (a linear rotation and scaling of data via weight matrix $bold(W)$ and bias $bold(b)$) followed by a Non-Linear Activation ($sigma$):

$ bold(z)^((l)) = bold(W)^((l)) bold(a)^((l-1)) + bold(b)^((l)) $
$ bold(a)^((l)) = sigma(bold(z)^((l))) $

The non-linearity prevents the deep stack from collapsing into a single linear equation. Modern networks rely on the Rectified Linear Unit (ReLU), $f(x) = max(0, x)$. Its derivative (0 or 1) preserves the magnitude of the gradient, allowing error signals to travel through deep structures without vanishing. ]

#figure(include("figures/activations.typ"), caption:[Activation Functions. The Sigmoid (left) saturates gradients. The ReLU (right) preserves gradient magnitude for positive inputs.])

#serif-text()[ During the Backward Pass, Backpropagation recursively applies the Chain Rule via Automatic Differentiation to attribute the total error $J(bold(theta))$ to specific weights.

To achieve high throughput, these operations are vectorized. The affine transformation for an entire layer is executed as a Dense Matrix Multiplication. This is the defining characteristic of modern AI hardware. A deep network is effectively a sequence of massive matrix multiplications, which is highly parallelizable on @gpu:pl]

#figure(include("figures/matrixmath.typ"), caption:[Deep Learning as Matrix Multiplication. Forward and backward passes rely on dense matrix products, necessitating high-bandwidth memory access.])

#v(1em)
=== Convolutional Neural Networks (CNNs)

#serif-text()[ For visual tasks, standard Multi-Layer Perceptrons scale poorly; connecting every pixel to every neuron ignores the spatial structure of the data and creates an intractable number of weights. To solve this, @dl utilizes @cnn:pl.

CNNs apply small, learnable weight matrices known as "kernels" or "filters" that slide (convolve) across the input image. This architecture introduces two critical inductive biases: ]
#box-text()[
1. *Local Connectivity:* Neurons only process a small, local receptive field, analogous to the biological visual cortex.
2. *Weight Sharing:* The exact same kernel is applied across the entire image, drastically reducing the number of tunable parameters and establishing translation invariance (a feature learned in one corner of an image can be recognized anywhere else).]

#serif-text()[While CNNs are the standard baseline for spatial processing, they remain fundamentally synchronous and frame-based, evaluating the entire image structure in dense mathematical passes regardless of local activity. ]

#v(2em)
== Why Is Deep Learning Inefficient?

#serif-text()[ While the matrix-centric formulation of Deep Learning enables high-throughput parallelization on GPUs, it fundamentally conflicts with the physical constraints of modern computing hardware. As models scale to billions of parameters, the primary bottleneck shifts from algorithmic capability to physical realizability. This inefficiency manifests in four distinct engineering dimensions: ]

#v(1em)
=== The Von Neumann Bottleneck & Data Movement

#serif-text()[ The most significant physical limitation is the Von Neumann Architecture, which physically separates the Processing Unit from the Memory Unit. To perform a single inference step, the processor must fetch the entire weight matrix from off-chip DRAM to on-chip registers, perform the calculation, and write the results back.

According to Horowitz and Dally @Placeholder, fetching a 32-bit value from off-chip DRAM consumes approximately 640 pJ, whereas performing a floating-point addition on that value consumes only 0.1 pJ. The system expends 99.9% of its energy transporting data, and only 0.1% actually computing. ]

#figure(include("figures/vonneuman.typ"), caption:[The Von Neumann Bottleneck. The separation of memory and compute forces massive energy expenditure on data transport.])

#v(1em)
=== Dense Processing of Sparse Data

#serif-text()[ Standard Deep Learning implementations rely on Dense Matrix Multiplication (GEMM). This approach is algorithmically rigid: it executes the same number of operations regardless of the data content.

Real-world sensory data is often highly sparse, and the ReLU activation function naturally produces activation maps where the majority of values are zero. However, a standard GPU is "blind" to this sparsity. It will dutifully fetch a zero from memory and multiply it by a weight ($0 times w = 0$), consuming energy and clock cycles to produce a null result. Deep Learning's inability to exploit this silence represents a massive structural inefficiency. ]

#v(1em)
=== The High Cost of Synchrony

#serif-text()[ Deep Learning hardware is typically Synchronous, operating in lockstep with a global clock. Driving a high-frequency clock signal across an entire silicon die forces billions of transistors to charge and discharge continuously, regardless of whether the chip is doing useful work. In high-performance processors, this clock distribution network alone can consume 30% to 40% of the total power budget. Furthermore, global synchronization enforces a "worst-case" latency: faster computations must sit idle and wait for the slowest operations to finish before the next clock cycle begins. ]

#v(1em)
=== Backpropagation and Global Dependencies

#serif-text()[ Finally, Backpropagation imposes severe constraints on memory and latency because it is non-local in both time and space. To update a specific weight, the system must wait for the Forward Pass to finish, calculate the global error, and wait for the backward pass to propagate the gradient.

This creates a "Locking Problem." The activations of every intermediate layer must be stored in high-speed memory (VRAM) for the duration of the entire pass, preventing that memory from being reused. Additionally, a local synapse cannot adapt to local changes instantly; it is shackled to the global error loop. ]

#v(2em)
== Principles of Neuromorphic Engineering

#serif-text()[ As established in the _History & Developments_ chapter, Neuromorphic Engineering is the translation of biological dynamics into silicon hardware. It replaces the rigid, clock-driven logic of standard computing with the adaptive, event-driven dynamics of neural tissue. This approach rests on three architectural pillars that directly address the bottlenecks of Deep Learning: ]

#box-text()[
- *Co-location of Memory and Compute (The Synaptic Principle):* Neuromorphic architectures eliminate the Von Neumann bottleneck by distributing memory across the silicon die. Each artificial neuron stores its own state and synaptic weights locally, processing data *in situ* to eliminate the energy cost of shuttling data.
- *Event-Driven Asynchrony (The Action Potential Principle):* Neuromorphic systems abandon the global clock. They operate asynchronously, driven strictly by the arrival of data. If a part of the network is not processing information, it consumes negligible power, ensuring energy scales linearly with task complexity rather than network size.
- *Sparse Communication (The Spike Principle):* Neuromorphic systems utilize binary Spikes for communication. Information is encoded in the precise timing of events rather than complex magnitudes, drastically reducing the bandwidth required to transmit information between neurons. ]

#v(2em)
== Training Spiking Networks

#serif-text()[ While the physical architecture of neuromorphic systems is highly efficient, training these networks presents a fundamental mathematical challenge. Standard deep learning relies on gradient descent, but backpropagation cannot be directly applied to native @snn:pl.

In a spiking network, the neuron's activation function is a discontinuous step function (the Dirac delta event threshold). The derivative of this function is zero everywhere except at the exact moment of the spike, where it is undefined. Consequently, gradients calculated using the chain rule immediately drop to zero—known as the "Dead Neuron" problem—preventing error signals from flowing backward through the network to update the weights.

To circumvent this non-differentiability and optimize network parameters, the field of neuromorphic engineering generally employs two distinct paradigms: ]

#v(1em)
=== Direct Weight Transfer

#serif-text()[ A pragmatic engineering approach to bypass the dead neuron problem is offline training. In this paradigm, a standard, continuous @ann (such as a network utilizing ReLU activations) is trained conventionally using backpropagation. Once convergence is achieved, the learned weights are directly mapped onto a structurally identical Spiking Neural Network.

The underlying premise is that the continuous activation values of the ANN can be approximated by the discrete firing rates of the SNN over a set time window. While this method allows the spiking system to inherit the high accuracy of gradient descent, direct weight transfer requires careful scaling and normalization. If the weights are copied without adjustment, the resulting SNN may suffer from catastrophic saturation (firing constantly) or severe signal degradation (failing to reach the spiking threshold). ]

#v(1em)
=== Native Local Learning (STDP)

#serif-text()[ To fully exploit the energy efficiency and event-driven dynamics of neuromorphic hardware, training must ideally occur natively on the spiking substrate. This requires abandoning global backpropagation in favor of biologically plausible, mathematically local learning rules.

As established in @bio_learning, Spike-Timing-Dependent Plasticity (@stdp) adjusts synaptic weights based strictly on the temporal correlation of local pre- and post-synaptic spikes. Because STDP relies exclusively on local physical events rather than global error gradients, it does not require a differentiable loss function. This allows the network to completely bypass the dead neuron problem, enabling unsupervised feature extraction and real-time adaptation directly on the spiking architecture. ]

#v(1em)
=== Surrogate Gradient Descent

#serif-text()[ For completeness, it must be noted that the current dominant paradigm in SNN research utilizes Surrogate Gradients. In this approach, the network operates using the discontinuous spike step-function during the forward pass, but temporarily replaces the undefined derivative with a smooth, continuous approximation (a "surrogate") during the backward pass. While this thesis focuses on evaluating direct weight transfer and native unsupervised STDP, surrogate methods represent a highly effective hybrid approach, allowing backpropagation-like algorithms to estimate gradients across discrete spiking layers. ]

#v(2em)
== Neuromorphic Hardware Techniques

#serif-text()[ Central to realizing these computational efficiencies in physical hardware is Address Event Representation (AER), a communication protocol that mirrors the sparse nature of biological spikes. Instead of continuous data streaming, the hardware only transmits the "address" of a firing neuron across a shared digital bus, allowing a single physical wire to represent thousands of virtual axonal projections. ]

#figure( include("figures/inmemory.typ"), caption: [In-Memory Computing via a Crossbar Array. Unlike von Neumann architectures, memory and computation are physically co-located. Input voltages ($V$) are applied to the wordlines. Memory elements at the junctions hold programmable conductances ($G$). Multiplication is natively performed at each junction by Ohm's Law ($I=V times G$), and resulting currents are summed along the bitlines via Kirchhoff's Current Law. This allows dense matrix-vector multiplications to occur in a single analog time step with zero data transport cost.])

#serif-text()[ The crossbar array provides a direct structural surrogate for the neural neuropil. Because the architecture handles multiplication and summation natively through physical laws, it is uniquely suited to implement biological "macro-motifs." By routing bitline currents through local feedback loops, the hardware can instantiate complex dynamics such as Lateral Inhibition and Winner-Take-All circuits without the overhead of high-level software instructions.

This synergy between physical topology and functional motifs allows the hardware to inherit the computational efficiency of the neocortex, effectively making the architecture itself the algorithm. ]

#figure( include("figures/inmemoryhierarcy.typ"), caption:[Architectural Comparison. (Left) The Von Neumann architecture separates memory and compute, creating a bottleneck. (Right) The Neuromorphic architecture co-locates them, mimicking the distributed topology of biological neural networks.] )

#v(3em)
#line(length:100%)
#v(3em)
#serif-text()[ Having established the theoretical limitations of traditional deep learning and the physical principles of neuromorphic engineering, the following chapter details the specific methodologies, network architectures, and software frameworks utilized in this thesis to evaluate ANN-to-SNN weight conversion and native STDP learning on visual event data. ]

#pagebreak()

= Method <method>

#serif-text()[ This chapter details the specific implementations of the neuromorphic architectures proposed to address the limitations of standard deep learning. Aligning with the biological constraints of sparsity, asynchrony, and locality established in previous chapters, we outline the construction and evaluation of a @snn.

To empirically validate the theoretical advantages of neuromorphic algorithms, we evaluate the system on a benchmark image classification task. The experiment is bifurcated into three distinct phases:]

#box-text()[
1. *Neuron Model Evaluation:* Evaluating the decoding efficiency and accuracy of different simulated spiking models, specifically comparing a biologically inspired Leaky Integrate-and-Fire (LIF) model against a computationally streamlined Current-Accumulating (Ramp) model.
2. *Inference via Weight Transfer:* Evaluating the zero-shot performance of these SNNs initialized with weights directly mapped from a classically trained Artificial Neural Network (ANN).
3. *Native Unsupervised Learning:* Training the SNN from scratch utilizing local Spike-Timing-Dependent Plasticity (@stdp).
]

#v(2em)
== Dataset & Pre-processing

#serif-text()[ To benchmark these algorithms, we require a dataset that necessitates the extraction of complex spatial features but remains computationally tractable for rapid experimental iteration. We utilize the MNIST database of handwritten digits @Placeholder.

The dataset consists of a training set of 60,000 examples and a test set of 10,000 examples of digits (0-9). Each instance is a $28 times 28$ pixel grayscale image. While standard deep learning models routinely score over 90% accuracy on this task, making it largely a solved problem in classical AI, its well-understood feature space makes it an ideal, isolated baseline. Because the spatial hierarchy of MNIST is relatively shallow, it allows us to evaluate the efficacy of neuromorphic learning rules without the confounding variables introduced by massive, multi-layered convolutional architectures.

Crucially, the MNIST images are pre-processed by the dataset creators to be size-normalized and centered within the pixel grid using the center of mass of the pixels. This spatial alignment is a vital prerequisite for our chosen network topology. Unlike Convolutional Neural Networks (@cnn:pl), which slide localized filters across an image, the Fully Connected Network (@fcn) utilized in this thesis lacks translation invariance. If a digit were shifted several pixels off-center, the @fcn would perceive it as an entirely novel pattern. The pre-centered nature of MNIST mitigates this limitation, ensuring that the network can reliably map specific geometric strokes to specific input neurons.

Furthermore, the dataset exhibits a high degree of spatial sparsity. In a typical MNIST image, the vast majority of pixels represent the empty background. From a neuromorphic engineering perspective, this sparsity is highly advantageous. As established in the theoretical framework, event-driven systems expend energy strictly when events occur. A sparse input array ensures that the majority of input neurons remain quiescent, minimizing bus congestion and validating the energy-efficiency claims of the proposed Spiking Neural Network (@snn).

Before the raw images can be converted into temporal spike trains, they must undergo standard spatial pre-processing to ensure compatibility with the network's mathematical boundaries. This consists of two primary transformations: ]

#box-text()[
1. *Normalization*: Raw pixel intensities in the MNIST dataset range from $0$ (pure black) to $255$ (pure white). To stabilize the learning algorithms and ensure consistent weight scaling, these values are strictly normalized to a continuous float range of $p_i \in [0.0, 1.0]$.
2. *Flattening*: Because this thesis utilizes a Fully Connected Network (FCN) to facilitate direct weight transfer, the 2D spatial structure of the images must be unrolled. Each $28 times 28$ matrix is flattened into a 1-dimensional vector of $784$ elements. ]

#serif-text()[ Consequently, every individual image is presented to the system as a discrete array of $784$ normalized intensities. In the classical Artificial Neural Network (ANN), these continuous values are fed directly into the input neurons. However, because Spiking Neural Networks (SNNs) operate exclusively on discrete events, these normalized values must be passed through a temporal encoding algorithm before inference or learning can begin. ]

#figure( include("figures/dataexample.typ"), caption: [Sample of the MNIST dataset. The 28x28 images are normalized and flattened into 1D vectors before being translated into temporal spike events.])

#figure( include("figures/dataexample.typ"), caption: [Sample of the MNIST dataset. The 28x28 images are normalized and flattened into 1D vectors before being translated into temporal spike events.])


#v(2em)
== Information Representation

#serif-text()[ The choice of neural code lays the foundation for information flow and dictates the efficiency of the entire system. While Rate Coding (encoding pixel intensity as spike frequency) is straightforward and simple to implement with standard Integrate-and-Fire neurons, it is inefficient compared to @ttfs. Rate codes require integration over extended time windows to calculate an average, introducing latency and saturating the network bus with redundant spikes. Furthermore, on digital hardware rate coding imposes additional stress on the system due to rapid switching which is very bad for transistor power draw and bus congestion.

To maximize energy efficiency and processing speed, this implementation utilizes a @ttfs temporal encoding. In this regime, a single spike carries the information payload. A high-intensity (bright) pixel triggers an early spike, while a low-intensity (dark) pixel triggers a late spike. This compresses the spatial information into a highly sparse, priority-driven queue; downstream neurons begin processing as soon as the most salient features arrive, without waiting for an entire frame to integrate.

As noted in @neuralcoding, temporal codes suffer from Phase Ambiguity—downstream neurons need a reference "clock" to decode latency. To resolve this without relying on a rigid, global system clock, we simulate the biological concept of a *saccade* (the rapid movement of the eye to fixate on a target). The initial presentation of the image acts as a synchronized global event ($t_0$). All subsequent input spikes are evaluated relative to this saccade onset, providing a natural, biologically plausible temporal reference frame. ]


#v(1em)
=== Encoding

#serif-text()[ To convert the continuous pixel intensities of the MNIST dataset into the discrete TTFS spike trains, the input space must be mathematically normalized and mapped to a temporal delay. For a given input image, we extract the luminance of each pixel and normalize it to a bounded range, where $p_i in [0, 1]$ ($1$ representing maximum intensity and $0$ representing the background).

We implement two distinct conversion mappings to evaluate latency dynamics: Linear and Logarithmic. Let $T_"max"$ represent the maximum allowed simulation window for a single inference step.

For the Linear mapping, the spike latency $t_i$ is inversely proportional to the pixel intensity: ]

#figure( kind: "eq", supplement: [Equation], caption: [Linear Intensity-to-Delay Encoding], [
$ t_i = T_"max" - (T_"max" dot p_i) $
])

#serif-text()[ For the Logarithmic mapping, the delay is scaled logarithmically, which allocates higher temporal resolution to brighter pixels, further segregating the most salient features at the start of the simulation window: ]

#figure( kind: "eq", supplement: [Equation], caption: [Logarithmic Intensity-to-Delay Encoding], [
$ t_i = T_"max" - (T_"max" dot (log(1 + p_i))\(log(2))) $
])

#serif-text()[ Under both mappings, the brightest pixels fire immediately near $t=0$, transmitting the most critical structural features of the digit first, while background pixels are suppressed. ]


#v(1em)
=== Decoding and Neuron Models

#serif-text()[ Decoding the temporal information generated by the @ttfs encoding requires a neuron model capable of accumulating discrete events. However, a fundamental engineering tension exists between biological fidelity, computational efficiency, and the ability to strictly decode temporal order.

To systematically evaluate these trade-offs, this thesis implements and benchmarks four distinct neuron models. These models range from simple arithmetic accumulators requiring global synchronization to complex, fully asynchronous dynamic systems. Each model processes incoming spikes by updating its membrane potential $V_m(t)$ when an event arrives at time $t$, carrying a synaptic weight $w_i$. ]

#v(1em)
#mini-header()[Model A: The Simple Window Integrator (Standard IF)]

#serif-text()[ The most computationally lightweight approach is the standard Integrate-and-Fire (IF) model without any leak or decay mechanisms. In this paradigm, the neuron acts as a pure arithmetic accumulator during the simulation window. ]

#figure( kind: "eq", supplement: [Equation], caption: [Simple Window Integration], [
$ V_m(t) = V_m(t_"prev") + w_i $
])

#serif-text()[ *Computational Cost:* This model is extremely cheap to execute on digital hardware, requiring only a single addition operation $O(1)$ per incoming spike.
*Temporal Decoding Capability:* While highly efficient, this model is theoretically flawed for pure @ttfs decoding. It cannot differentiate the relative order of incoming signals. If Spike A ($w=5$) arrives at $t=1$ and Spike B ($w=2$) arrives at $t=10$, the final potential is identical to the reverse arrival order.
*Synchronization:* Because the potential never naturally decays, this model relies entirely on a rigid, globally synchronized "saccade" (a hard reset of $V_m$ to $0$ at the end of the time window) to prevent the network from firing continuously due to lingering historical noise. ]


#figure( include("figures/neuronramp.typ"), caption: [Network architecture. The 28x28 images are flattened and passed through a Fully Connected Network. Both the ANN and SNN share this identical macroscopic topology.])

#figure(
kind: "algo",
caption: [Model A: Simple Window Integrator (Standard IF)],
supplement: [Algorithm],
mono-text(pseudocode-list(hooks: .5em, indentation: 1em, booktabs: true)[
evaluate_model_A_IF(incoming_spikes, weights, threshold) -> integer:
  + sort(incoming_spikes, by_time)
  + v_m = 0.0
  + firing_time = None
  +
  + for spike in incoming_spikes:
    + weight = weights[spike.source]
    + v_m = v_m + weight  // Pure arithmetic accumulation
    +
    + if v_m >= threshold:
      + firing_time = spike.time
      + break
  +
  + return firing_time
]))

#v(1em)
#mini-header()[Model B: The Standard Leaky Integrate-and-Fire (LIF)]

#serif-text()[ To inject biological realism, the standard LIF model penalizes late-arriving spikes by decaying the membrane potential exponentially over time, governed by a membrane time constant $tau_m$. ]

#figure( kind: "eq", supplement: [Equation], caption: [Standard Leaky Integration], [
$ V_m(t) = max (0, V_m(t_"prev") dot exp(-(t - t_"prev")\(tau_m)) + w_i ) $
])

#serif-text()[ *Computational Cost:* This model is significantly more intensive, requiring the calculation of exponential functions for every discrete event, which consumes substantial clock cycles on standard arithmetic logic units.
*Temporal Decoding Capability:* The exponential leak naturally favors spikes that arrive in rapid succession, providing a basic temporal filter. However, standard LIF struggles to strictly prioritize *order* in a @ttfs scheme unless the time constant $tau_m$ is perfectly tuned to the specific temporal distribution of the dataset.
*Synchronization:* Similar to Model A, while the leak reduces residual noise, it generally still requires a global saccade reset between distinct inference phases to guarantee a clean slate for the next image. ]

#figure( include("figures/thresholdsensitive.typ"), caption: [Network architecture. The 28x28 images are flattened and passed through a Fully Connected Network. Both the ANN and SNN share this identical macroscopic topology.])

#figure(
kind: "algo",
caption: [Model B: Standard Leaky Integrate-and-Fire (LIF)],
supplement: [Algorithm],
mono-text(pseudocode-list(hooks: .5em, indentation: 1em, booktabs: true)[
evaluate_model_B_LIF(incoming_spikes, weights, threshold, tau_m) -> integer:
  + sort(incoming_spikes, by_time)
  + v_m = 0.0
  + t_prev = 0.0
  + firing_time = None
  +
  + for spike in incoming_spikes:
    + weight = weights[spike.source]
    + delta_t = spike.time - t_prev
    +
    + // Apply exponential leak based on time delta
    + v_m = max(0.0, v_m \* exp(-delta_t / tau_m) + weight)
    +
    + if v_m >= threshold:
      + firing_time = spike.time
      + break
    +
    + t_prev = spike.time
  +
  + return firing_time
]))

#v(1em)
#mini-header()[Model C: The Current-Accumulating Linear Ramp]

#serif-text()[ To strictly enforce order differentiation without the computational overhead of exponentials, Model C introduces a linear time-dependent accumulator. In a @ttfs code, earlier spikes must have a disproportionate influence. To achieve this, an incoming spike adds its weight to an internal current variable $I(t)$, which acts as the persistent "slope" of the membrane potential. ]

#v(1em)
#figure( kind: "eq", supplement: [Equation], caption: [Current-Accumulating Dynamics], [
$ I(t) = I(t_"prev") + w_i $ \
$ V_m(t) = V_m(t_"prev") + I(t) dot (t - t_"prev") + w_i $
])
#v(1em)

#serif-text()[ *Computational Cost:* Moderate. It replaces expensive exponentials with simple linear multiplication based on the time delta ($t - t_"prev"$).
*Temporal Decoding Capability:* Excellent. The first spike initiates the linear counter. Because earlier spikes have more time to multiply their slope against the passing time ticks, an early arrival will drive the potential to the threshold vastly faster than a late arrival of the exact same weight. It mathematically recognizes order.
*Synchronization:* This model requires strict global synchronization (a saccade clock). Without a global reset to zero out the slope $I(t)$ and potential $V_m(t)$, the linear ramp would continue to grow infinitely to the mathematical limits of the hardware. ]

#figure( include("figures/thresholdsensitive.typ"), caption: [Network architecture. The 28x28 images are flattened and passed through a Fully Connected Network. Both the ANN and SNN share this identical macroscopic topology.])

#figure(
kind: "algo",
caption: [Model C: Current-Accumulating Linear Ramp],
supplement: [Algorithm],
mono-text(pseudocode-list(hooks: .5em, indentation: 1em, booktabs: true)[
evaluate_model_C_Ramp(incoming_spikes, weights, threshold) -> integer:
  + sort(incoming_spikes, by_time)
  + v_m = 0.0
  + current_I = 0.0
  + t_prev = 0.0
  + firing_time = None
  +
  + for spike in incoming_spikes:
    + weight = weights[spike.source]
    + delta_t = spike.time - t_prev
    +
    + // Accumulate voltage based on historical current slope
    + v_m = v_m + (current_I \* delta_t) + weight
    + // Update current slope for future ticks
    + current_I = current_I + weight
    +
    + if v_m >= threshold:
      + firing_time = spike.time
      + break
    +
    + t_prev = spike.time
  +
  + return firing_time
]))

#v(1em)
#mini-header()[Model D: Fully Asynchronous State-Dependent Decay]

#serif-text()[ The final model represents the ideal target for purely event-driven neuromorphic systems: a model capable of temporal order decoding *without* requiring any global clocks, saccades, or synchronization pulses.

In this model, the exponential decay is strictly proportional to the current membrane potential, creating a self-regulating dynamical system. Because the decay drives the neuron toward a resting equilibrium continuously, the neuron naturally "forgets" ancient history. ]

#figure( kind: "eq", supplement: [Equation], caption: [Asynchronous State-Dependent Update], [
$ (dif V_m)\(dif t) = -(V_m)\(tau_m) + sum_j w_(i j) delta(t - t_j) $
])

#serif-text()[ *Computational Cost:* High, similar to Model B, due to the continuous calculation of state-dependent decay parameters.
*Temporal Decoding Capability:* High. The continuous state-dependent decay ensures that order matters significantly; a strong early spike establishes a baseline that subsequent spikes build upon, while isolated late spikes decay before reaching the threshold.
*Synchronization:* None. This model is 100% asynchronous. It requires no global reset signal and no defined simulation window. Between image presentations, the natural decay functions as a "soft reset," returning the neuron to a resting state dynamically. This allows the system to operate continuously on a raw, uninterrupted stream of event data. ]

#figure( include("figures/thresholdsensitive.typ"), caption: [Network architecture. The 28x28 images are flattened and passed through a Fully Connected Network. Both the ANN and SNN share this identical macroscopic topology.])

#figure(
kind: "algo",
caption: [Model D: Fully Asynchronous State-Dependent Decay],
supplement: [Algorithm],
mono-text(pseudocode-list(hooks: .5em, indentation: 1em, booktabs: true)[
evaluate_model_D_Async(continuous_spike_stream, weights, threshold, tau_m):
  + // Note: No global saccade reset or T_max boundary
  + v_m = 0.0
  + t_prev = 0.0
  +
  + for spike in continuous_spike_stream:
    + weight = weights[spike.source]
    + delta_t = spike.time - t_prev
    +
    + // Natural continuous decay acts as a soft-reset for ancient history
    + v_m = v_m \* exp(-delta_t / tau_m) + weight
    +
    + if v_m >= threshold:
      + emit_output_spike(spike.time)
      + v_m = 0.0 // Local reset only
      + current_I = 0.0
    +
    + t_prev = spike.time
]))

#v(1em)
=== Thresholding and Lateral Inhibition

#serif-text()[ Regardless of the chosen internal integration dynamics, all four models share identical discrete output behavior. Following the integration step, the neuron evaluates its firing condition: if $V_m(t) > V_"th"$, an output spike is emitted, and the internal state variables ($V_m$, and $I$ if applicable) are reset. Furthermore, if the neuron belongs to the output classification layer, its spike triggers a Winner-Takes-All (WTA) condition, instantly suppressing all competitors to finalize the classification. ]

#v(2em)
== Network Architecture

#serif-text()[ To facilitate a direct, one-to-one mapping of synaptic weights from the offline Artificial Neural Network (@ann) to the native Spiking Neural Network (@snn), both models must share an identical macroscopic topology. Therefore, this implementation utilizes a Fully Connected Network (@fcn), also known as a Multi-Layer Perceptron (@mlp), rather than a Convolutional Neural Network (@cnn).

While @cnn:pl are the standard baseline for vision tasks due to their spatial inductive biases, transferring convolutional kernels into a spiking substrate introduces significant mapping complexities—specifically the need to physically unroll and duplicate shared weights across the spiking array. An @fcn provides a straightforward, mathematically transparent architecture for cleanly evaluating direct weight transfer and @stdp without confounding architectural variables.

The network is structured as a shallow hierarchy to capture the primitive geometric features of the dataset. Let $N_l$ denote the number of neurons in layer $l$. The formal architecture is defined as follows: ]

#box-text()[
- *Input Layer ($L_0$):* The $28 times 28$ pixel grayscale images are flattened into a 1D vector, requiring $N_0 = 784$ input neurons.
- *Hidden Layer ($L_1$):* A fully connected intermediate layer consisting of $N_1 =$ [INSERT NUMBER OF HIDDEN NEURONS, e.g., 256] neurons. The synaptic connections are defined by the weight matrix $W^{(1)} in bb(R)^{N_1 times N_0}$.
- *Output Layer ($L_2$):* $N_2 = 10$ neurons, corresponding directly to the categorical digit classes (0 through 9). The connections from the hidden layer are defined by the weight matrix $W^{(2)} \in bb(R)^{N_2 times N_1}$.
]

#figure( include("figures/architechture.typ"), caption: [Network architecture. The 28x28 images are flattened and passed through a Fully Connected Network. Both the ANN and SNN share this identical macroscopic topology.])

#v(1em)
=== Weight Initialization and Biases

#serif-text()[ For the baseline @ann, the weight matrices are initialized using [INSERT INITIALIZATION METHOD, e.g., Kaiming/He normal initialization or standard PyTorch defaults] to ensure stable variance during the initial forward passes.

A critical architectural consideration in ANN-to-SNN conversion is the handling of bias vectors ($b^{(l)}$). While standard ANNs utilize biases to shift activation functions, representing a static continuous bias in a spiking network requires either injecting a constant background current into the neuron or actively modifying the neuron's firing threshold. To maintain pure, event-driven sparsity and isolate the performance of the synaptic weights, explicit bias terms were [CHOOSE ONE: omitted entirely from both architectures / converted to constant input currents in the SNN]. ]

#v(1em)
=== ANN-SNN Compatibility

#serif-text()[ Despite the macroscopic symmetry required for weight sharing, the microscopic dynamics of the two networks differ fundamentally. The offline @ann utilizes standard continuous activation functions (specifically ReLU) to compute smooth gradients during backpropagation.

In contrast, the @snn replaces these continuous functions with Integrate-and-Fire neurons governed by a strict voltage threshold. This simulates the biological "all-or-nothing" action potential, acting as a hard step function. Furthermore, the spiking architecture utilizes lateral inhibition at the output layer. This engenders a Winner-Takes-All (@wta) dynamic: as the network integrates evidence over time, the first output neuron to reach its threshold heavily suppresses its competitors, forcing a definitive categorical decision and actively filtering sub-threshold noise.

The snn uses pytorch matrix multiply
the matrix multiply becomes more of a vector multiply because of how the snn works

add pseudo code for the forward pass and training here]

#v(2em)
== Simulation Loop

#serif-text()[ The inference execution is governed by a discrete "saccade" simulation loop. The temporal window for a single MNIST image is strictly bounded to $T_{max} = 64$ discrete time steps (ticks).

To ensure that deep layer spikes have sufficient time to propagate through the network before the simulation window closes, the input encoding is temporally padded. Using the linear intensity-to-delay algorithm defined in Section 3.3.1, the maximum input delay is capped at $t=32$. Therefore, a pure black pixel spikes at $t=32$, leaving $32$ subsequent ticks for the hidden and output layers to integrate the final data.

At each time step $t \in [0, 64)$, the simulator executes a parallelized Window Integrator evaluation: ]

#box-text()[
1. *Event Detection:* The simulator queries all input and hidden neurons to identify which neurons are scheduled to spike at the exact current tick $t$.
2. *Parallel Integration:* Using optimized matrix-vector multiplication (`torch.mv`), the synaptic weights of the active neurons are concurrently summed into the membrane potentials ($V_m$) of the downstream layer.
3. *Thresholding & Single-Spike Constraint:* A neuron in the hidden layer ($N_1 = 128$) fires if its potential crosses a predefined high threshold ($V_"th_h"} = 200.0$). The output layer ($N_2 = 10$) operates on a lower threshold ($V_"th_o"} = 100.0$).
]

#serif-text()[ Crucially, to enforce the Time-to-First-Spike (@ttfs) paradigm, a strict single-spike constraint is applied. A boolean mask tracks the firing history of every neuron; once a neuron exceeds its threshold, it emits a spike, records its timestamp, and is permanently locked out from firing again for the remainder of the 64-tick saccade. This eliminates burst firing, guaranteeing maximum sparsity and preventing bus congestion. The final classification is determined by the output neuron that records the earliest spike timestamp. ]

#figure(include("figures/simulatorarch.typ"),caption:[Simulator architechture block diagram])

#serif-text()[ add pseduo code for the event loop ]

#v(2em)
== Training Methodologies

#serif-text()[ start with fully connected and then run strengthen connections that have "statistical significance" (not random / patterns) or is correlated to something (reward signal) at the end of an epoch prune connectionn making the network more effecient. This is similar to how the brain does it when sleeping

Do some math with references to the optimasation section

If the post neuron fires then we should strengthen the weights. Synapses gets grown at random this might be inneficient but it is highly parallelizable an idea could be to grow synapses with a kind of gravity where post synapses has a pulling effect if they are already strong

The encoding fit for images are a kind of population code where a pattern comes at the same time. The relative timing between patterns is used for strength earlier patterns are stronger. In a inhibitory network this allows for winner take all

learning with binary weights

learning with more steps of quantized or continous weights

Say we want to detect the pattern ABC and the pattern ABD. First of all if the order does not matter set all the weights equal. If the order does matter the weights determine the order. Now if a neuron learns pattern ABC so well that it learns to fire on only AB then it can fire faster. However if a second neuron wants to learn ABD then inhibition from the AB neuron prohibits it. A solution can be that if a neuron originally learned ABC but now fires on AB but stil has a strong weight on C it should remember this and if it fires on AB but then C does not arrive it should be like "oh, C did not show maybe I am wrong to fire early" eg. Decrease weights for A and B
It predicts!

A second way is to have a hierarchy with bypass. So one layer detects only AB then the next layer has bypass of the first layer and the second combining AB and C or D

A second problem is how to decode order. When do we start the decreasing timer, how fast, should it be in time or in amount of spikes, what to do with phase? The phase should correct itself. The weights need to be as presise as the timing of the spikes? Or we could make the neuron sensitivity proportional to its inverse potential and add leaking ]

#v(1em)
=== Weight Transfer

#serif-text()[ Direct one-to-one transfer of floating-point weights from an ANN to an SNN often results in catastrophic failure; the continuous activation scales do not naturally align with discrete spiking thresholds. Furthermore, physical neuromorphic hardware (such as crossbar arrays) imposes strict limitations on synaptic weight resolution, rarely supporting 32-bit floating-point numbers.

To emulate these hardware constraints and ensure threshold compatibility, the continuous weights of the trained ANN ($W_"ANN"$) undergo a pseudo-INT8 quantization process before being loaded into the SNN simulator. The weights are multiplied by a static scaling factor of $64$, rounded to the nearest integer, and clamped to an 8-bit signed integer range $[-128, 127]$: ]

#v(1em)
#figure( kind: "eq", supplement: [Equation], caption: [Weight Scaling and Quantization], [
$ W_"SNN"^{(l)} = max (-128, \min (127, "round"(W_"ANN"^(l) dot 64) ) ) $
])
#v(1em)

#serif-text()[ This transformation maps a standard continuous weight of $2.0$ to the maximum synaptic efficacy of $128$. By porting these quantized discrete weights ($W_1$ and $W_2$) directly to the GPU, the simulator strictly enforces the memory boundaries of physical neuromorphic silicon. ]

#figure(
kind:"algo",
caption: [Unsupervised local learning rule for induvidual neurons. Based on @stdp],
supplement: [Algorithm],
mono-text(pseudocode-list(hooks:.5em, indentation:1em, booktabs:true)[
+ start with a collection of neurons with arbitrary connections #h(1fr)
+ *if* a pre-synaptic neuron fires *then*
  + it has a chance to grow a synapse to a random post-synaptic neuron
+ *if* a post-synaptic neuron fires *then*
  + strengthen all connections to pre-synaptic neruons that fired before
  + remove connections to pre-synaptic neurons that did not fire or fired after
- 🛈  a neuron can be both pre-synaptic and post-synaptic
]))

#figure(
kind:"algo",
caption: [Growing rules for synapses],
supplement: [Algorithm],
mono-text(pseudocode-list(hooks:.5em, indentation:1em, booktabs:true)[
+ probability of growing a synapse is inversely\ proportional to the amount it already has #h(1fr)
+ earlier firings should get a better chance to grow synapses,\ although this is regulated by
  inhibitory action
]))

#v(1em)
=== TTFS STDP Inspired Learning Rule

#serif-text()[ In standard continuous networks, synaptic weights are updated via global error gradients. In the STDP paradigm, a synapse $w_(i j)$ connecting a pre-synaptic neuron $i$ to a post-synaptic neuron $j$ is updated based strictly on the temporal difference between their respective firing times.

Let $t_i$ denote the spike time of the input neuron, and $t_j$ denote the spike time of the output neuron. The relative arrival time is defined as $Delta t = t_j - t_i$. Because this architecture utilizes a strict Time-to-First-Spike (@ttfs) encoding where neurons fire at most once per saccade, the classical continuous STDP curve is adapted into a discrete, deterministic update rule: ]

#box-text()[
- *Long-Term Potentiation (LTP):* If $t_i lt t_j$, the pre-synaptic spike arrived before (or exactly at) the moment the post-synaptic neuron fired. This indicates causality. The synapse is strengthened, with the magnitude of the update decaying exponentially the further apart the spikes occurred.
- *Long-Term Depression (LTD):* If $t_i > t_j$, the pre-synaptic spike arrived after the post-synaptic neuron had already fired. The input was irrelevant to the decision, and the synapse is subsequently weakened.
- *Unused Synapses (Penalty):* If a pre-synaptic neuron fires but the post-synaptic neuron never fires during the saccade, a slight negative decay is applied to encourage forgetting of dead connections. ]

#figure( kind: "eq", supplement: [Equation], caption: [Additive STDP Weight Update], [
$ Delta w_{i j} =
cases(
  A_+ dot exp(-(t_j - t_i) \/ tau_+) & "if" space t_i lt t_j "(LTP)",
  -A_- dot exp(-(t_i - t_j) \/ tau_-) & "if" space t_i > t_j "(LTD)"
) $
])

#serif-text()[ To prevent runaway synaptic growth or catastrophic sign-flipping, the updated weights are strictly clamped to a positive physical range $w_(i j) in [0, W_"max"]$. Algorithm 2 details the exact logical implementation of this rule. ]

#figure(
kind: "algo",
caption: [The TTFS STDP Weight Update Function],
supplement: [Algorithm],
mono-text(pseudocode-list(hooks: .5em, indentation: 1em, booktabs: true)[
apply_stdp(t_pre, t_post, current_weight, A_plus, A_minus, tau, W_max):
  + // If post-synaptic neuron never fired, apply a small decay penalty
  + if t_post == -1:
    + return max(0.0, current_weight - (A_minus \* 0.1))
  +
  + // Calculate temporal difference
  + delta_t = t_post - t_pre
  +
  + // Pre-before-Post (Causality -> Potentiation)
  + if delta_t >= 0:
    + update = A_plus \* exp(-delta_t / tau)
    + new_weight = current_weight + update
  +
  + // Post-before-Pre (Late arrival -> Depression)
  + else:
    + update = A_minus \* exp(delta_t / tau) // delta_t is negative
    + new_weight = current_weight - update
  +
  + // Enforce physical hardware constraints
  + return clamp(new_weight, min=0.0, max=W_max)
]))

=== Training Loop

#serif-text()[ To execute unsupervised feature extraction on the MNIST dataset, the STDP rule is embedded within the saccade simulation loop. The network is initialized with randomized synaptic weights $W ~ cal(U)(0, 1)$.

During the training phase, an image is presented, and the network executes a forward pass. However, unlike the zero-shot inference phase, lateral inhibition (Winner-Takes-All) plays a critical structural role during training. When an output neuron fires, it aggressively inhibits its neighbors. This competitive dynamic forces different neurons to specialize in different geometric features; if one neuron learns to recognize a "loop," the WTA mechanism prevents other neurons from learning that exact same redundant feature.

At the conclusion of the 64-tick saccade, the simulator halts, compares the timestamp arrays of the hidden and output layers, and computes the STDP weight updates in parallel before the next image is presented. Algorithm 3 outlines this autonomous learning pipeline. ]

#figure(
kind: "algo",
caption: [Unsupervised SNN Training Pipeline],
supplement: [Algorithm],
mono-text(pseudocode-list(hooks: .5em, indentation: 1em, booktabs: true)[
train_unsupervised_epoch(dataset, W1, W2, T_max):
  + for image in dataset:
    + // 1. Execute Forward Saccade (with Lateral Inhibition)
    + spikes_pre = encode_to_ttfs(image)
    + spikes_post = simulate_saccade_wta(spikes_pre, W1, W2, T_max)
    +
    + // 2. Apply STDP to all synapses in parallel
    + for i in range(num_input_neurons):
      + for j in range(num_output_neurons):
        + t_pre = spikes_pre[i]
        + t_post = spikes_post[j]
        +
        + // Update weight using Algorithm 2
        + W1[i][j] = apply_stdp(t_pre, t_post, W1[i][j], ...)
    +
    + // 3. Normalize Weights (Homeostasis)
    + W1 = enforce_homeostatic_normalization(W1)
  +
  + return W1
]))

#v(2em)
== Evaluation Metrics

#serif-text()[ To rigorously validate the proposed Spiking Neural Network architectures, the evaluation framework must measure two distinct domains: *Classification Effectiveness* (accuracy and temporal decoding) and *Computational Efficiency* (resource usage and sparsity). Because the networks are evaluated across both supervised transfer and unsupervised native learning phases, specialized metrics are required for each. ]

#v(1em)
=== Effectiveness and Classification Performance

#serif-text()[ For Phase II (Zero-Shot Weight Transfer), evaluating classification performance is straightforward. The SNN is tasked with classifying the 10,000-image MNIST test set, and performance is measured using standard statistical metrics: ]

#box-text()[
- *Top-1 Accuracy:* The percentage of images where the first output neuron to fire (via the Winner-Takes-All mechanism) corresponds to the correct ground-truth label.
- *Confusion Matrices:* Utilized to identify specific class overlaps (e.g., distinguishing a '4' from a '9'). Because the SNN utilizes discrete temporal thresholds rather than continuous probabilities, confusion matrices help identify if certain geometric features are disproportionately lost during INT8 quantization.
]

#serif-text()[ However, evaluating Phase III (Unsupervised STDP) requires a different approach. Because the network learns without labels, the output neurons do not inherently map to digits 0 through 9; they map to arbitrary geometric clusters. To evaluate classification accuracy on an unsupervised model, we employ a *Post-Hoc Labeling* strategy: ]

#box-text()[
1. *Freezing:* After the STDP training phase concludes, synaptic plasticity is disabled (learning rate set to zero).
2. *Assignment:* A subset of the labeled training data is passed through the network. The simulator tracks the firing distribution of each output neuron. Each output neuron is permanently assigned the label of the digit class that most frequently triggered it to fire.
3. *Testing:* The standard 10,000-image test set is then passed through the network, and Top-1 accuracy is calculated using these newly assigned post-hoc labels.
]

#v(1em)
=== Computational Efficiency (Hardware Proxies)

#serif-text()[ While the primary goal of neuromorphic engineering is immense energy reduction, evaluating true physical power draw (Joules-per-inference) requires deploying the network on dedicated silicon (such as Intel Loihi or IBM TrueNorth). Because this thesis utilizes a PyTorch-based software simulator running on standard von Neumann hardware (GPUs), the simulator actually consumes *more* energy than a standard ANN due to the overhead of the temporal event loop.

To mathematically isolate and theorize the energy savings of the underlying *algorithm*, we abandon direct power measurements and utilize hardware-agnostic proxy metrics universally recognized in SNN literature: ]

#v(1em)
=== Sparsity and Synaptic Operations (SyOPs)

#serif-text()[ In a standard Artificial Neural Network, every forward pass forces every neuron to compute a dense Multiply-Accumulate (MAC) operation, regardless of the input data. The computational cost is fixed.

In a Spiking Neural Network, computation only occurs when a spike is emitted. By tracking the total number of spikes generated across the hidden layer ($N_"spikes"$) during a single 64-tick saccade, we can calculate the network's spatial and temporal sparsity.

Because Integrate-and-Fire neurons do not require multiplication (a spike simply adds its weight to the potential), MAC operations are replaced by simpler Synaptic Operations (SyOPs), which are purely arithmetic additions. The computational cost of a single inference step in the SNN is estimated as: ]

#figure( kind: "eq", supplement: [Equation], caption: [SNN Operational Cost Proxy], [
$ "Total SyOPs" = sum_{l=1}^{L} N_"spikes"^{(l)} dot F_"out"^{(l)} $
])

#serif-text()[ Where $F_"out"$ is the fan-out (number of outgoing synaptic connections) of the spiking neurons. By comparing the total SyOPs of the SNN against the fixed MAC count of the baseline ANN, we can derive a theoretical energy efficiency ratio. If the @ttfs encoding is highly sparse, the SyOP count should be orders of magnitude lower than the ANN MAC count. ]

#v(1em)
=== Temporal Latency Metrics

#serif-text()[ Finally, to evaluate the specific efficacy of the Time-to-First-Spike (@ttfs) encoding and the four distinct neuron models, we measure *Time-to-Decision Latency*.

Latency is defined as the exact simulation tick $t in [0, 64)$ at which the Winner-Takes-All output neuron fires. A lower average latency indicates a highly efficient temporal decoder that successfully prioritizes salient information, allowing the system to power down or reset earlier in the simulation window. Conversely, if a neuron model consistently requires the full 64 ticks to reach a decision, it fails to capitalize on the advantages of the temporal priority queue. ]


#v(1em)
=== Qualitative Evaluation via Latent Space Projection (t-SNE)

#serif-text()[ While post-hoc labeling provides a quantitative measure of classification accuracy, it inherently forces discrete semantic labels onto continuous geometric representations. To truly understand the internal representations learned by the unsupervised network, we must evaluate the topology of the hidden layer's latent space.

In the proposed architecture, the hidden layer consists of $N_1 = 128$ neurons. Each MNIST image produces a unique 128-dimensional spike-time signature. To qualitatively visualize how the network groups these high-dimensional signatures, we employ t-distributed Stochastic Neighbor Embedding (t-SNE). This non-linear dimensionality reduction technique projects the 128-dimensional manifold down to a 2D plane, preserving local proximities such that similar activation patterns appear as localized clusters.

This visualization provides a critical comparative tool between the supervised and unsupervised models: ]

#box-text()[
- *Supervised Latent Space (Phase II):* In the weight-transferred network, the continuous weights were explicitly optimized via cross-entropy loss to separate the 10 categorical digit classes. Consequently, the t-SNE projection is expected to reveal highly segregated clusters corresponding strictly to semantic labels (digits 0 through 9).
- *STDP Latent Space (Phase III):* In contrast, the native STDP algorithm possesses no semantic knowledge of the dataset. It operates purely as a temporal coincidence detector, strengthening synapses when overlapping geometric features (such as lines or curves) trigger coincident pre-synaptic spikes.
]

#serif-text()[ Therefore, the STDP t-SNE projection is hypothesized to cluster images based on morphological similarity rather than strict categorical class. Digits that share fundamental structural primitives (e.g., the top-loops of '8's and '9's, or the straight vertical edges of '1's and '7's) should form contiguous manifolds in the latent space. Demonstrating this geometric clustering via t-SNE will empirically validate that the local STDP rule successfully self-organized the network into a biologically plausible feature extractor. ]

// #figure( include("figures/tsne.typ"), caption: [Anticipated t-SNE projections. Left: Supervised clustering cleanly separating semantic classes. Right: Unsupervised STDP clustering based on morphological and structural coincidence.])

#v(2em)
== Experiment Setup and Evaluation Phases

#serif-text()[ To systematically evaluate the theoretical claims, computational trade-offs, and classification performance of the proposed Spiking Neural Network architectures, the experiment is structured into three distinct execution phases. This tri-phasic approach isolates the specific variables of temporal decoding, offline parameter transfer, and native online learning. ]

#v(1em)
=== Phase I: Synthetic Temporal Benchmarks

#serif-text()[ Before evaluating the networks on high-dimensional visual data, it is necessary to empirically validate the temporal decoding capabilities of the four neuron models (Simple IF, LIF, Linear Ramp, and Asynchronous Decay) established in Section 3.3.2.

The objective of this phase is to test whether the models can successfully differentiate between input patterns that possess identical spatial weights but differ purely in their temporal order of arrival. The models are subjected to synthetic micro-tasks consisting of controlled spike trains: ]

#box-text()[
- *Permutation Sensitivity Test:* A target neuron is presented with a sequence of spikes (e.g., Spike A at $t=1$, Spike B at $t=5$). The sequence is then reversed (Spike B at $t=1$, Spike A at $t=5$). The models are evaluated on their ability to reach the firing threshold for the correct sequence while remaining sub-threshold for the reversed sequence.
- *Coincidence Detection Test:* Spikes are injected with varying temporal spread (Inter-Spike Intervals). The models are evaluated on their ability to act as coincidence detectors, firing only when inputs arrive within a tight temporal window, demonstrating robustness against background noise.
]

#serif-text()[ This phase serves as a functional unit test, mathematically verifying which models are viable for strict Time-to-First-Spike (@ttfs) decoding before they are deployed on the MNIST dataset. ]

#v(1em)
=== Phase II: Zero-Shot Inference via Weight Transfer

#serif-text()[ The second phase evaluates the performance of the spiking models on real-world visual data using the offline ANN-to-SNN weight transfer methodology detailed in Section 3.4.

A baseline Artificial Neural Network (@ann) with the identical $784 arrow 128 arrow 10$ Fully Connected topology is trained on the MNIST dataset using standard Backpropagation and Cross-Entropy Loss until convergence. The learned floating-point weights are then quantized, scaled to the INT8 range $[-128, 127]$, and loaded directly into the SNN.

The SNN is then tasked with classifying the unseen 10,000-image MNIST test set without any further learning. During this phase, the four neuron models are benchmarked against the baseline @ann on three critical metrics: ]

#box-text()[
1. *Classification Accuracy:* The percentage of images correctly identified, measuring how much accuracy was lost during the quantization and temporal translation.
2. *Latency (Time-to-Decision):* The average number of simulation ticks required for the output layer to trigger the Winner-Takes-All mechanism, measuring the speed of the TTFS encoding.
3. *Sparsity (Energy Efficiency Proxy):* The total number of spikes generated across the hidden layer per inference. Models requiring fewer spikes to reach a decision demonstrate higher theoretical energy efficiency for neuromorphic hardware.
]

#v(1em)
=== Phase III: Native Unsupervised Learning (STDP)

#serif-text()[ While Phase II evaluates the inference capabilities of the hardware, it relies on global, offline backpropagation. To achieve true neuromorphic autonomy, the network must be capable of adapting to stimuli locally and dynamically.

The final phase discards the pre-trained ANN weights entirely. The SNN is initialized with randomized synaptic weights and subjected to the MNIST dataset utilizing a biologically inspired, unsupervised learning rule based on Spike-Timing-Dependent Plasticity (@stdp).

Because this phase is unsupervised, the network is not provided with the correct digit labels. Instead, the objective is to evaluate whether the local STDP rules, combined with lateral inhibition, can spontaneously self-organize the hidden layer neurons to cluster distinct geometric features (such as loops and edges) purely based on the temporal correlations of the input spikes. The performance of the four neuron models will be analyzed to determine which internal dynamics best support stable, unsupervised feature extraction. ]

#pagebreak()

= Results <results>

== Neruon Models

== MNIST With Copied Weights
#serif-text()[
#lorem(100)

#figure(image("figures/snnclasification.png"),caption:[Neural network before learing])

#figure(image("figures/snnweights.png"),caption:[Neural network before learing])

#figure(include("figures/network.typ"),caption:[Neural network during learning])

#figure(include("figures/network.typ"),caption:[Neural network after learning])

#lorem(100)

#lorem(100)
]

== MNIST With SNN Trained Weights

#figure(table(columns: 4,
  [0], [0], [1], [1],
  [1], [1], [1], [1]
),caption:[Number of operations])

#serif-text()[
#lorem(100)

#lorem(100)
#lorem(100)
#lorem(100)
#lorem(100)
#lorem(100)
#lorem(100)
#lorem(100)
]

#pagebreak()

= Discussion <discussion>

#serif-text()[ While the experimental results validate the core principles of Time-to-First-Spike (@ttfs) encoding and spiking integration, extrapolating these methods from controlled benchmarks to real-world deployment reveals significant engineering bottlenecks. This chapter critically analyzes the limitations observed during the implementation phase, specifically regarding data encoding, spatial representation, architectural translation, and hardware constraints. ]

#v(2em)
== Encoding Modalities and the Contrast Problem

#serif-text()[ In the engineered test environment, absolute pixel intensity was mapped directly to spike latency. For a highly controlled toy dataset like MNIST, this approach yields adequate performance, as the digits are isolated in high-contrast, low-resolution environments. However, absolute luminance is a notoriously brittle feature for real-world computer vision.

Robust biological and artificial vision systems rely on local contrast (the relative difference between adjacent pixels) to determine object boundaries, as contrast remains invariant under shifting global illumination. Attempting to compute true, normalized contrast directly within a @ttfs spiking network presents a severe temporal bottleneck. To accurately assess relative darkness in a purely temporal code, the downstream neurons must wait for the slowest (darkest) signals to arrive, effectively nullifying the high-speed advantages of the @ttfs priority queue.

Consequently, attempting to calculate contrast natively within the SNN layers is computationally wasteful. The optimal solution is to offload this processing to the sensory periphery. Dedicated neuromorphic sensors, such as Dynamic Vision Sensors (DVS), natively output logarithmic intensity differences. Because the difference in log-space mathematically corresponds to a true contrast ratio independent of absolute luminance, passing this pre-encoded contrast data directly into the @ttfs network avoids the temporal delay problem entirely. ]

#v(2em)
== Representing Space and Dimensionality

#serif-text()[ Encoding precise spatial coordinates using a pure @ttfs scheme proved inherently difficult. In biological visual and motor cortices, spatial locations are often represented via orthogonal population codes, where specific populations activate to indicate direction or position. However, these biological populations largely utilize rate coding; the "intensity" or certainty of a spatial position translates naturally into a higher firing frequency.

Conversely, @ttfs is highly optimized for rapid, categorical decision-making (e.g., triggering a Winner-Takes-All classification), but struggles to map continuous numerical or spatial values without complex phase-ambiguity resolution. A potential architectural workaround is the implementation of hierarchical "space cells." Rather than mapping a massive $32 times 32$ grid to individual temporal delays, the space could be subdivided into localized grids (e.g., overlapping $8 times 8$ populations). This reduces the dimensionality of the temporal encoding, though it introduces resolution artifacts at the boundaries of the receptive fields. ]

#v(2em)
== Architectural Translation (ANN to SNN)

#serif-text()[ The zero-shot inference phase highlighted the frictions of translating classical architectures to spiking substrates. While the Fully Connected topology provided a transparent baseline, mapping state-of-the-art Convolutional Neural Networks (@cnn:pl) is decidedly not a one-to-one process.

Classical continuous networks heavily utilize negative weights, static biases, and mathematical operations like Max Pooling. Neuromorphic systems handle these concepts fundamentally differently. An SNN replaces mathematical pooling with dynamic lateral inhibition, and negative continuous weights must be modeled via discrete inhibitory spike trains. These architectural mismatches dictate that SNNs cannot simply act as "drop-in" replacements for deep learning models; they require network topologies natively designed for event-driven dynamics. ]

#v(2em)
== Plasticity Dynamics and Forgetting

#serif-text()[ During the evaluation of unsupervised learning, a fundamental tension emerged between the network's learning velocity and its stability. Unsupervised, local learning rules like Spike-Timing-Dependent Plasticity (@stdp) require aggressive hyperparameter tuning. If the plasticity rate is too high, the network adapts quickly but suffers from rapid catastrophic forgetting—overwriting previously learned geometric features when presented with novel stimuli. Conversely, if the plasticity is too low, the network fails to converge on meaningful representations within an acceptable time frame. Designing homeostatic mechanisms to stabilize this plasticity remains a core challenge for native learning. ]

#v(1em)
=== The Sparsity Paradox and GPU Simulation Overhead

#serif-text()[ A core theoretical advantage of the proposed @ttfs network is its extreme spatial and temporal sparsity. By design, only a small fraction of neurons emit spikes, mathematically reducing the dense Multiply-Accumulate (MAC) operations of a standard ANN to a minimal set of Synaptic Operations (SyOPs).

However, evaluating this sparse algorithm on conventional Graphics Processing Units (GPUs) introduces a significant "Sparsity Paradox." Standard deep learning frameworks (such as PyTorch) and standard GPU architectures are heavily optimized for dense, contiguous matrix-matrix multiplications via SIMD (Single Instruction, Multiple Data) execution. In the SNN simulation loop, sparsity is enforced via boolean masking (multiplying inactive neuron outputs by zero). While this successfully zeroes out the potential, the GPU ALUs typically still execute the underlying floating-point multiplication cycle.

Consequently, the hardware does not naturally skip the computation for quiescent neurons unless specialized, unstructured sparse tensor kernels are deployed. When combined with the overhead of maintaining the temporal loop (the "saccade" ticks), the SNN simulator ironically consumes more absolute clock cycles and physical power on a GPU than the continuous @ann baseline.

This paradox highlights that true energy efficiency cannot be realized via matrix-masking on von Neumann hardware. Realizing the calculated SyOP savings requires deployment on native event-driven Neuromorphic ASICs (Application-Specific Integrated Circuits). These chips discard the matrix-multiplication paradigm entirely, utilizing Address Event Representation (AER) to asynchronously route data packets only when a spike physically occurs, reducing idle power draw to near zero. ]

#v(1em)
=== Mitigation via Synaptic Pruning

#serif-text()[ While dynamic activation sparsity struggles on GPUs, structural sparsity offers a viable software-level mitigation. Future iterations of this work could introduce aggressive Synaptic Pruning, particularly following the unsupervised @stdp learning phase.

Because @stdp naturally depresses irrelevant synapses toward zero, a thresholding function could permanently sever these connections, converting the dense weight matrices ($W^{(1)}$, $W^{(2)}$) into highly sparse structures. By utilizing block-sparse tensor formats (e.g., Compressed Sparse Row), both software simulators and specialized sparse-accelerator chips can mathematically bypass the zero-weights, yielding physical reductions in memory bandwidth and computation even before transitioning to pure neuromorphic silicon. ]

#v(2em)
== The Physical Hardware Gap

#serif-text()[ Ultimately, the theoretical energy efficiency of neuromorphic algorithms is bounded by physical hardware. The connection density of the biological mammalian cortex vastly exceeds the routing capabilities of modern CMOS (Complementary Metal-Oxide-Semiconductor) fabrication.

While the software simulations in this thesis demonstrate the algorithmic viability of sparse processing, achieving true biological efficiency requires novel materials. Non-volatile memory technologies, such as memristors and spintronics, offer the ability to physically colocate extreme-density analog weights with logic gates, perfectly mirroring biological synapses. Until these exotic substrates become commercially viable, the near-term future of applied neuromorphic computing lies in massively parallel, asynchronous digital ASICs designed using standard CMOS, serving as a transitional bridge to fully analog systems. ]

#v(2em)
== Future Work <future_work>

#serif-text()[ The findings and limitations discussed in this thesis present several promising avenues for future research in neuromorphic engineering: ]

#box-text()[
- *Event-Based Dataset Validation:* Transitioning the experimental framework from static images (MNIST) to native temporal datasets, such as Neuromorphic-MNIST (N-MNIST) or DVS Gesture datasets, to fully exploit the asynchronous dynamics of the evaluated neuron models.
- *Surrogate Gradient Integration:* While this work focused on direct weight transfer and native @stdp, future implementations should evaluate the efficacy of Surrogate Gradient Descent, combining the optimization power of backpropagation with the inference efficiency of spiking dynamics.
- *Hardware Deployment:* Porting the verified TTFS algorithms and current-accumulating neuron models from GPU simulation onto dedicated neuromorphic silicon (e.g., Intel Loihi) to empirically measure true joule-per-inference energy consumption against classical von Neumann baselines.
]


#pagebreak()

= Conclusion <conclusion>

#serif-text()[
#lorem(100)

#lorem(100)

#lorem(100)

#lorem(100)

#lorem(100)

#lorem(100)
]

#pagebreak()

#set text(weight: "medium")
#bibliography("references.bib")
