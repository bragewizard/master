#import "@preview/droplet:0.3.1": dropcap
#import "@preview/wordometer:0.1.5": word-count, total-words
#import "@preview/lovelace:0.3.0": pseudocode-list
#import "@preview/glossarium:0.5.10": make-glossary, register-glossary, print-glossary, gls, glspl
#import "frontpage/frontpage.typ": cover, colors
#import "glossary.typ": entry-list
#import "style.typ": style, serif-text, mono-text, box-text, mini-header

#show: style
#show: make-glossary
#show: word-count

#register-glossary(entry-list)
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
#h(1fr) Wordcount = #total-words
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

  #serif-text()[ The development of modern Deep Learning has achieved unprecedented performance across various domains, yet it remains fundamentally bottlenecked by the energy and memory inefficiencies of the von Neumann architecture. To address these limitations, this thesis investigates Neuromorphic Computing with @snn:pl as a biologically plausible, highly energy-efficient alternative. By shifting from synchronous, continuous-value matrix multiplications to asynchronous, event-driven sparse computations, neuromorphic systems emulate the physical principles of the biological brain.

This work explores the implementation of these principles on standard CPU/GPU hardware. Two primary methodologies are developed and evaluated on visual classification tasks: (1) an inference-optimized @snn that translates weights from a conventionally trained @fcn using @ttfs temporal encoding, and (2) an unsupervised, biologically inspired learning simulator incorporating structural plasticity (dynamic synaptogenesis and pruning). The results demonstrate the viability of temporal coding and local learning rules in extracting meaningful features from visual stimuli, highlighting the potential of neuromorphic algorithms to drastically reduce the computational footprint of artificial intelligence.]]
]]

#pagebreak()

#align(center)[
#block(width:100%)[
  #align(left)[
  #text(weight:"semibold",size:16pt,[ACKNOWLEDGEMENTS & DECLARATIONS])

  #serif-text()[
I would like to thank my supervisors and the very kind and helpful comunity at the ROBIN and NANO research groups at the departemnt of informatics.

#v(1em)
#mini-header()[Declaration of the use of generative artificial intelligence]

In this scientific work, generative @ai has been used. All data and personal information have been processed in accordance with the University of Oslo's regulations, and I, as the author of the document, take full responsibility for its content, claims, and references. An overview of the use of generative @ai is provided below.

The service Gemini, developed by Google, has been used to improve the content of the thesis. Sections of text like a subsection, paragraph or source code for figures was given to the model along with prompts to make the language free of errors and more profesional.  The text was then iterated a few times through the model where new prompts were used to get the correct structure and flow of the text. In the case for figures the model was used to speed up the development of the figures with the model generating numbers for positioning elements. The final result was cut out, fact-checked, and partly rewritten by the author.
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
print-glossary(entry-list, disable-back-references: true)
}

#pagebreak()

= Introduction <s.intro>

#serif-text()[ The development of intelligent machines is a significant objective in modern science and engineering. While the concept has historical roots in philosophy and early automata, the field has transitioned from speculative theory to practical application. Currently, artificial intelligence is central to technological and economic development. Understanding the mechanisms of intelligence and reproducing them in synthetic systems offers the potential for improved analysis of biological minds and the creation of tools for applications ranging from personalized medicine to automated scientific discovery.

In recent years, great strides has been made towards this goal. Deep Learning, which utilizes multilayered neural networks, has exceeded previous performance benchmarks. These systems have demonstrated high proficiency in tasks previously limited to human capability. For example, AlphaFold has addressed complex problems in protein folding @Placeholder, reinforcement learning agents have mastered the complexity of games such as Go @Placeholder, and Large Language Models have shown capabilities in text generation that approach human fluency. Consequently, @ai is increasingly viewed as a general-purpose technology that may influence societal infrastructure.

However, despite these advances, there are significant limitations to the current approach. The success of modern deep learning relies heavily on scaling, which involves increasing data volume and computational power. This strategy is approaching physical and economic boundaries. Training state-of-the-art models consumes substantial energy and results in a large carbon footprint @Placeholder. Although specialized hardware allows for more efficient computations, the underlying architecture and algorithms imposes an intrinsic limit on scalability independent of the underlying hardware. Furthermore, the requirement for massive datasets presents challenges in sourcing and curation. Additionally, evidence suggests that this scaling approach yields diminishing returns. Models often function as statistical correlation engines; they lack common-sense reasoning, struggle with out-of-distribution generalization, and are prone to brittle failure modes @Placeholder.

These limitations are evident when comparing artificial systems to biological intelligence. The human brain demonstrates that high-level intelligence is possible without massive energy consumption or dataset sizes. The brain operates on approximately 20 watts @Placeholder. With this limited energy budget, it manages biological functions, processes real-time multi-sensory data, and performs abstract reasoning. In contrast, deep learning models require GPU clusters with significantly higher power requirements to match a fraction of these capabilities. There is also a discrepancy in learning efficiency. Deep learning models are sample-inefficient, often requiring vast numbers of examples to learn a representation. Biological systems, however, are capable of "one-shot" or "few-shot" learning and can acquire new information without catastrophic forgetting. This suggests the inefficiency of current @ai is a paradigmatic issue rather than just an engineering problem.

The proposed direction for addressing these issues involves biological inspiration in both hardware and algorithm design, specifically Neuromorphic Computing. This field attempts to engineer computer architectures that mimic the biological structure of the nervous system. Unlike traditional @ai, which runs as software on general-purpose hardware, neuromorphic engineering aims to align the algorithm with the physical substrate. It moves away from clock-driven processing toward asynchronous, event-driven systems. In this paradigm, information is encoded as sparse, discrete events or "@spike:pl". Similar to biological neurons, a neuromorphic processor consumes minimal energy when inactive, processing information only when triggered. This approach is being pursued by both industrial groups, such as Intel’s Loihi @Placeholder, and academic projects like SpiNNaker @Placeholder. These systems represent a shift from calculation-based machines to those capable of real-time adaptation.

Although neuromorphic systems achieve optimal performance on co-designed platforms---where the algorithm is embedded directly into the hardware---there is significant value in executing neuromorphic algorithms on traditional von Neumann architectures. In this thesis, we explore biologically inspired algorithms deployed on traditional CPU and GPU hardware. We examine how event-driven, biologically plausible computation can address limitations in scalability, data efficiency, and energy consumption, even when simulated on standard processors. We present approaches for efficient information coding and learning algorithms inspired by neural mechanisms. Concretely, this thesis aims to: ]

#box-text()[
- *Investigate Sparse Information Flow*: Explore how information can be encoded and processed using sparse, asynchronous events (@spike:pl) within a neural network.

- *Develop Biologically Inspired Learning*: Explore and evaluate learning algorithms that are suitable for such networks, adhering to the constraints of locality and efficiency.
]

#serif-text()[ The succeeding chapter lays the historical and theoretical foundation, covering early neuroscience and the development of artificial neural networks based on simple models of the brain. Following this, we review relevant modern neuroscience literature, extracting key concepts that will inform the methodology. We also provide consise overview on machine learning concepts and frameworks. Finally, we detail the implementation of these principles in a neuromorphic context and evaluate their performance against standard benchmarks. ]

#pagebreak()

= History & Developments <s.history>

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
== The Perceptron <s.perceptron>

#serif-text()[ In 1957, Frank Rosenblatt advanced these theoretical concepts by engineering the Perceptron. The "Mark I Perceptron" was a hardware implementation of the neural model, distinguished by a crucial innovation: a weight-adjustment mechanism based on Hebbian principles. Rosenblatt introduced the perceptron learning rule, an iterative algorithm capable of minimizing error automatically. The system processed an input pattern (e.g., a pixelated character) and produced a binary classification. When the output deviated from the target, the algorithm adjusted the weights proportional to the error: strengthening connections that should have contributed to a correct firing and weakening those that led to false positives. ]

#figure(include("figures/perceptron.typ"),caption:[The perceptron model. Inputs $x_i$ are multiplied by weights $w_i$ and summed. If the linear combination $sum x_i w_i$ exceeds the bias $b$, the neuron activates. ])

#serif-text()[ Consequently, the Perceptron was capable of converging on a solution for any problem where the data was linearly separable. This success generated significant enthusiasm, with contemporary reports suggesting that such machines would soon mimic human consciousness @Placeholder.

These expectations were abruptly tempered by theoretical limitations. In 1969, Marvin Minsky and Seymour Papert published Perceptrons, a rigorous mathematical analysis of the architecture. They demonstrated that a single-layer perceptron is fundamentally a linear classifier. While capable of learning operations like AND or OR, it is mathematically incapable of solving the XOR (Exclusive OR) problem. In the XOR case, the classes cannot be separated by a single hyperplane. This proof highlighted a severe boundary on the utility of single-layer networks for complex, non-linear tasks. ]

#figure(include("figures/gates.typ"),caption:[The XOR problem. Unlike AND/OR, the data points for XOR cannot be separated by a single linear boundary.])

#serif-text()[ The publication of Perceptrons coincided with a significant reduction in neural network research funding, a period retrospectively termed the "First AI Winter". It is worth noting that Minsky and Papert acknowledged that a @mlp, a network stacking multiple layers of neurons, could theoretically solve the XOR problem by creating complex, non-linear decision boundaries.

However, a critical algorithmic gap remained: the "credit assignment problem". While researchers knew that hidden layers could represent complex features, there was no known method to propagate error signals back through the layers to adjust the weights of hidden neurons effectively. Rosenblatt’s rule was mathematically valid only for the output layer. The field remained stagnant until a method for training multi-layer networks could be formalized. ]

#v(2em)
== Deep Learning <s.deeplearningintro>

#serif-text()[ The critique presented by Minsky and Papert precipitated a contraction in funding; despite this, theoretical inquiry persisted. It was widely hypothesized that the limitations of the single perceptron could be overcome by a @mlp. By organizing neurons (single perceptrons) into hierarchical layers, the network could theoretically perform successive non-linear transformations on the input space, enabling the formation of complex decision boundaries. The primary impediment was not the architecture itself, but the absence of a viable learning algorithm.

In a single-layer perceptron, error attribution is immediate: if the output deviates from the target, the error is directly derived from the weights of the output layer. However, in a multi-layer architecture, quantifying the contribution of a specific neuron within the "hidden" layers to the final output error presents a significant challenge. This is formally known as the Credit Assignment Problem @Placeholder, and it remained the central theoretical obstacle for over a decade. ]

#figure(include("figures/network.typ"),caption:[A @mlp. By inserting "hidden layers" between input and output, the network can approximate non-linear functions such as XOR. The historical challenge lay in deriving a method to train these intermediate layers.])

#serif-text()[ The solution to this theoretical impasse was popularized in 1986 by Rumelhart, Hinton, and Williams in their seminal paper _Learning representations by back-propagating errors_ @Placeholder. They demonstrated that the Chain Rule of calculus could be applied recursively to propagate the error signal from the output layer backwards through the hidden layers. This algorithm, known as Backpropagation, allowed the network to calculate the gradient of the loss function with respect to every weight in the system. Effectively, it provided a mathematical method to tell each hidden neuron exactly how much it contributed to the total error, finally solving the credit assignment problem.

Unlike Hebbian plasticity, which is local and biological, Backpropagation relies on global error signals and precise backward data flow—mechanisms effectively absent in organic tissue. Consequently, the field of @ann effectively decoupled from neuroscience. It transitioned into a branch of engineering and applied mathematics, prioritizing statistical optimization over biological realism. Paradoxically, it was this abandonment of biological fidelity that enabled the rapid scaling and performance breakthroughs that followed. ]

#v(1em)
=== Achievements

#serif-text()[ With the training mechanism solved, the field exploded. The combination of Backpropagation, massive datasets, and GPU hardware led to a "Cambrian Explosion" of neural architectures, each solving domains previously thought impossible for computers.

The revolution began in earnest with computer vision. @cnn:pl, such as AlexNet (2012) @Placeholder and later ResNet @Placeholder, introduced the idea of learning hierarchical features---detecting edges, then shapes, then objects---much like the human visual cortex. This allowed machines to classify images with superhuman accuracy.

Soon after, the focus shifted to sequence data. @rnn:pl and @lstm architectures gave machines a short-term memory, enabling breakthroughs in speech recognition and machine translation. However, the true paradigm shift occurred with the introduction of the Transformer architecture in 2017. By utilizing an "attention mechanism" to parallelize the processing of language, Transformers allowed for the training of massive @llm:pl like the @gpt.

These techniques have even transcended media generation. Deep Learning has solved fundamental scientific problems; notably, DeepMind's AlphaFold utilized these architectures to predict the 3D structure of proteins from their amino acid sequences, a 50-year-old grand challenge in biology @Placeholder. ]

#v(1em)
=== Shortcomings

#serif-text()[ Deep learning's reliance on computational scaling masks fundamental inefficiencies in both its hardware implementation and underlying algorithms. By simulating biological concepts on digital architectures not designed for them, the current paradigm is approaching physical and economic limits.

A primary limitation is the Von Neumann architecture, which physically separates processing units from memory. Deep neural networks, defined by massive matrices of synaptic weights, necessitate constant data transfer. For every inference step, billions of parameters must be fetched from off-chip DRAM, processed, and written back. This creates a severe memory bottleneck where system performance is bounded by bandwidth rather than processing speed @Placeholder.

Consequently, the energy cost of moving data significantly exceeds the cost of computation itself. Retrieving a single byte from DRAM consumes approximately three orders of magnitude more energy than performing a floating-point operation @Placeholder. Compounding this hardware friction, the dense matrix multiplications required for training scale quadratically with network size, making the pursuit of trillion-parameter models increasingly unsustainable.

Furthermore, the optimization algorithms driving this scale are fundamentally incompatible with physical biological systems. Backpropagation, while mathematically elegant, relies on a global error signal and suffers from the "weight transport problem"—the requirement that the backward pass utilizes the exact same synaptic weights as the forward pass. In organic tissue, synapses are unidirectional, and there is no known mechanism for a neuron to access the exact weight of a downstream synapse to calculate a gradient.

While a detailed technical analysis of these inefficiencies is presented in @s.technicaldetailsofml, the central issue is clear: modern AI prioritizes statistical optimization over physical realism. Overcoming the limitations of the Von Neumann bottleneck and backpropagation requires a paradigm shift toward architectures that inherently co-locate memory and computation. ]

#v(2em)
== Birth Of Neuromorphic <s.birthneuromorphic>

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

= Biological Principles <s.biological>

#serif-text()[ The biological brain remains the gold standard for energy-efficient, robust, and adaptive computation. Since the establishment of the Neuron Doctrine, modern neuroscience has uncovered the specific physical mechanisms that underpin this efficiency. To engineer systems that truly rival biological performance, we must transcend the "spherical cow" abstractions of early cybernetics. We cannot simply mimic the brain's output; we must emulate its internal dynamics. This requires viewing the neuron not as a static summing unit, but as it functions in reality: a complex, time-dependent, and event-driven processor.

This section provides a mechanistic overview of the nervous system, translating biological observations into the computational primitives required for neuromorphic engineering. It explores the structural hierarchy of the neuron, the physics of the action potential, and the mathematical models used to capture these dynamics in silicon. ]

#v(2em)
== Neuron Structure & Function <s.neuronstructure>

#serif-text()[ In @s.history we established the neuron as the fundamental computational unit of the brain. While it shares standard cellular machinery like mitochondria and a nucleus with other cells, it is morphologically specialized for information transmission. A neuron consists of three functional zones: ]

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

Computation occurs through the modulation of this voltage by competing synaptic inputs. Excitatory inputs cause ion channels to open, allowing positive ions to influx; this reduces the negative charge (depolarization) and pushes the potential toward the firing threshold. Conversely, inhibitory inputs activate channels for negative ions (like Chloride, $"Cl"^-$), driving the potential away from the threshold (hyperpolarization). The soma integrates these opposing push and pull signals. If the aggregate membrane potential surpasses a critical threshold (approximately $-55$ mV), the system undergoes a bifurcating phase transition. Voltage-gated sodium channels cascade open, triggering an @actionpotential—a rapid, non-linear depolarization spike that propagates down the axon. This mechanism is governed by the "all-or-nothing" principle: the output is discrete and binary, effectively filtering out sub-threshold noise. ]

#serif-text()[ Immediately following a spike, the neuron enters a "refractory period" during which ion gradients are restored, imposing a hard limit on the maximum firing frequency and ensuring the temporal separation of events.

It is important to acknowledge that the biological brain exhibits significant cellular diversity beyond this idealized model. The nervous system contains non-neuronal cells known as "glia", which provide structural support and manage energy delivery, though they are generally not considered direct participants in fast information transmission. Additionally, while the vast majority of cortical neurons communicate via uniform action potentials (spikes), certain sensory neurons utilize "graded potentials", where the signal amplitude varies continuously. However, as spiking neurons represent the dominant computational paradigm for information processing in the cortex, this thesis focuses exclusively on the spiking model as the basis for neuromorphic emulation. ]

#v(2em)
== Action Potential & Spike Trains <s.actionpotential>

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
== Neuron Models <s.neuronmodels>

#serif-text()[ In the quest to simulate the brain, there exists a fundamental trade-off between biological realism and computational efficiency. At the high end of the spectrum lie conductance-based models, most notably the Hodgkin-Huxley model. This formalism describes the neuron not as a simple computational unit, but as an electrical circuit with variable resistors representing the precise, non-linear opening and closing dynamics of specific ion channels (sodium, potassium, leak) @Placeholder.

Large-scale initiatives, such as the Blue Brain Project, utilize even more granular "multi-compartment" models. These simulations treat the neuron as a complex 3D structure, discretizing the dendritic arbor and axon into hundreds of segments to model how current flows through the specific morphology of the cell @Placeholder. While invaluable for pharmacological research, these models are computationally prohibitive for large-scale neuromorphic engineering. Simulating a mere second of biological time for a small network using these equations requires supercomputing resources.

To build practical, scalable neuromorphic hardware, we must abstract these biophysical details into a phenomenological model. We seek a mathematical framework that captures the essential computational properties—integration, leakage, and thresholding—without simulating the underlying molecular physics. ]

#v(1em)
=== The Leaky Integrate-and-Fire (LIF) Model <s.biolif>

#serif-text()[ The standard approximation used in neuromorphic engineering is the @lif model. This framework aligns perfectly with the "point process" abstraction established in the previous section, as it treats action potentials as instantaneous, discrete events. Its state is defined by a single scalar variable, the membrane potential $u(t)$. The sub-threshold dynamics are governed by a linear differential equation analogous to a simple $R C$ (Resistor-Capacitor) circuit. We implement this model as the computational baseline for our visual classification tasks in @s.decoding (Model B). ]

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
=== The Generalized (Adaptive) LIF Model <s.glif>

#serif-text()[ While the standard @lif model is computationally efficient, its one-dimensional nature limits it primarily to tonic spiking (regular firing under constant input). It struggles to replicate the complex, non-linear behaviors observed in the cortex, such as bursting (clusters of rapid spikes) or spike-frequency adaptation (slowing down after sustained activity).

To capture these dynamics without reverting to the computationally heavy Hodgkin-Huxley equations, we employ the @glif model. This extends the system by introducing a second state variable, $w(t)$, representing cellular adaptation. ]

#figure( kind: "eq", supplement: [Equation], caption: [The Adaptive GLIF system. The adaptation variable $w$ provides negative feedback, enabling complex dynamics like bursting and adaptation.], [
$ tau_m (dif u)/(dif t) &= -(u - u_"rest") + R I(t) - w \
  tau_w (dif w)/(dif t) &= a(u - u_"rest") - w $
])<glif_eq>

#serif-text()[ In this coupled system, $w$ provides a negative feedback loop. Every time the neuron spikes, $w$ increments by a constant $b$, acting as a physiological drag on the membrane potential. By adjusting the coupling parameters between $u$ and $w$, this two-dimensional system can be tuned to emulate the full spectrum of biological firing patterns.

It is natural to question whether such a mathematically reduced model can genuinely capture the behavior of biological neurons. While the @glif model discards the specific ionic mechanisms of the Hodgkin-Huxley equations, empirical validation demonstrates that it retains superior computational dynamics for large-scale modeling.

In the 2008 _Quantitative Single-Neuron Modeling Competition_ @Placeholder organized by the INCF, phenomenological models like the Generalized @lif (specifically the Adaptive Exponential Integrate-and-Fire) unexpectedly outperformed highly detailed biophysical models in predicting the precise spike times of real cortical neurons.

This counter-intuitive success is due to parameter sensitivity. Complex conductance-based models have dozens of unobservable parameters that are difficult to tune. In contrast, the @glif model captures the "net effect" of these mechanisms using macroscopic parameters that can be robustly fitted to data. As demonstrated by Izhikevich (2003), this simple system of two differential equations is capable of reproducing all known firing patterns observed in the mammalian cortex @Placeholder. We leverage this state-dependent adaptation to implement Model D (@s.decoding), evaluating its effectiveness in temporal sequence decoding. ]

#figure(include("figures/izhikevichpatterns.typ"), caption:[The Generalized @lif model is capable of reproducing the diverse firing patterns of biological cortical neurons, as categorized by Izhikevich (2003) @Placeholder.], placement: auto)

#serif-text()[ Consequently, for the purpose of neuromorphic engineering, the @glif model represents the optimal trade-off between biological fidelity and computational efficiency. ]

#v(2em)
== Neural Coding <s.neuralcoding>

#serif-text()[ In classical digital computing, information is represented by combining bits into richer structures, such as floating-point or integer numbers. For instance, the luminance of a pixel is typically stored as a discrete 8-bit or 32-bit integer. Conversely, analog electronics represent values as continuous currents or voltages, offering infinite resolution within the dynamic range of the hardware. ]

#figure(include("figures/digitalanalogrepresentation.typ"), caption:[ Digital left analog right representation])

#serif-text()[ The biological brain occupies a unique middle ground. While neurons operate using analog membrane potentials, their communication output—the action potential—is discrete and binary. As established in @s.actionpotential, the waveform of a spike is stereotypical; it looks like a "digital bit" in amplitude. However, unlike a digital computer which is synchronized to a rigid clock, these spikes occur in continuous time. Therefore, the information in the nervous system is not stored in the shape of the signal, but in the structure of the spike train itself.

Deciphering the "Neural Code"—the set of rules by which sensory stimuli are translated into these spike sequences—remains one of the central challenges in neuroscience. Currently, several coding schemes are hypothesized to coexist, each offering different trade-offs between latency, information density, and robustness. ]

#v(1em)
=== Rate Coding <s.ratecoding>

#serif-text()[ The most traditional interpretation of neural activity is rate coding. In this paradigm, information is conveyed by the mean firing frequency of a neuron over a specific temporal window. A strong stimulus (e.g., high pressure on skin) elicits a high firing rate, while a weak stimulus results in sparse activity.

This model effectively treats the neuron as an Analog-to-Digital converter where the precise timing of individual spikes is treated as noise; only the average count carries the signal. While rate coding is robust and easily observed in motor neurons, it suffers from a fundamental latency barrier. To estimate a rate with reasonable precision, the post-synaptic neuron must integrate spikes over a significant duration (tens or hundreds of milliseconds). This contradicts the rapid reaction times (often $<100$ ms) observed in biological agents, suggesting that rate coding alone cannot account for time-critical processing. ]

#figure(include("figures/rateencoding.typ"), caption:[Rate Coding: The stimulus intensity is encoded in the frequency of the spike train. Stronger stimuli elicit more spikes per second.])

#v(1em)
=== Temporal Coding <s.temporalcoding>

#serif-text()[ To explain the speed of biological processing, neuromorphic engineering emphasizes temporal coding. In this regime, the precise timing of a spike carries significant information. A primary example is @ttfs coding.

In a @ttfs scheme, the intensity of a stimulus is inversely mapped to the latency of the response relative to a stimulus onset. A stronger input causes the neuron to integrate and cross the threshold faster, firing earlier than neurons receiving weak inputs. This shifts the computational model from counting spikes to a "race" between spikes.

In a network utilizing lateral inhibition (@wta), the first neuron to fire inhibits its neighbors, allowing a decision to be made as soon as the first meaningful bit of data arrives. This eliminates the need to wait for a time window to close, drastically reducing latency. Furthermore, since @ttfs coding prioritizes the strongest signals, it acts as a natural filter: the most prominent features arrive first, allowing the system to process signal over noise. This @ttfs scheme forms the primary encoding modality for our MNIST experiments, as detailed in @s.encoding. ]

#figure(include("figures/temporalcoding.typ"), caption:[Temporal Codiing (@ttfs): Stimulus intensity is encoded in the latency of the response. Stronger inputs ($I_1$) trigger an earlier spike ($t_1$) compared to weaker inputs ($I_2$).])

#v(1em)
=== The Phase Ambiguity Problem <s.phaseambiguity>

#serif-text()[ A critical challenge in temporal coding is the need for a temporal reference frame. In Rate Coding, the "phase" (absolute timing) is irrelevant. However, in Temporal Coding, a spike at time $t$ only has meaning relative to a reference $t_0$. If the receiver does not know when the stimulus started, it cannot decode the latency.

In engineering, this is solved by a global clock or a "frame start" signal. In the brain, evidence suggests that background oscillatory rhythms (brain waves, such as theta or gamma cycles) may serve as this global reference, allowing populations of neurons to synchronize their "clocks" and decode relative timings accurately. ]

#figure(include("figures/phaseambiguity.typ"), caption:[The phase ambiguity problem in temporal encoding. Spikes occurring at the same relative phase ($phi_1$ and $phi_2$) across different oscillation cycles are mathematically indistinguishable ($phi_1 = phi_2 (mod 2pi)$). Without a mechanism to track the global cycle count, downstream neurons cannot determine whether a spike represents a delayed response to a previous stimulus or an early response to a new one.])

#v(1em)
=== Population & Sparse Coding <s.populationcoding>

#serif-text()[
While single-neuron codes provide the basic signaling mechanism, the brain employs ensemble strategies to ensure robustness and precision. In population coding, variables are represented by the joint activity of a large group of neurons, each with broad, overlapping tuning curves. A classic example is found in the Primary Visual Cortex (V1), where orientation-selective neurons each respond maximally to a preferred angle but also fire weakly for nearby orientations. By reading the weighted population vector across the group, the network reconstructs the stimulus with far greater precision than any individual cell could provide alone.
The brain further optimizes for metabolic efficiency through sparse coding, where only a small fraction of neurons are active at any moment. This strikes a mathematical balance between representational capacity and energy cost, and is naturally enforced by lateral inhibition circuits that suppress weaker, competing responses. ]

#v(1em)
=== Coexistence of Codes <s.coexistenceofcodes>

#serif-text()[ These schemes are not mutually exclusive but complementary. A circuit may use @ttfs for a rapid initial response — alerting the system to a salient change — before transitioning to rate-based activity for sustained processing. Neuromorphic systems often adopt this hybrid approach, using temporal codes for energy-efficient sparse event transmission and rate-based readouts for interfacing with downstream control systems. This thesis follows the same principle, using @ttfs encoding for the transmission of visual features combined with a population-level representation at the hidden layer. ]

#v(2em)
== Neural Networks <s.neuralnetworks>

#serif-text()[ Having established the mathematical description of the individual neuron, we now turn to the collective behavior of these units. A single neuron, regardless of its dynamical complexity, is of limited computational utility in isolation. Functional intelligence emerges only when these units are organized into specific structural topologies. We implement these topologies in our experimental framework, as detailed in @s.network.

The brain is not a random mesh of connections; it is constructed from recurring architectural "motifs" that appear across various cortical areas. Understanding these motifs is essential for designing neuromorphic systems that transcend simple feed-forward processing. ]

#v(1em)
=== Synaptic Efficacy & Weights <s.synapticefficacy>

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
=== Directionality <s.directionality>

#serif-text()[ Structurally, neural topologies can be categorized by the flow of information.

In sensory peripheries (such as the retina) and early processing stages, information flows unidirectionally from input to output. This topology supports rapid, reflex-like feature extraction. This configuration is known as a feed-forward network, which is mathematically equivalent to a Directed Acyclic Graph (@dag) and serves as the standard architecture for most Deep Learning @cnn:pl.

In higher cognitive areas, the dominant topology is recurrence. Neurons form feedback loops, connecting back to themselves or to distinct layers. This recurrence introduces a time component to the computation, transforming the network into a dynamical system where the current output depends not only on the input but on the network's previous state (history). ]

#figure(include("figures/connectivity.typ"), caption:[Network topologies. (A) Feed-Forward. (B) Recurrent.])

#v(1em)
=== Synaptic Hypothesis: Structure As Function <s.synaptichypothesis>

#serif-text()[ A foundational premise in neuromorphic engineering, derived from biological observation, is that the neuron operates largely as a generic processing unit. The functional identity of a neural circuit—whether it processes visual stimuli or governs motor control—is determined principally by the topology and efficacy of its synaptic interconnections.

This paradigm, known as the Synaptic Hypothesis, posits that the physical configuration of synaptic weights constitutes the substrate for all computation and memory. Unlike traditional Von Neumann architectures, where data is retrieved from a distinct memory module and processed in a central CPU, biological systems eliminate the distinction between "data" and "program." Memory is not a static artifact, but a latent computational potential distributed across the network's structural graph. Consequently, learning in a neuromorphic system is realized through the physical alteration of these synaptic weights, ensuring robust, decentralized processing that is inherently resistant to localized hardware failure (graceful degradation). ]

#v(1em)
=== Inhibition Patterns <s.inhibitionpatterns>

#serif-text()[ A ubiquitous micro-circuit motif in the cortex is lateral inhibition. In this configuration, an active excitatory neuron stimulates distinct inhibitory interneurons, which in turn suppress the activity of neighboring excitatory neurons. This competition engenders a @wta dynamic: as one neuron—representing a specific feature or decision—becomes active, it effectively silences its competitors. In the context of neuromorphic engineering, @wta circuits are indispensable; they provide a physical mechanism for both noise reduction, by actively suppressing weak, sub-threshold signals, and categorical decision making, enabling the circuit to autonomously select the most salient option without the need for a central processor to sort or compare values. We utilize this Winner-Take-All motif in our output layer (@s.tresholding) to enforce categorical decision-making. ]

#figure(include("figures/lateralinhibition.typ"), caption:[The mechanism of lateral inhibition. (A) A highly stimulated neuron in the input layer strongly excites its corresponding output neuron while simultaneously sending lateral inhibitory signals to its immediate neighbors. (B) This architectural motif acts as a spatial filter, producing a contrast enhancement effect. A broad input stimulus (dashed blue line) is transformed into a sharper output response (solid purple line) characterized by an amplified center and suppressed surroundings (a "Mexican hat" profile), thereby sharpening signal boundaries.])

#serif-text()[ While lateral inhibition processes information in the spatial domain, Feed-Forward Inhibition (FFI) operates in the temporal domain. Structurally, this motif bifurcates an input signal into two parallel pathways: a direct excitatory route to the target neuron, and a disynaptic inhibitory route that reaches the same target with a slight synaptic delay. This architecture creates a narrow "temporal window of opportunity." Because the excitation triggers the neuron immediately before the delayed inhibition abruptly truncates the response, the neuron is prevented from integrating noise over extended durations. Consequently, FFI forces the neuron to function as a precise Coincidence Detector rather than a sluggish integrator, a dynamic that is fundamental to sound localization in the auditory cortex and fine-grain timing in the somatosensory system. ]

#v(2em)
== Biological Learning <s.biologicallearning>

#serif-text()[ As previously established, the functional identity of a neural circuit is not defined by a transient software state, but by its physical hardware configuration. Consequently, "learning" in a biological substrate cannot be viewed as a simple parameter optimization; it is a fundamental morphological process. If structure dictates function, then the acquisition of new skills or memories necessitates the physical restructuring of the connectome itself.

Because the brain lacks a central supervisor or global communication bus, this restructuring must be driven by Locality. A synapse can only change based on information physically available at the cleft: the activity of the pre-synaptic axon, the voltage of the post-synaptic dendrite, and the immediate neurochemical environment. Despite this constraint, the brain successfully credits specific synaptic events with outcomes that occur seconds or minutes later.

This adaptation occurs across multiple timescales and spatial resolutions via two distinct mechanisms: Structural Plasticity (the rewiring of the network topology) and Synaptic Plasticity (the modulation of connection strength). ]

#v(1em)
=== Structural Plasticity <s.structuralplasticity>

#serif-text()[ While synaptic weight adjustment accounts for rapid learning and pattern recognition, the long-term storage of information and the optimization of energy efficiency are governed by structural plasticity. This mechanism involves the physical genesis (synaptogenesis) and removal (pruning) of synapses and even entire axonal branches. ]

#box-text()[
- *Synaptogenesis*: When neurons are repeatedly co-active but lack a direct connection, the brain can physically grow new dendritic spines and axonal boutons to bridge the gap. This effectively alters the network's topology, creating new pathways for information flow where none existed before.
- *Pruning*: Equally critical is the removal of redundant or noisy connections. During sleep and developmental critical periods, the brain aggressively prunes weak synapses. This "sparsification" reduces metabolic cost and increases the signal-to-noise ratio of the circuit by removing irrelevant pathways. ]

#serif-text()[In the context of the Synaptic Hypothesis, structural plasticity represents the "compiling" of temporary associations into permanent hardware architecture. ]

#v(1em)
=== Synaptic Plasticity <s.synapticplasticity>

#serif-text()[ Once a structural connection exists, its efficacy—the magnitude of the post-synaptic response to a pre-synaptic spike—must be tuned. In biological terms, this "weight" corresponds to the amount of neurotransmitter released and the density of receptors on the receiving side. This fine-grained adjustment is governed by local learning rules. ]

#v(1em)
=== Hebbian Learning: Rate-Based Correlation <s.hebbianlearning>

#serif-text()[ The foundational axiom of biological learning was postulated by Donald Hebb in 1949. Hebb proposed that synaptic efficiency is a function of the correlated activity between two neurons. Colloquially summarized as "Neurons that fire together, wire together," this rule implies that the brain learns by detecting statistical regularities in sensory input.

Mathematically, if neuron $A$ consistently takes part in firing neuron $B$, the connection from $A$ to $B$ is strengthened. This mechanism allows the brain to perform unsupervised clustering, physically encoding associations between features that occur simultaneously in the environment (e.g., the smell of smoke and the sight of fire). ]

#v(1em)
=== Spike-Timing-Dependent Plasticity (STDP) <s.stdp>

#serif-text()[ Modern neuroscience has refined Hebb’s macroscopic theory into a precise, millisecond-scale mechanism known as @stdp. Unlike rate-based models, @stdp operates on the precise timing of individual action potentials, introducing the critical element of causality.

The @stdp rule adjusts the synaptic weight based on the relative timing difference ($Delta t$) between the pre-synaptic input and the post-synaptic output: ]

#box-text()[
- *@ltp*: If the input spike arrives *before* the output spike ($Delta t > 0$), it implies the input contributed to the firing. The synapse is strengthened to reinforce this causal link.
- *@ltd*: If the input spike arrives *after* the output spike ($Delta t < 0$), the input was irrelevant to the decision. The synapse is weakened. ]

#serif-text()[ This asymmetry allows the network to self-organize, naturally filtering out random noise while reinforcing specific spatiotemporal patterns. We adapt this causal rule to a discrete temporal window for our unsupervised learning experiments in @s.ttfsstdp. ]

#figure(include("figures/stdpcurve.typ"), caption:[The @stdp Learning Curve. Synaptic weight change is plotted against spike timing difference. Pre-before-post timing triggers strengthening @ltp, while post-before-pre triggers weakening @ltd.],placement: auto)

#v(1em)
=== Homeostatic Plasticity <s.homeostatic>

#serif-text()[ If Hebbian mechanisms (@ltp) were the sole drivers of plasticity, neural networks would be inherently unstable. A positive feedback loop would emerge where strengthened synapses drive higher firing rates, which in turn induce further strengthening, leading to runaway excitation (seizures). Conversely, unchecked LTD could silence a network entirely.

To maintain stability, the brain employs Homeostatic Plasticity (or Synaptic Scaling). This is a global regulatory mechanism that operates on a slower timescale (minutes to hours). It functions as a negative feedback loop: if a neuron's average firing rate exceeds a target set-point, the cell chemically downscales the strength of all its incoming synapses. This ensures that neurons remain within a sensitive dynamic range, preventing saturation regardless of how strong the inputs become. ]

#v(3em)
#line(length:100%)
#v(3em)
#serif-text()[ The following chapter shifts perspective---from biology to engineering. We examine the mathematical framework of modern Deep Learning, to identify where its abstractions diverge from biological reality and what computational cost those divergences impose. This analysis will make explicit the bottlenecks that neuromorphic architectures are designed to resolve, grounding the methodological choices of this thesis in a concrete technical rationale. ]

#pagebreak()

= Technical Details Of Machine Learning <s.technicaldetailsofml>

#serif-text()[ This chapter delineates the technical foundations of modern artificial intelligence, contrasting the established paradigms of @dl with the emerging principles of Neuromorphic Engineering. We begin by analyzing the algorithmic architecture of standard Deep Learning, identifying the computational bottlenecks inherent in its reliance on dense matrix multiplication and backpropagation.

A critical distinction must be drawn between biological plausibility and bio-inspired engineering. From an engineering perspective, the primary objective is functional utility. An engineer may treat the brain merely as a source of heuristic inspiration rather than a blueprint to be copied dogmatically. However, the pursuit of biologically plausible systems remains vital; it offers potential advantages in robustness and energy efficiency while serving as a verification tool for neuroscience. ]

#v(2em)
== Optimization <s.optimization>

#serif-text()[ Optimization is the selection of a "best candidate" with regard to defined criteria. Biological learning fits this description, where the optimal candidate is the configuration of synaptic weights that performs well for a specific task. Therefore, it is useful to establish a mathematical framework for this process.

Fundamentally, a deep learning model operates as a function approximator. We assume the existence of an unknown underlying function $f: X arrow Y$ that perfectly maps inputs to their target outputs. Since this true function is unknown, we construct a family of hypothesis functions $f_bold(theta)(bold(x))$ to approximate it. Here, $bold(theta) in RR^d$ represents the state of the system—a vector containing all tunable parameters, such as synaptic weights or biases. The dimensionality $d$ corresponds to the degrees of freedom of the model.

The key problems in optimization are defining the objective goal (the loss function) and finding the parameter configuration $bold(theta)$ that achieves that goal. ]

#v(1em)
=== Supervised Learning <s.supervised>

#serif-text()[ To guide the search for optimal parameters $hat(bold(theta))$, we must quantify the divergence between the model's predictions and the ground truth. We define a scalar Loss Function $cal(L)(hat(bold(y)), bold(y))$ that evaluates the error on a single data point. To ensure generalization, we seek to minimize the Empirical Cost Function $J(bold(theta))$, defined as the average loss over a dataset of size $N$:

$ J(bold(theta)) = 1/N sum_(i=1)^N cal(L)( f_bold(theta)(bold(x)_i), bold(y)_i) $

Geometrically, the cost function $J(bold(theta))$ induces an Optimization Landscape. Finding a low-energy state in this non-convex topology is the central challenge of AI training. We rely on iterative optimization algorithms, principally Gradient Descent. This method updates the system state in the direction opposite to the gradient vector $nabla_(bold(theta)) J(bold(theta))$ (the steepest ascent). The update rule for iteration $t$ is:

$ bold(theta)_(t+1) arrow.l bold(theta)_t - eta nabla_(bold(theta)) J(bold(theta)_t) $

Here, $eta$ represents the Learning Rate. Because computing the gradient over the entire dataset $N$ is computationally prohibitive, modern AI employs Stochastic Gradient Descent (SGD), approximating the gradient using small random subsets (mini-batches). This introduces beneficial noise, preventing the system from getting trapped in shallow local minima.

Crucially, gradient descent requires the loss function to be differentiable. As will be discussed later, this presents a significant challenge for optimizing neuromorphic systems utilizing discrete, non-differentiable spike trains. ]

#figure(include("figures/gradientdecent.typ"), caption:[The Optimization Landscape. The system seeks to traverse the high-dimensional surface defined by $J(bold(theta))$ to find the global minimum $bold(theta)^*$, using the gradient $nabla J$ as a navigational compass.])

#serif-text()[ Strictly minimizing the empirical cost carries the risk of overfitting — the model memorizes training data including noise rather than learning the underlying function. In biological systems this is naturally regulated by metabolic constraints; the brain prunes weak connections to maintain a sparse topology, effectively trading model complexity for generalization. In artificial systems this is managed via explicit regularization penalties added to the cost function. ]

#v(1em)
=== Unsupervised Learning <s.unsupervised>

#serif-text()[ While supervised learning relies on labeled targets, biological systems predominantly learn via Unsupervised Learning. In this regime, the dataset consists only of input vectors $X = {bold(x)_1, ..., bold(x)_N}$. The optimization objective shifts from minimizing prediction error to minimizing representation error.

Mathematically, the goal is often to discover a lower-dimensional manifold that efficiently captures the structure of the data. A common formulation is the minimization of Reconstruction Loss, where the system attempts to compress the input into a latent code and reconstruct it:

$ J(bold(theta)) = 1/N sum_(i=1)^N || bold(x)_i - f_"decode"(f_"encode"(bold(x)_i; bold(theta))) ||^2 $

Alternatively, the system may optimize for clustering density or distances between feature centroids. The distinction between supervised and unsupervised learing is critical for Neuromorphic Engineering, as biological plasticity rules (like @stdp) are unsupervised, functioning by detecting statistical correlations in the input stream to build internal representations without external labels. ]

#v(2em)
== Deep Learning Framework <s.deeplearningframework>

#serif-text()[ Modern Deep Learning aggregates simple units into high-dimensional layers. A deep network with $L$ layers is expressed as a composite function mapping input $bold(x)$ to output $bold(y)$ through nested operations:

$ bold(y) = f_L ( ... f_2 ( f_1 ( bold(x) ) ) ) $

During the Forward Pass, each layer performs an Affine Transformation (a linear rotation and scaling of data via weight matrix $bold(W)$ and bias $bold(b)$) followed by a Non-Linear Activation ($sigma$):

$ bold(z)^((l)) = bold(W)^((l)) bold(a)^((l-1)) + bold(b)^((l)) $
$ bold(a)^((l)) = sigma(bold(z)^((l))) $

The non-linearity prevents the deep stack from collapsing into a single linear equation. Modern networks rely on the Rectified Linear Unit (ReLU), $f(x) = max(0, x)$. Its derivative (0 or 1) preserves the magnitude of the gradient, allowing error signals to travel through deep structures without vanishing. ]

#figure(include("figures/activations.typ"), caption:[Activation Functions. The Sigmoid (left) saturates gradients. The ReLU (right) preserves gradient magnitude for positive inputs.])

#serif-text()[ During the Backward Pass, Backpropagation recursively applies the Chain Rule via Automatic Differentiation to attribute the total error $J(bold(theta))$ to specific weights.

To achieve high throughput, these operations are vectorized. The affine transformation for an entire layer is executed as a Dense Matrix Multiplication. This is the defining characteristic of modern AI hardware. A deep network is effectively a sequence of massive matrix multiplications, which is highly parallelizable on GPUs]

#figure(include("figures/matrixmath.typ"), caption:[Deep Learning as Matrix Multiplication. Forward and backward passes rely on dense matrix products, necessitating high-bandwidth memory access.])

#v(1em)
=== Convolutional Neural Networks (CNNs) <s.cnn>

#serif-text()[ For visual tasks, standard Multi-Layer Perceptrons scale poorly; connecting every pixel to every neuron ignores the spatial structure of the data and creates an intractable number of weights. To solve this, @dl utilizes @cnn:pl.

CNNs apply small, learnable weight matrices known as "kernels" or "filters" that slide (convolve) across the input image. This architecture introduces two critical inductive biases: ]
#box-text()[
1. *Local Connectivity:* Neurons only process a small, local receptive field, analogous to the biological visual cortex.
2. *Weight Sharing:* The exact same kernel is applied across the entire image, drastically reducing the number of tunable parameters and establishing translation invariance (a feature learned in one corner of an image can be recognized anywhere else).]

#serif-text()[While CNNs are the standard baseline for spatial processing, they remain fundamentally synchronous and frame-based, evaluating the entire image structure in dense mathematical passes regardless of local activity. ]

#v(2em)
== Why Is Deep Learning Inefficient? <s.whyisdlinefficient>

#serif-text()[ While the matrix-centric formulation of Deep Learning enables high-throughput parallelization on GPUs, it fundamentally conflicts with the physical constraints of modern computing hardware. As models scale to billions of parameters, the primary bottleneck shifts from algorithmic capability to physical realizability. This inefficiency manifests in four distinct engineering dimensions: ]

#v(1em)
=== The Von Neumann Bottleneck & Data Movement <s.vonneumanbottleneck>

#serif-text()[ The most significant physical limitation is the Von Neumann Architecture, which physically separates the Processing Unit from the Memory Unit. To perform a single inference step, the processor must fetch the entire weight matrix from off-chip DRAM to on-chip registers, perform the calculation, and write the results back.

According to Horowitz and Dally @Placeholder, fetching a 32-bit value from off-chip DRAM consumes approximately 640 pJ, whereas performing a floating-point addition on that value consumes only 0.1 pJ. The system expends 99.9% of its energy transporting data, and only 0.1% actually computing. ]

#figure(include("figures/vonneuman.typ"), caption:[The Von Neumann Bottleneck. The separation of memory and compute forces massive energy expenditure on data transport.])

#v(1em)
=== Dense Processing of Sparse Data <s.denseprocessing>

#serif-text()[ Standard Deep Learning implementations rely on Dense Matrix Multiplication (GEMM). This approach is algorithmically rigid: it executes the same number of operations regardless of the data content.

Real-world sensory data is often highly sparse, and the ReLU activation function naturally produces activation maps where the majority of values are zero. However, a standard GPU is "blind" to this sparsity. It will dutifully fetch a zero from memory and multiply it by a weight ($0 times w = 0$), consuming energy and clock cycles to produce a null result. Deep Learning's inability to exploit this silence represents a massive structural inefficiency. ]

#v(1em)
=== The High Cost of Synchrony <s.highcostofsync>

#serif-text()[ Deep Learning hardware is typically Synchronous, operating in lockstep with a global clock. Driving a high-frequency clock signal across an entire silicon die forces billions of transistors to charge and discharge continuously, regardless of whether the chip is doing useful work. In high-performance processors, this clock distribution network alone can consume 30% to 40% of the total power budget. Furthermore, global synchronization enforces a "worst-case" latency: faster computations must sit idle and wait for the slowest operations to finish before the next clock cycle begins. ]

#v(1em)
=== Backpropagation and Global Dependencies <s.globaldependencies>

#serif-text()[ Finally, Backpropagation imposes severe constraints on memory and latency because it is non-local in both time and space. To update a specific weight, the system must wait for the Forward Pass to finish, calculate the global error, and wait for the backward pass to propagate the gradient.

This creates a "Locking Problem." The activations of every intermediate layer must be stored in high-speed memory (VRAM) for the duration of the entire pass, preventing that memory from being reused. Additionally, a local synapse cannot adapt to local changes instantly; it is shackled to the global error loop. ]

#v(2em)
== Principles of Neuromorphic Engineering <s.principlesofneuromorphic>

#serif-text()[ As established in the _History & Developments_ chapter, Neuromorphic Engineering is the translation of biological dynamics into silicon hardware. It replaces the rigid, clock-driven logic of standard computing with the adaptive, event-driven dynamics of neural tissue. This approach rests on three architectural pillars that directly address the bottlenecks of Deep Learning: ]

#box-text()[
- *Co-location of Memory and Compute (The Synaptic Principle):* Neuromorphic architectures eliminate the Von Neumann bottleneck by distributing memory across the silicon die. Each artificial neuron stores its own state and synaptic weights locally, processing data *in situ* to eliminate the energy cost of shuttling data.
- *Event-Driven Asynchrony (The Action Potential Principle):* Neuromorphic systems abandon the global clock. They operate asynchronously, driven strictly by the arrival of data. If a part of the network is not processing information, it consumes negligible power, ensuring energy scales linearly with task complexity rather than network size.
- *Sparse Communication (The Spike Principle):* Neuromorphic systems utilize binary Spikes for communication. Information is encoded in the precise timing of events rather than complex magnitudes, drastically reducing the bandwidth required to transmit information between neurons. ]

#v(2em)
== Training Spiking Networks <s.trainingsnn>

#serif-text()[ While the physical architecture of neuromorphic systems is highly efficient, training these networks presents a fundamental mathematical challenge. Standard deep learning relies on gradient descent, but backpropagation cannot be directly applied to native @snn:pl.

In a spiking network, the neuron's activation function is a discontinuous step function (the Dirac delta event threshold). The derivative of this function is zero everywhere except at the exact moment of the spike, where it is undefined. Consequently, gradients calculated using the chain rule immediately drop to zero—known as the "Dead Neuron" problem—preventing error signals from flowing backward through the network to update the weights.

To circumvent this non-differentiability and optimize network parameters, the field of neuromorphic engineering generally employs two distinct paradigms: ]

#v(1em)
=== Direct Weight Transfer <s.weighttransfer_theory>

#serif-text()[ A pragmatic engineering approach to bypass the dead neuron problem is offline training. In this paradigm, a standard, continuous @ann (such as a network utilizing ReLU activations) is trained conventionally using backpropagation. Once convergence is achieved, the learned weights are directly mapped onto a structurally identical Spiking Neural Network.

The underlying premise is that the continuous activation values of the ANN can be approximated by the discrete firing rates of the @snn over a set time window. While this method allows the spiking system to inherit the high accuracy of gradient descent, direct weight transfer requires careful scaling and normalization. If the weights are copied without adjustment, the resulting @snn may suffer from catastrophic saturation (firing constantly) or severe signal degradation (failing to reach the spiking threshold). This conversion process and its associated quantization constraints are evaluated in our implementation in @s.weighttransfer_method. ]

#v(1em)
=== Native Local Learning (STDP) <s.locallearning>

#serif-text()[ To fully exploit the energy efficiency and event-driven dynamics of neuromorphic hardware, training must ideally occur natively on the spiking substrate. This requires abandoning global backpropagation in favor of biologically plausible, mathematically local learning rules.

As established in @s.biologicallearning, Spike-Timing-Dependent Plasticity (@stdp) adjusts synaptic weights based strictly on the temporal correlation of local pre- and post-synaptic spikes. Because @stdp relies exclusively on local physical events rather than global error gradients, it does not require a differentiable loss function. This allows the network to completely bypass the dead neuron problem, enabling unsupervised feature extraction and real-time adaptation directly on the spiking architecture. ]

#v(1em)
=== Surrogate Gradient Descent <s.surrogategradientdecent>

#serif-text()[ For completeness, it must be noted that the current dominant paradigm in @snn research utilizes Surrogate Gradients. In this approach, the network operates using the discontinuous spike step-function during the forward pass, but temporarily replaces the undefined derivative with a smooth, continuous approximation (a "surrogate") during the backward pass. While this thesis focuses on evaluating direct weight transfer and native unsupervised @stdp, surrogate methods represent a highly effective hybrid approach, allowing backpropagation-like algorithms to estimate gradients across discrete spiking layers. ]

#v(2em)
== Neuromorphic Hardware Techniques <s.neuromorphichardware>

#serif-text()[ Central to realizing these computational efficiencies in physical hardware is @aer, a communication protocol that mirrors the sparse nature of biological spikes. Instead of continuous data streaming, the hardware only transmits the "address" of a firing neuron across a shared digital bus, allowing a single physical wire to represent thousands of virtual axonal projections. ]

#figure( include("figures/inmemory.typ"), caption: [In-Memory Computing via a Crossbar Array. Unlike von Neumann architectures, memory and computation are physically co-located. Input voltages ($V$) are applied to the wordlines. Memory elements at the junctions hold programmable conductances ($G$). Multiplication is natively performed at each junction by Ohm's Law ($I=V times G$), and resulting currents are summed along the bitlines via Kirchhoff's Current Law. This allows dense matrix-vector multiplications to occur in a single analog time step with zero data transport cost.])

#serif-text()[ The crossbar array provides a direct structural surrogate for the neural neuropil. Because the architecture handles multiplication and summation natively through physical laws, it is uniquely suited to implement biological "macro-motifs." By routing bitline currents through local feedback loops, the hardware can instantiate complex dynamics such as Lateral Inhibition and Winner-Take-All circuits without the overhead of high-level software instructions.

This synergy between physical topology and functional motifs allows the hardware to inherit the computational efficiency of the neocortex, effectively making the architecture itself the algorithm. ]

#figure( include("figures/inmemoryhierarcy.typ"), caption:[Architectural Comparison. (Left) The Von Neumann architecture separates memory and compute, creating a bottleneck. (Right) The Neuromorphic architecture co-locates them, mimicking the distributed topology of biological neural networks.] )

#v(3em)
#line(length:100%)
#v(3em)
#serif-text()[ Having established the theoretical limitations of traditional deep learning and the physical principles of neuromorphic engineering, the following chapter details the specific methodologies, network architectures, and software frameworks utilized in this thesis to evaluate @ann\-to-@snn weight conversion and native @stdp learning on visual event data. ]

#pagebreak()

= Method <s.method>

#serif-text()[ This chapter details the specific implementations of the neuromorphic architectures proposed to address the limitations of standard deep learning. Aligning with the biological constraints of sparsity, asynchrony, and locality established in previous chapters, we outline the construction and evaluation of a @snn.

To empirically validate the theoretical advantages of neuromorphic algorithms, we evaluate the system on a benchmark image classification task. The experiment is bifurcated into three distinct phases:]

#box-text()[
1. *Neuron Model Evaluation:* Evaluating the decoding efficiency and accuracy of different simulated spiking models.
2. *Inference via Weight Transfer:* Evaluating the zero-shot performance of these @snn:pl initialized with weights directly mapped from a classically trained Artificial Neural Network (ANN).
3. *Native Unsupervised Learning:* Training the @snn from scratch utilizing local @stdp.
]

#v(2em)
== Dataset & Pre-processing <s.dataset>

#serif-text()[ To benchmark these algorithms, we require a dataset that necessitates the extraction of complex spatial features but remains computationally tractable for rapid experimental iteration. We utilize the MNIST database of handwritten digits @Placeholder.

The dataset consists of a training set of 60,000 examples and a test set of 10,000 examples of digits (0-9). Each instance is a $28 times 28$ pixel grayscale image. While standard deep learning models routinely score over 90% accuracy on this task, making it largely a solved problem in classical AI, its well-understood feature space makes it an ideal, isolated baseline. Because the spatial hierarchy of MNIST is relatively shallow, it allows us to evaluate the efficacy of neuromorphic learning rules without the confounding variables introduced by massive, multi-layered convolutional architectures.

Crucially, the MNIST images are pre-processed by the dataset creators to be size-normalized and centered within the pixel grid using the center of mass of the pixels. This spatial alignment is a vital prerequisite for our chosen network topology. Unlike @cnn:pl, which slide localized filters across an image, the @fcn utilized in this thesis lacks translation invariance. If a digit were shifted several pixels off-center, the @fcn would perceive it as an entirely novel pattern. The pre-centered nature of MNIST mitigates this limitation, ensuring that the network can reliably map specific geometric strokes to specific input neurons.

#figure( image("figures/mnistexamples.png"), caption: [Sample of the MNIST dataset. The 28x28 images are normalized and flattened into 1D vectors before being translated into temporal spike events.])

Furthermore, the dataset exhibits a high degree of spatial sparsity. In a typical MNIST image, the vast majority of pixels represent the empty background. From a neuromorphic engineering perspective, this sparsity is highly advantageous. As established in the theoretical framework, event-driven systems expend energy strictly when events occur. A sparse input array ensures that the majority of input neurons remain quiescent, minimizing bus congestion and validating the energy-efficiency claims of the proposed @snn.

#figure( image("figures/mnistpixeldisribution.png"), caption: [Sample of the MNIST dataset. The 28x28 images are normalized and flattened into 1D vectors before being translated into temporal spike events.])

Before the raw images can be converted into temporal spike trains, they must undergo standard spatial pre-processing to ensure compatibility with the network's mathematical boundaries. This consists of two primary transformations: ]

#box-text()[
1. *Normalization*: Raw pixel intensities in the MNIST dataset range from $0$ (pure black) to $255$ (pure white). To stabilize the learning algorithms and ensure consistent weight scaling, these values are strictly normalized to a continuous float range of $p_i in [0.0, 1.0]$.
2. *Flattening*: Because this thesis utilizes a Fully Connected Network (FCN) to facilitate direct weight transfer, the 2D spatial structure of the images must be unrolled. Each $28 times 28$ matrix is flattened into a 1-dimensional vector of $784$ elements. ]

#serif-text()[ Consequently, every individual image is presented to the system as a discrete array of $784$ normalized intensities. In the classical Artificial Neural Network (ANN), these continuous values are fed directly into the input neurons. However, because @snn:pl operate exclusively on discrete events, these normalized values must be passed through a temporal encoding algorithm before inference or learning can begin. ]


#figure( image("figures/mnistdisribution.png"), caption: [Sample of the MNIST dataset. The 28x28 images are normalized and flattened into 1D vectors before being translated into temporal spike events.])


// #figure( include("figures/dataexample.typ"), caption: [Sample of the MNIST dataset. The 28x28 images are normalized and flattened into 1D vectors before being translated into temporal spike events.])


#v(2em)
== Network Architecture <s.network>

#serif-text()[ To facilitate a direct, one-to-one mapping of synaptic weights from the @ann to the @snn, both models must share an identical macroscopic topology. That is the motication for why this implementation utilizes a @fcn, also known as a @mlp, rather than a @cnn.

While @cnn:pl are the standard baseline for vision tasks due to their spatial inductive biases, transferring convolutional kernels into a spiking substrate introduces significant mapping complexities—specifically the need to physically unroll and duplicate shared weights across the spiking array. An @fcn provides a straightforward, mathematically transparent architecture for cleanly evaluating direct weight transfer and @stdp without confounding architectural variables.

The network is structured as a shallow hierarchy to capture the primitive geometric features of the dataset. Let $N_l$ denote the number of neurons in layer $l$. The formal architecture is defined as follows: ]

#box-text()[
- *Input Layer ($L_0$):* The $28 times 28$ pixel grayscale images are flattened into a 1D vector, requiring $N_0 = 784$ input neurons.
- *Hidden Layer ($L_1$):* A fully connected intermediate layer consisting of $N_1 =128$ neurons. The synaptic connections are defined by the weight matrix $W^((1)) in bb(R)^{N_1 times N_0}$.
- *Output Layer ($L_2$):* $N_2 = 10$ neurons, corresponding directly to the categorical digit classes (0 through 9). The connections from the hidden layer are defined by the weight matrix $W^((2)) in bb(R)^{N_2 times N_1}$.
]

#figure( include("figures/architecture.typ"), caption: [Network architecture. The 28x28 images are flattened and passed through a Fully Connected Network. Both the @ann and @snn share this identical macroscopic topology.])



#v(1em)
=== Weight Initialization and Biases <s.weightinit>

#serif-text()[ For the baseline @ann, the weight matrices are initialized using standard PyTorch defaults to ensure stable variance during the initial forward passes.

A critical architectural consideration in @ann\-to-@snn conversion is the handling of bias vectors ($b^((l))$). While standard ANNs utilize biases to shift activation functions, representing a static continuous bias in a spiking network requires either injecting a constant background current into the neuron or actively modifying the neuron's firing threshold. To maintain pure, event-driven sparsity and isolate the performance of the synaptic weights, explicit bias terms were omitted entirely from both architectures. ]

#v(1em)
=== ANN-SNN Compatibility <s.ann-snncompatibility>

#serif-text()[ Despite the macroscopic symmetry required for weight sharing, the microscopic dynamics of the two networks differ fundamentally. The offline @ann utilizes standard continuous activation functions (specifically ReLU) to compute smooth gradients during backpropagation.

In contrast, the @snn replaces these continuous functions with Integrate-and-Fire neurons governed by a strict voltage threshold. This simulates the biological "all-or-nothing" action potential, acting as a hard step function. Furthermore, the spiking architecture utilizes lateral inhibition at the output layer. This engenders a @wta dynamic: as the network integrates evidence over time, the first output neuron to reach its threshold heavily suppresses its competitors, forcing a definitive categorical decision and actively filtering sub-threshold noise. ]



#v(2em)
== SNN Information Representation <s.informationrepresentation>

#serif-text()[ The choice of neural code lays the foundation for information flow and dictates the efficiency of the entire system. While Rate Coding (encoding pixel intensity as spike frequency) is straightforward and simple to implement with standard Integrate-and-Fire neurons, it is inefficient compared to @ttfs. Rate codes require integration over extended time windows to calculate an average, introducing latency and saturating the network bus with redundant spikes. Furthermore, on digital hardware rate coding imposes additional stress on the system due to rapid switching which is very bad for transistor power draw and bus congestion.

To maximize energy efficiency and processing speed, this implementation utilizes a @ttfs temporal encoding. In this regime, a single spike carries the information payload. A high-intensity (bright) pixel triggers an early spike, while a low-intensity (dark) pixel triggers a late spike. This compresses the spatial information into a highly sparse, priority-driven queue; downstream neurons begin processing as soon as the most salient features arrive, without waiting for an entire frame to integrate.

As noted in @s.neuralcoding, temporal codes suffer from Phase Ambiguity—downstream neurons need a reference "clock" to decode latency. To resolve this without relying on a rigid, global system clock, we simulate the biological concept of a saccade (the rapid movement of the eye to fixate on a target). The initial presentation of the image acts as a synchronized global event ($t_0$). All subsequent input spikes are evaluated relative to this saccade onset, providing a natural, biologically plausible temporal reference frame. ]


#v(1em)
=== Encoding <s.encoding>

#serif-text()[ Following the @ttfs principles described in @s.temporalcoding, we convert the continuous pixel intensities of the MNIST dataset into discrete spike trains. For a given input image, we extract the luminance of each pixel and normalize it to a bounded range, where $p_i in [0, 1]$ ($1$ representing maximum intensity and $0$ representing the background).

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
=== Decoding and Neuron Models <s.decoding>

#serif-text()[ Decoding the temporal information generated by the @ttfs encoding requires a neuron model capable of accumulating discrete events. However, a fundamental engineering tension exists between biological fidelity, computational efficiency, and the ability to strictly decode temporal order.

To systematically evaluate these trade-offs, this thesis implements and benchmarks four distinct neuron models. These models range from simple arithmetic accumulators requiring global synchronization to complex, fully asynchronous dynamic systems. Each model processes incoming spikes by updating its membrane potential $V_m(t)$ when an event arrives at time $t$, carrying a synaptic weight $w_i$. ]

#v(1em)
#mini-header()[Model A: The Simple Window Integrator (Standard IF)]

#serif-text()[ The most computationally lightweight approach is the standard Integrate-and-Fire (IF) model without any leak or decay mechanisms. In this paradigm, the neuron acts as a pure arithmetic accumulator during the simulation window.
]

#figure(
  kind: "eq",
  supplement: [Equation],
  caption: [Discrete Recurrence Relation for Model A],
  [
    $ u(t_i^+) = u(t_(i-1)^+) + w_i $
  ]
)

#box-text()[
  *Computational Complexity:* Minimal. This model is extremely cheap to execute on digital hardware, requiring only a single addition operation $O(1)$ per incoming spike.

  *Temporal Decoding:* Low. While highly efficient, this model is theoretically flawed for pure @ttfs decoding because it cannot differentiate the relative order of incoming signals. If Spike A ($w=5$) arrives at $t=1$ and Spike B ($w=2$) arrives at $t=10$, the final potential is identical to the reverse arrival order.

  *Synchronization:* Because the potential never naturally decays, this model relies entirely on a rigid, globally synchronized @saccade (a hard reset of $u$ to $0$ at the end of the simulation window) to prevent the network from firing continuously due to lingering historical noise.
]

#figure(
  include("figures/ifmodel.typ"),
  caption: [Network architecture. The 28x28 images are flattened and passed through a Fully Connected Network. Both the @ann and @snn share this identical macroscopic topology.]
)

#v(1em)
#figure(
  kind: "algo",
  caption: [Model A: Simple Window Integrator Algorithm],
  supplement: [Algorithm],
  mono-text(pseudocode-list(hooks: .5em, indentation: 1em, booktabs: true)[
    simpleIntegratorNeuron(S, W, theta) -> t_fire: #h(1fr)
      + // Let S be the ordered set of spikes $(t_i, w_i)$
      + $u arrow.l 0.0$
      +
      + for each $(t_i, w_i) in S$:
        + // Pure arithmetic accumulation
        + $u arrow.l u + w_i$
        +
        + if $u >= theta$:
          + return $t_i$
      +
      + return $infinity$
  ])
)

#v(1em)
#mini-header()[Model B: The Standard Leaky Integrate-and-Fire (LIF)]

#serif-text()[ Model B---the @lif model is covered in great detail in @s.biolif. Model B fires only if the incoming spike train has a sufficient density of spikes. A sparse spike train cannot overcome the exponentially decaying membrane potential.
]

#figure(
  kind: "eq",
  supplement: [Equation],
  caption: [Discrete Recurrence Relation for Model B],
  [
    $ u(t_i^+) = max(0, u(t_(i-1)^+) dot exp(-(t_i - t_(i-1)) / tau_m) + w_i) $
  ]
)

#box-text()[
  *Computational Complexity:* High. This model is significantly more intensive, requiring the calculation of exponential functions for every discrete event, which consumes substantial clock cycles on standard arithmetic logic units.

  *Temporal Decoding:* Moderate. The exponential leak naturally favors spikes that arrive in rapid succession, providing a basic temporal filter. However, standard LIF struggles to strictly prioritize *order* in a @ttfs scheme unless the time constant $tau_m$ is perfectly tuned to the specific temporal distribution of the dataset.

  *Synchronization:* Similar to Model A, while the leak reduces residual noise, it generally still requires a global saccade reset between distinct inference phases to guarantee a clean slate for the next image.
]

#figure(
  kind: "algo",
  caption: [Model B: Standard Leaky Integrate-and-Fire Algorithm],
  supplement: [Algorithm],
  mono-text(pseudocode-list(hooks: .5em, indentation: 1em, booktabs: true)[
    leakyIntegratorNeuron(S, W, theta, tau_m) -> t_fire: #h(1fr)
      + // Let S be the ordered set of spikes $(t_i, w_i)$
      + $u arrow.l 0.0$
      + $t_"prev" arrow.l 0.0$
      +
      + for each $(t_i, w_i) in S$:
        + $Delta t arrow.l t_i - t_"prev"$
        +
        + // Apply exponential leak based on time delta
        + $u arrow.l max(0.0, u dot exp(-Delta t / tau_m) + w_i)$
        +
        + if $u >= theta$:
          + return $t_i$
        +
        + $t_"prev" arrow.l t_i$
      +
      + return $infinity$
  ])
)

#v(1em)
#mini-header()[Model C: The Current-Accumulating Linear Ramp]

#serif-text()[
  To enforce temporal order differentiation without the transcendental computational overhead of exponential kernels, Model C utilizes a linear time-dependent accumulator. In a @ttfs encoding scheme, the relative latency of an afferent spike must dictate its magnitude of influence on the post-synaptic potential. This is achieved by modeling the membrane potential $u(t)$ as a system of coupled differential equations driven by a sequence of Dirac delta impulses:

  $ dot(I)(t) = sum_i w_i delta(t - t_i) $
  $ dot(u)(t) = I(t) + sum_i w_i delta(t - t_i) $

  This formulation ensures that an afferent spike at time $t_i$ with weight $w_i$ exerts a dual influence: it induces an immediate translocation of the potential state while simultaneously incrementing the rate of change for all subsequent integration intervals. Because early spikes establish the baseline slope $I(t)$ for the remainder of the simulation window, they exert a disproportionate influence on the time-to-threshold. Consequently, the network becomes inherently sensitive to rank-order inputs.
]

#figure(
  kind: "eq",
  supplement: [Equation],
  caption: [Discrete Recurrence Relations for Model C],
  [
    $ I(t_i^+) = I(t_(i-1)^+) + w_i $
    $ u(t_i^+) = u(t_(i-1)^+) + I(t_(i-1)^+) dot (t_i - t_(i-1)) + w_i $
  ]
)

#box-text()[
  *Computational Complexity:* Optimized. The model replaces power-intensive transcendental operations with floating-point additions and a single multiplication per spike event ($I dot Delta t$).

  *Temporal Decoding:* High. The quadratic-like growth of the potential relative to spike arrival times allows for precise differentiation of closely timed input sequences.

  *Synchronization:* This model requires a rigorous global synchronization protocol (e.g., a saccade-driven clock). In the absence of a periodic global reset to zero the state variables $I(t)$ and $u(t)$, the linear integration would diverge toward hardware saturation limits.
]

#figure(
  include("figures/rampmodel.typ"),
  caption: [Evolution of state variables in Model C. This demonstrates the resulting piecewise linear membrane potential $u(t)$. Note how earlier spikes increase the integration gradient, accelerating the trajectory toward the firing threshold $theta$.]
)

#figure(
  kind: "algo",
  caption: [Model C: Discrete State Update Algorithm],
  supplement: [Algorithm],
  mono-text(pseudocode-list(hooks: .5em, indentation: 1em, booktabs: true)[
    linearRampNeuron(S, W, theta) -> t_fire: #h(1fr)
      + // Let S be the ordered set of spikes $(t_i, w_i)$
      + $u arrow.l 0.0$
      + $I arrow.l 0.0$
      + $t_"prev" arrow.l 0.0$
      +
      + for each $(t_i, w_i) in S$:
        + $Delta t arrow.l t_i - t_"prev"$
        + $u arrow.l u + (I dot Delta t) + w_i$
        + $I arrow.l I + w_i$
        +
        + if $u >= theta$:
          + return $t_i$
        +
        + $t_"prev" arrow.l t_i$
      +
      + return $infinity$
  ])
)

#v(1em)
#mini-header()[Model D: Threshold-Sensitive Integration (State-Dependent Discounting)]

#serif-text()[
  Drawing inspiration from the adaptation mechanisms of the @glif model (@s.glif), Model D introduces a state-dependent penalty to incoming spikes. Rather than penalizing spikes based on the passage of time, this model penalizes spikes based on the current internal state of the neuron.

  In this paradigm, the increase in membrane potential is inversely proportional to the current potential itself. When a spike arrives at a neuron in its resting state ($u = 0$), there is no discount, and the full synaptic weight is added. However, if a spike arrives when the neuron is already close to the firing threshold $theta$, its impact is exponentially discounted.
]

#figure(
  kind: "eq",
  supplement: [Equation],
  caption: [Discrete Recurrence Relation for Model D Weight Discounting],
  [
    $ u(t_i^+) = u(t_(i-1)^+) + w_i dot exp(-gamma (u(t_(i-1)^+))/theta) $
  ]
)

#box-text()[
  *Computational Complexity:* Moderate. While it requires an exponential calculation (or a linear approximation), it does not need to continuously track the elapsed time ($Delta t$) between individual spikes, saving memory overhead compared to continuous-time dynamic models.

  *Temporal Decoding:* High. This mechanism inherently prioritizes the sequence of arrival. The earliest spikes encounter an "empty" neuron and contribute their absolute maximum weight. Spikes arriving later encounter a partially filled potential and are severely dampened. Consequently, high-weight spikes must arrive early to have a meaningful impact, naturally aligning with the @ttfs encoding hierarchy.

  *Synchronization:* Similar to Models A and C, this model requires a rigorous global synchronization protocol (e.g., a saccade-driven clock). Because this model drops the continuous temporal leak in favor of event-driven weight scaling, the potential does not naturally decay to zero between images and relies on a periodic reset at the conclusion of the simulation window.
]

#figure(
  include("figures/discountmodel.typ"),
  caption: [Evolution of state variables in Model D. This demonstrates a neuron model where new incoming spikes have an exponentially decaying influence based on the current potential $u(t)$.]
)

#v(1em)
#figure(
  kind: "algo",
  caption: [Model D: Threshold-Sensitive Integration Algorithm],
  supplement: [Algorithm],
  mono-text(pseudocode-list(hooks: .5em, indentation: 1em, booktabs: true)[
    thresholdSensitiveNeuron(S, W, theta, gamma) -> t_fire: #h(1fr)
      + // Let S be the ordered set of spikes $(t_i, w_i)$
      + $u arrow.l 0.0$
      +
      + for each $(t_i, w_i) in S$:
        + // Calculate discount factor based on current potential
        + // Approaches 1.0 at rest, and 0 near threshold
        + $eta arrow.l exp(-gamma dot u / theta)$
        +
        + // Apply the state-dependent increase
        + $u arrow.l u + (w_i dot eta)$
        +
        + if $u >= theta$:
          + return $t_i$
      +
      + return $infinity$
  ])
)

#v(1em)
=== Thresholding and Lateral Inhibition <s.tresholding>

#serif-text()[ Regardless of the chosen internal integration dynamics, all four models share identical discrete output behavior. To implement the @wta dynamics established in @s.inhibitionpatterns, the output classification layer utilizes lateral inhibition. Following the integration step, the neuron evaluates its firing condition: if $V_m(t) > V_"th"$, an output spike is emitted, and the internal state variables ($V_m$, and $I$ if applicable) are reset. Furthermore, if the neuron belongs to the output classification layer, its spike triggers a @wta condition, instantly suppressing all competitors to finalize the classification. ]


#v(2em)
== Training Methodologies <s.training>

#serif-text()[
  Following the optimization principles established in @s.optimization, our training methodology combines weight-based synaptic plasticity with topological structural plasticity. In a neuromorphic context, learning is not merely parameter tuning but a physical self-organization of the network. We implement a dual-stage learning process: unsupervised feature discovery via local plasticity rules, followed by a global structural optimization phase.
]

#v(1em)
=== Structural Plasticity and Synaptogenesis

#serif-text()[
  Drawing from the biological concept of synaptogenesis (@s.structuralplasticity), our model begins with a sparse set of potential connections. During the initial training phase, pre-synaptic activity triggers the probabilistic creation of new synapses. This "random growth" mechanism is computationally efficient and highly parallelizable on digital hardware.

  To ensure functional efficiency, we implement a "synaptic gravity" heuristic. If a post-synaptic neuron already exhibits strong activation, it exerts a higher probability of attracting new connections from coincident pre-synaptic sources. This effectively reinforces emerging patterns, allowing the network to converge on salient features more rapidly than a purely random growth model.
]

#figure(
  kind: "algo",
  caption: [Unsupervised Local Structural Plasticity],
  supplement: [Algorithm],
  mono-text(pseudocode-list(hooks: .5em, indentation: 1em, booktabs: true)[
    structuralPlasticityPhase(N) -> W: #h(1fr)
      + // Let N be the set of all neurons in the network
      + for each neuron $n_"pre" in N$ that fires:
        + // Pre-synaptic firing triggers random growth chance
        + with probability $p_"grow"$, initialize new synapse to random $n_"post"$
      +
      + for each neuron $n_"post" in N$ that fires:
        + for each incoming synapse from $n_"pre"$:
          + if $t_"pre" <= t_"post"$:
            + $w_("pre", "post") arrow.l w_("pre", "post") + Delta w^+$ // Strengthen causal
          + else:
            + $w_("pre", "post") arrow.l 0$ // Prune non-causal or silent
      +
      + // Note: A neuron can act as both pre- and post-synaptic
  ])
)

#figure(
  kind: "algo",
  caption: [Synaptic Growth Probability Heuristics],
  supplement: [Algorithm],
  mono-text(pseudocode-list(hooks: .5em, indentation: 1em, booktabs: true)[
    calculateGrowthProbability(n_"pre", n_"post", t_"pre") -> p_"grow": #h(1fr)
      + // Let C be the current number of synapses on n_pre
      + $p_"base" prop 1 / (C + 1)$  // Inversely proportional to existing connections
      +
      + // Earlier firings have a higher chance to trigger growth
      + $p_"time" prop exp(-t_"pre" / tau_"grow")$
      +
      + // Modulate by lateral inhibition state
      + return $min(1.0, p_"base" dot p_"time" dot eta_"inhibition")$
  ])
)

#v(1em)
=== Pattern Prediction and Order Decoding

#serif-text()[ The core of our learning objective is the detection of spatiotemporal sequences. In a @ttfs paradigm, the relative arrival order of spikes defines the pattern identity. We hypothesize that a neuron successfully learns a pattern (e.g., sequence $A arrow B arrow C$) when it adjusts its internal thresholds to fire as soon as the first sufficient evidence ($A arrow B$) arrives.

  However, this early firing introduces a prediction risk: if the neuron fires based on a prefix ($A arrow B$) but the expected suffix ($C$) fails to arrive, the synaptic efficacy must be penalized. This mimics the biological concept of error-driven learning without a global supervisor; the neuron "predicts" the completion of a learned pattern and self-corrects based on subsequent local evidence. If the predicted input is absent, the weights associated with the prefix are slightly decayed, preventing the network from becoming overly sensitive to incomplete or noisy patterns.
]

#v(1em)
=== Competitive Specialization

#serif-text()[ To prevent all neurons from converging on the same dominant features, we utilize the lateral inhibition motifs described in @s.inhibitionpatterns. When a neuron successfully classifies a pattern and fires, its inhibitory output suppresses neighboring units. This forces competing neurons to specialize in distinct, non-overlapping geometric primitives. In our implementation, this competition is governed by the relative timing of spikes; the fastest neuron to recognize a feature "wins" the representation, while others are forced to adapt to secondary or tertiary patterns in the data stream. ]

#v(1em)
#figure(
  kind: "algo",
  caption: [Winner-Takes-All (WTA) Lateral Inhibition],
  supplement: [Algorithm],
  mono-text(pseudocode-list(hooks: .5em, indentation: 1em, booktabs: true)[
    applyWTAInhibition(N_"pool", theta, I_"inh") -> $n_"winner"$: #h(1fr)
      + // Let N_pool be the set of neurons in the competitive layer
      + $n_"winner" arrow.l "null"$
      + $t_"min" arrow.l infinity$
      +
      + // Phase 1: Identify the earliest firing neuron (The Winner)
      + for each neuron $i in N_"pool"$:
        + if $u_i >= theta$ and $t_i < t_"min"$:
          + $n_"winner" arrow.l i$
          + $t_"min" arrow.l t_i$
      +
      + // Phase 2: Apply lateral inhibition to the losers
      + if $n_"winner" != "null"$:
        + for each neuron $j in N_"pool"$:
          + if $j != n_"winner"$:
            + // Substantially suppress the membrane potential
            + $u_j arrow.l max(0.0, u_j - I_"inh")$
            +
            + // Veto any subsequent firing during this saccade
            + $t_j arrow.l infinity$
      +
      + return $n_"winner"$
  ])
)

#v(1em)
=== Weight Quantization and Stability

#serif-text()[
  To bridge the gap between continuous simulation and physical neuromorphic hardware, we evaluate learning across varying degrees of weight resolution. While standard @stdp assumes continuous weight updates, we implement a quantized learning scheme where synapses are restricted to binary or low-bit integer states. This mimics the constraints of memristive crossbar arrays, where weights are represented by discrete conductance levels. Stability is maintained through homeostatic scaling (@s.homeostatic), ensuring that the total synaptic drive to any individual neuron remains within a fixed dynamic range regardless of the connection density.
]

#v(1em)
=== Weight Transfer <s.weighttransfer_method>

#serif-text()[
  Following the theory of offline training and mapping discussed in @s.weighttransfer_theory, direct one-to-one transfer of floating-point weights from an @ann to an @snn often results in catastrophic failure; the continuous activation scales do not naturally align with discrete spiking thresholds. Furthermore, physical neuromorphic hardware (such as crossbar arrays) imposes strict limitations on synaptic weight resolution, rarely supporting 32-bit floating-point numbers.

  To emulate these hardware constraints and ensure threshold compatibility, the continuous weights of the trained ANN ($W_"ANN"$) undergo a pseudo-INT8 quantization process before being loaded into the @snn simulator. The weights are multiplied by a static scaling factor of $64$, rounded to the nearest integer, and clamped to an 8-bit signed integer range $[-128, 127]$:
]

#figure(
  kind: "eq",
  supplement: [Equation],
  caption: [Weight Scaling and Quantization],
  [
    $ W_"SNN"^{(l)} = max (-128, min (127, "round"(W_"ANN"^{(l)} dot 64) ) ) $
  ]
)

#serif-text()[
  This transformation maps a standard continuous weight of $2.0$ to the maximum synaptic efficacy of $128$. By porting these quantized discrete weights ($W_1$ and $W_2$) directly to the GPU, the simulator strictly enforces the memory boundaries of physical neuromorphic silicon.
]

#v(1em)
=== TTFS STDP Inspired Learning Rule <s.ttfsstdp>

#serif-text()[
  Our learning rule adapts the @stdp mechanism from @s.stdp to the @ttfs domain. In standard continuous networks, synaptic weights are updated via global error gradients. In the @stdp paradigm, a synapse $w_(i j)$ connecting a pre-synaptic neuron $i$ to a post-synaptic neuron $j$ is updated based strictly on the temporal difference between their respective firing times.

  Let $t_i$ denote the spike time of the input neuron, and $t_j$ denote the spike time of the output neuron. The relative arrival time is defined as $Delta t = t_j - t_i$. Because this architecture utilizes a strict @ttfs encoding where neurons fire at most once per @saccade, the classical continuous @stdp curve is adapted into a discrete, deterministic update rule:
]

#box-text()[
  *@ltp:* If $t_i < t_j$, the pre-synaptic spike arrived before (or exactly at) the moment the post-synaptic neuron fired. This indicates causality. The synapse is strengthened, with the magnitude of the update decaying exponentially the further apart the spikes occurred.

  *@ltd:* If $t_i > t_j$, the pre-synaptic spike arrived after the post-synaptic neuron had already fired. The input was irrelevant to the decision, and the synapse is subsequently weakened.

  *Unused Synapses (Penalty):* If a pre-synaptic neuron fires but the post-synaptic neuron never fires during the saccade, a slight negative decay is applied to encourage forgetting of dead connections.
]

#figure(
  kind: "eq",
  supplement: [Equation],
  caption: [Additive STDP Weight Update],
  [
    $ Delta w_(i j) =
    cases(
      A_+ dot exp(-(t_j - t_i) / tau_+) & "if" t_i < t_j space "(LTP)",
      -A_- dot exp(-(t_i - t_j) / tau_-) & "if" t_i > t_j space "(LTD)"
    ) $
  ]
)

#serif-text()[
  To prevent runaway synaptic growth or catastrophic sign-flipping, the updated weights are strictly clamped to a positive physical range $w_(i j) in [0, W_"max"]$. Algorithm 5 details the exact logical implementation of this rule.
]

#figure(
  kind: "algo",
  caption: [The TTFS STDP Weight Update Function],
  supplement: [Algorithm],
  mono-text(pseudocode-list(hooks: .5em, indentation: 1em, booktabs: true)[
    applySTDP(t_"pre", t_"post", w, A_+, A_-, tau, W_"max") -> w_"new": #h(1fr)
      + // If post-synaptic neuron never fired, apply a small decay penalty
      + if $t_"post" == infinity$:
        + return $max(0.0, w - (A_- dot 0.1))$
      +
      + // Calculate temporal difference
      + $Delta t arrow.l t_"post" - t_"pre"$
      +
      + if $Delta t >= 0$:
        + // Pre-before-Post (Causality -> Potentiation)
        + $w arrow.l w + (A_+ dot exp(-Delta t / tau))$
      + else:
        + // Post-before-Pre (Late arrival -> Depression)
        + $w arrow.l w - (A_- dot exp(Delta t / tau))$
      +
      + // Enforce physical hardware constraints
      + return $max(0.0, min(w, W_"max"))$
  ])
)

#serif-text()[
  To execute unsupervised feature extraction on the MNIST dataset, the @stdp rule is embedded within the saccade simulation loop. The network is initialized with randomized synaptic weights $W tilde cal(U)(0, 1)$.

  During the training phase, an image is presented, and the network executes a forward pass. However, unlike the zero-shot inference phase, lateral inhibition (Winner-Takes-All) plays a critical structural role during training. When an output neuron fires, it aggressively inhibits its neighbors. This competitive dynamic forces different neurons to specialize in different geometric features; if one neuron learns to recognize a "loop," the @wta mechanism prevents other neurons from learning that exact same redundant feature.

  At the conclusion of the 64-tick saccade, the simulator halts, compares the timestamp arrays of the hidden and output layers, and computes the @stdp weight updates in parallel before the next image is presented. Algorithm 3 outlines this autonomous learning pipeline.
]

#v(2em)
== Evaluation Metrics

#serif-text()[
  To rigorously validate the proposed Spiking Neural Network (SNN) architectures, the evaluation framework must bridge two distinct domains: *Classification Effectiveness* (decoding accuracy) and *Computational Efficiency* (hardware-agnostic resource usage). Because the networks undergo both supervised transfer (Phase II) and unsupervised native learning (Phase III), specialized metrics are required to capture the evolution of the internal representations.
]

#v(1em)
=== Effectiveness and Classification Performance

#serif-text()[
  For Phase II (Zero-Shot Weight Transfer), performance is measured against the standard 10,000-image MNIST test set. Since the labels are pre-defined by the source ANN, effectiveness is quantified using standard statistical tools:
]

#box-text()[
- *Top-1 Accuracy:* The percentage of images where the first output neuron to fire via the Winner-Takes-All (WTA) mechanism matches the ground-truth label.
- *Quantization-Aware Confusion Matrices:* Utilized to identify specific morphological overlaps (e.g., '4' vs. '9'). These matrices help determine if specific geometric features are disproportionately lost during the INT8 quantization process, evaluating the robustness of the weight-transfer mapping.
]

#serif-text()[
  Evaluating Phase III (Unsupervised STDP) requires a shift in methodology. Because the network learns without labels, output neurons do not inherently map to categorical digits; they map to geometric clusters. To generate a comparable accuracy metric, we employ a *Post-Hoc Labeling* strategy:
]

#box-text()[
1. *Freezing:* Synaptic plasticity is disabled (learning rate set to zero) after the STDP training phase to prevent further weight drift.
2. *Assignment:* A subset of the labeled training data is passed through the network. Each output neuron is permanently assigned the label of the digit class that most frequently triggered it to fire.
3. *Testing:* The standard 10,000-image test set is passed through the "labeled" network to calculate the final Top-1 accuracy.
]

#v(1em)
=== Computational Efficiency (Hardware Proxies)

#serif-text()[
  While the primary goal of neuromorphic engineering is immense energy reduction, evaluating true physical power draw ($J/"inference"$) is restricted by the use of PyTorch-based software simulators running on standard GPUs. Because these platforms incur heavy overhead simulating temporal event loops on von Neumann hardware, we abandon direct power measurements in favor of hardware-agnostic proxy metrics universally recognized in SNN literature:
]

#v(1em)
==== Sparsity and Synaptic Operations (SyOPs)

#serif-text()[
  In a standard ANN, every forward pass requires a fixed number of Multiply-Accumulate (MAC) operations. In an SNN, computation is driven by discrete events. Because Integrate-and-Fire neurons do not require multiplication (a spike simply triggers the addition of its weight to the post-synaptic potential), MACs are replaced by simpler Synaptic Operations (SyOPs).

  The total computational cost for a single 64-tick saccade is estimated as:
]

#figure(
  kind: "eq",
  supplement: [Equation],
  caption: [SNN Operational Cost Proxy],
  [ $ "Total SyOPs" = sum_{l=1}^{L} N_"spikes"^{(l)} dot F_"out"^{(l)} $ ]
)

#serif-text()[
  Where $N_"spikes"$ is the total spikes emitted in layer $l$ and $F_"out"$ is the fan-out of the neurons. By comparing the SNN SyOPs against the fixed MAC count of the baseline ANN, we derive a theoretical energy efficiency ratio.
]

#v(1em)
==== Temporal Latency Metrics

#serif-text()[
  To evaluate the efficacy of the Time-to-First-Spike (TTFS) encoding, we measure the *Time-to-Decision Latency*. This is defined as the exact simulation tick $t in [0, 64)$ at which the WTA output layer reaches a decision. A lower average latency indicates a superior temporal decoder that successfully prioritizes salient information, allowing the system to theoretically power down early in the simulation window.
]

#v(1em)
=== Qualitative Evaluation via Latent Space Projection (t-SNE)

#serif-text()[
  While quantitative metrics provide a measure of accuracy, they often obscure the underlying topology of the learned representations. To visualize the hidden layer's latent space ($N_1 = 128$), we employ t-distributed Stochastic Neighbor Embedding (t-SNE). This non-linear technique projects the high-dimensional manifold onto a 2D plane, preserving local proximities.

  This visualization serves as the primary comparative tool for two distinct hypotheses:
]

#box-text()[
- *Supervised Latent Space (Phase II):* The projection is expected to reveal highly segregated clusters corresponding to the 10 categorical digit classes, as weights were explicitly optimized via cross-entropy loss.
- *STDP Latent Space (Phase III):* In contrast, the native STDP algorithm possesses no semantic knowledge. It operates as a temporal coincidence detector, clustering images based on *morphological similarity*. Digits sharing fundamental structural primitives (e.g., the vertical bars of '1's and '7's) should form contiguous manifolds in the latent space.
]

#serif-text()[
  Demonstrating this geometric clustering via t-SNE empirically validates that the local STDP rule has successfully self-organized into a biologically plausible feature extractor, effectively mapping the input distribution to a structured manifold without the supervision of categorical labels.
]

#v(2em)
== Experiment Setup and Evaluation Phases

#serif-text()[
  To systematically evaluate the proposed Spiking Neural Network (SNN) architectures, the experimental framework is structured as a comparative pipeline between standard connectionist models and neuromorphic event-driven models. This approach isolates three critical variables: *parameter robustness during quantization*, *temporal decoding efficiency*, and *unsupervised feature emergence*.
]

=== The ANN Baseline (Phase 0: Offline Training)

#serif-text()[
  Before the SNN simulation begins, a baseline Artificial Neural Network (ANN) is established to provide the "ideal" weight distribution. This serves as the performance ceiling for the subsequent Phase II weight transfer.
]

#box-text()[
- *Topology:* A standard Multi-Layer Perceptron (MLP) with a $784 -> 128 -> 10$ architecture, utilizing ReLU activation functions in the hidden layer and a Softmax output layer.
- *Optimization:* The network is trained using the *Adam Optimizer* ($lr=10^(-3)$) and *Categorical Cross-Entropy Loss* for 20 epochs on the MNIST training set.
- *Biases:* To facilitate direct mapping to Integrate-and-Fire neurons, the ANN is trained without bias terms ($b=0$), ensuring that synaptic integration depends solely on input weights.
]

=== The SNN Simulation Engine

#serif-text()[
  The SNN environment is a custom-built, discrete-time simulator implemented in PyTorch. Unlike the ANN, which processes data in a single static "glance," the SNN processes data through a *Temporal Saccade* lasting $T_"max" = 64$ ticks.
]

#figure(
  kind: "algo",
  caption: [The Integrated ANN-SNN Experimental Pipeline],
  supplement: [Algorithm],
  mono-text(pseudocode-list(hooks: .5em, indentation: 1em, booktabs: true)[
    run_experiment_pipeline(config):
      + // 1. Phase 0: ANN Baseline
      + ann_model = train_baseline_ann(dataset.train, epochs=20)
      + ann_acc = test_ann(ann_model, dataset.test)
      +
      + // 2. The Parameter Bridge
      + fp_weights = ann_model.get_weights()
      + int8_weights = quantize_and_scale(fp_weights, range=[-128, 127])
      +
      + // 3. Phase II: SNN Weight Transfer
      + snn_transfer = load_snn(int8_weights)
      + s2_acc = evaluate_saccades(dataset.test, snn_transfer, T_max=64)
      +
      + // 4. Phase III: Unsupervised STDP
      + snn_stdp = initialize_random_snn()
      + snn_stdp = train_stdp(dataset.train, snn_stdp, epochs=10)
      + s3_acc = evaluate_with_post_hoc_labels(dataset.test, snn_stdp)
      +
      + return ann_acc, s2_acc, s3_acc
  ])
)

#figure(
  include("figures/softwarearch.typ"),
  caption: [Software architecture and data flow of the SNN simulator.]
)

#v(1em)
=== Hardware and Simulation Parameters

#serif-text()[
  To ensure reproducibility, the hardware-proxy parameters for both the ANN source and the SNN target are standardized as follows:
]

#table(
  columns: (1fr, 1fr, 2fr),
  inset: 10pt,
  align: horizon,
  [*Parameter*], [*Value*], [*Context*],
  [$T_"max"$], [64], [Total simulation ticks per image],
  [Input Cap], [32], [Maximum delay for lowest intensity pixel],
  [Quantization], [INT8], [Bit-depth for weight transfer],
  [$V_"th_h"$], [200.0], [Membrane threshold for hidden layer ($N_1$)],
  [$V_"th_o"$], [100.0], [Membrane threshold for output layer ($N_2$)],
  [Optimizer], [Adam], [Source optimizer for Phase 0 weights],
)

#v(1em)
=== Evaluation Phases

#serif-text()[
  The experiment is executed in three logical phases, each testing a specific claim regarding spiking computation and neuromorphic efficiency.
]

#mini-header()[Phase I: Temporal Sensitivity Benchmarks]
#serif-text()[
  This phase serves as a functional unit test for the four neuron models. We subject the models to synthetic spike trains where the spatial weight is identical, but the *temporal order* is permuted. A valid @ttfs neuron must differentiate between these orders, demonstrating it is not merely acting as a rate-coder.
]

#mini-header()[Phase II: Zero-Shot Weight Transfer]
#serif-text()[
  Phase II benchmarks the "SNN-as-Accelerator" hypothesis. Learned weights are transferred from the ANN to the SNN via the $"INT8"$ bridge. We measure the *Accuracy Decay Rate* ($Delta_"Acc" = "Acc"_"ANN" - "Acc"_"SNN"$) to quantify the information lost during temporal translation.
]

#mini-header()[Phase III: Native Unsupervised Learning]
#serif-text()[
  The final phase evaluates self-organization. Discarding all pre-trained weights, the network is exposed to MNIST via @stdp.
]

#box-text()[
- *Lateral Inhibition:* A @wta mechanism is enforced; the first output neuron to fire suppresses all others for the duration of the saccade.
- *Homeostasis:* Weights are normalized after each stimulus ($sum w_j = K$) to prevent single-neuron dominance and ensure a distributed feature representation. ]

#pagebreak()

= Results <s.results>

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

#serif-text()[ While the experimental results validate the core principles of @ttfs encoding and spiking integration, extrapolating these methods from controlled benchmarks to real-world deployment reveals significant engineering bottlenecks. This chapter critically analyzes the limitations observed during the implementation phase, specifically regarding data encoding, spatial representation, architectural translation, and hardware constraints. ]

#v(2em)
== Encoding Modalities and the Contrast Problem

#serif-text()[ In the engineered test environment, absolute pixel intensity was mapped directly to spike latency. For a highly controlled toy dataset like MNIST, this approach yields adequate performance, as the digits are isolated in high-contrast, low-resolution environments. However, absolute luminance is a notoriously brittle feature for real-world computer vision.

Robust biological and artificial vision systems rely on local contrast (the relative difference between adjacent pixels) to determine object boundaries, as contrast remains invariant under shifting global illumination. Attempting to compute true, normalized contrast directly within a @ttfs spiking network presents a severe temporal bottleneck. To accurately assess relative darkness in a purely temporal code, the downstream neurons must wait for the slowest (darkest) signals to arrive, effectively nullifying the high-speed advantages of the @ttfs priority queue.

Consequently, attempting to calculate contrast natively within the @snn layers is computationally wasteful. The optimal solution is to offload this processing to the sensory periphery. Dedicated neuromorphic sensors, such as Dynamic Vision Sensors (DVS), natively output logarithmic intensity differences. Because the difference in log-space mathematically corresponds to a true contrast ratio independent of absolute luminance, passing this pre-encoded contrast data directly into the @ttfs network avoids the temporal delay problem entirely. ]

#v(2em)
== Representing Space and Dimensionality

#serif-text()[ Encoding precise spatial coordinates using a pure @ttfs scheme proved inherently difficult. In biological visual and motor cortices, spatial locations are often represented via orthogonal population codes, where specific populations activate to indicate direction or position. However, these biological populations largely utilize rate coding; the "intensity" or certainty of a spatial position translates naturally into a higher firing frequency.

Conversely, @ttfs is highly optimized for rapid, categorical decision-making (e.g., triggering a Winner-Takes-All classification), but struggles to map continuous numerical or spatial values without complex phase-ambiguity resolution. A potential architectural workaround is the implementation of hierarchical "space cells." Rather than mapping a massive $32 times 32$ grid to individual temporal delays, the space could be subdivided into localized grids (e.g., overlapping $8 times 8$ populations). This reduces the dimensionality of the temporal encoding, though it introduces resolution artifacts at the boundaries of the receptive fields. ]

#v(2em)
== Architectural Translation (ANN to SNN)

#serif-text()[ The zero-shot inference phase highlighted the frictions of translating classical architectures to spiking substrates. While the Fully Connected topology provided a transparent baseline, mapping state-of-the-art Convolutional Neural Networks (@cnn:pl) is decidedly not a one-to-one process.

Classical continuous networks heavily utilize negative weights, static biases, and mathematical operations like Max Pooling. Neuromorphic systems handle these concepts fundamentally differently. An @snn replaces mathematical pooling with dynamic lateral inhibition, and negative continuous weights must be modeled via discrete inhibitory spike trains. These architectural mismatches dictate that @snn:pl cannot simply act as "drop-in" replacements for deep learning models; they require network topologies natively designed for event-driven dynamics. ]

#v(2em)
== Plasticity Dynamics and Forgetting

#serif-text()[ During the evaluation of unsupervised learning, a fundamental tension emerged between the network's learning velocity and its stability. Unsupervised, local learning rules like @stdp require aggressive hyperparameter tuning. If the plasticity rate is too high, the network adapts quickly but suffers from rapid catastrophic forgetting—overwriting previously learned geometric features when presented with novel stimuli. Conversely, if the plasticity is too low, the network fails to converge on meaningful representations within an acceptable time frame. Designing homeostatic mechanisms to stabilize this plasticity remains a core challenge for native learning. ]

#v(1em)
=== The Sparsity Paradox and GPU Simulation Overhead

#serif-text()[ A core theoretical advantage of the proposed @ttfs network is its extreme spatial and temporal sparsity. By design, only a small fraction of neurons emit spikes, mathematically reducing the dense Multiply-Accumulate (MAC) operations of a standard ANN to a minimal set of @syops.

However, evaluating this sparse algorithm on conventional GPUs introduces a significant "Sparsity Paradox." Standard deep learning frameworks (such as PyTorch) and standard GPU architectures are heavily optimized for dense, contiguous matrix-matrix multiplications via SIMD (Single Instruction, Multiple Data) execution. In the @snn simulation loop, sparsity is enforced via boolean masking (multiplying inactive neuron outputs by zero). While this successfully zeroes out the potential, the GPU @alu:pl typically still execute the underlying floating-point multiplication cycle.

Consequently, the hardware does not naturally skip the computation for quiescent neurons unless specialized, unstructured sparse tensor kernels are deployed. When combined with the overhead of maintaining the temporal loop (the "saccade" ticks), the @snn simulator ironically consumes more absolute clock cycles and physical power on a GPU than the continuous @ann baseline.

This paradox highlights that true energy efficiency cannot be realized via matrix-masking on von Neumann hardware. Realizing the calculated SyOP savings requires deployment on native event-driven Neuromorphic ASICs (Application-Specific Integrated Circuits). These chips discard the matrix-multiplication paradigm entirely, utilizing @aer to asynchronously route data packets only when a spike physically occurs, reducing idle power draw to near zero. ]

#v(1em)
=== Mitigation via Synaptic Pruning

#serif-text()[ While dynamic activation sparsity struggles on GPUs, structural sparsity offers a viable software-level mitigation. Future iterations of this work could introduce aggressive Synaptic Pruning, particularly following the unsupervised @stdp learning phase.

Because @stdp naturally depresses irrelevant synapses toward zero, a thresholding function could permanently sever these connections, converting the dense weight matrices ($W^{(1)}$, $W^{(2)}$) into highly sparse structures. By utilizing block-sparse tensor formats (e.g., Compressed Sparse Row), both software simulators and specialized sparse-accelerator chips can mathematically bypass the zero-weights, yielding physical reductions in memory bandwidth and computation even before transitioning to pure neuromorphic silicon. ]

#v(2em)
== The Physical Hardware Gap

#serif-text()[ Ultimately, the theoretical energy efficiency of neuromorphic algorithms is bounded by physical hardware. The connection density of the biological mammalian cortex vastly exceeds the routing capabilities of modern CMOS (Complementary Metal-Oxide-Semiconductor) fabrication.

While the software simulations in this thesis demonstrate the algorithmic viability of sparse processing, achieving true biological efficiency requires novel materials. Non-volatile memory technologies, such as memristors and spintronics, offer the ability to physically colocate extreme-density analog weights with logic gates, perfectly mirroring biological synapses. Until these exotic substrates become commercially viable, the near-term future of applied neuromorphic computing lies in massively parallel, asynchronous digital ASICs designed using standard CMOS, serving as a transitional bridge to fully analog systems. ]

#v(2em)
== Future Work <s.futurework>

#serif-text()[ The findings and limitations discussed in this thesis present several promising avenues for future research in neuromorphic engineering: ]

#box-text()[
- *Event-Based Dataset Validation:* Transitioning the experimental framework from static images (MNIST) to native temporal datasets, such as Neuromorphic-MNIST (N-MNIST) or DVS Gesture datasets, to fully exploit the asynchronous dynamics of the evaluated neuron models.
- *Surrogate Gradient Integration:* While this work focused on direct weight transfer and native @stdp, future implementations should evaluate the efficacy of Surrogate Gradient Descent, combining the optimization power of backpropagation with the inference efficiency of spiking dynamics.
- *Hardware Deployment:* Porting the verified @ttfs algorithms and current-accumulating neuron models from GPU simulation onto dedicated neuromorphic silicon (e.g., Intel Loihi) to empirically measure true joule-per-inference energy consumption against classical von Neumann baselines.
]


#pagebreak()

= Conclusion <s.conclusion>

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
