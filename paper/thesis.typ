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


*TODO*

- [ ] proof read, stavefeil, flow, struktur, ordlegging akademisk tone. formler, gjenta til fornøyd
- [ ] figurer
- [ ] referanser
- [ ] metode og resultater handcrafted/kopierte vekter
- [ ] metode og resultater neuromorfisk læring
- [ ] discusion
- [ ] conclusion
- [ ] abstract
- [ ] siste proof read


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

  #serif-text()[
    #lorem(200)
  ]]
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

#v(.5em)
#text(size: 9pt, weight: "medium")[
#h(1fr) Wordcount: #total-words
]

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

= Background <background>

#serif-text()[ This section outlines the historical and theoretical evolution of @ai, reviewing key concepts in modern neuroscience that motivate the methodology used in this thesis.

We begin at a shared origin point, a time when @ai research and neuroscience were intertwined. We then trace the diverging path that led to modern Deep Learning, examining why it has drifted from biological plausibility. Subsequently, we explore the "neuromorphic path". In @biologicalprinciples, we detail the specific physical principles and neuroscientific insights upon which the neuromorphic methods in this thesis is built. In @mltechnicalities, we contrast the architectural mechanics of deep learning and neuromorphic systems, specifically addressing why the former is computationally powerful yet energetically inefficient. We conclude with a review of existing frameworks, identifying their strengths and weaknesses to contextualize the contributions of this work. ]

#v(2em)
== History & Developments <historyanddevelopments>

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

#v(1em)
=== The Perceptron

#serif-text()[ In 1957, Frank Rosenblatt advanced these theoretical concepts by engineering the Perceptron. The "Mark I Perceptron" was a hardware implementation of the neural model, distinguished by a crucial innovation: a weight-adjustment mechanism based on Hebbian principles. Rosenblatt introduced the perceptron learning rule, an iterative algorithm capable of minimizing error automatically. The system processed an input pattern (e.g., a pixelated character) and produced a binary classification. When the output deviated from the target, the algorithm adjusted the weights proportional to the error: strengthening connections that should have contributed to a correct firing and weakening those that led to false positives. ]

#figure(include("figures/perceptron.typ"),caption:[The perceptron model. Inputs $x_i$ are multiplied by weights $w_i$ and summed. If the linear combination $sum x_i w_i$ exceeds the bias $b$, the neuron activates. ])

#serif-text()[ Consequently, the Perceptron was capable of converging on a solution for any problem where the data was linearly separable. This success generated significant enthusiasm, with contemporary reports suggesting that such machines would soon mimic human consciousness @Placeholder.

These expectations were abruptly tempered by theoretical limitations. In 1969, Marvin Minsky and Seymour Papert published Perceptrons, a rigorous mathematical analysis of the architecture. They demonstrated that a single-layer perceptron is fundamentally a linear classifier. While capable of learning operations like AND or OR, it is mathematically incapable of solving the XOR (Exclusive OR) problem. In the XOR case, the classes cannot be separated by a single hyperplane. This proof highlighted a severe boundary on the utility of single-layer networks for complex, non-linear tasks. ]

#figure(include("figures/gates.typ"),caption:[The XOR problem. Unlike AND/OR, the data points for XOR cannot be separated by a single linear boundary.])

#serif-text()[ The publication of Perceptrons coincided with a significant reduction in neural network research funding, a period retrospectively termed the "First AI Winter". It is worth noting that Minsky and Papert acknowledged that a @mlp, a network stacking multiple layers of neurons, could theoretically solve the XOR problem by creating complex, non-linear decision boundaries.

However, a critical algorithmic gap remained: the "credit assignment problem". While researchers knew that hidden layers could represent complex features, there was no known method to propagate error signals back through the layers to adjust the weights of hidden neurons effectively. Rosenblatt’s rule was mathematically valid only for the output layer. The field remained stagnant until a method for training multi-layer networks could be formalized. ]

#v(1em)
=== Deep Learning

#serif-text()[ The critique presented by Minsky and Papert precipitated a contraction in funding; despite this, theoretical inquiry persisted. It was widely hypothesized that the limitations of the single perceptron could be overcome by a @mlp. By organizing neurons into hierarchical layers, the network could theoretically perform successive non-linear transformations on the input space, enabling the formation of complex decision boundaries. The primary impediment was not the architecture itself, but the absence of a viable learning algorithm.

In a single-layer perceptron, error attribution is immediate: if the output deviates from the target, the error is directly derived from the weights of the output layer. However, in a multi-layer architecture, quantifying the contribution of a specific neuron within the "hidden" layers to the final output error presents a significant challenge. This is formally known as the Credit Assignment Problem @Placeholder, and it remained the central theoretical obstacle for over a decade. ]

#figure(include("figures/network.typ"),caption:[A @mlp. By inserting "hidden layers" between input and output, the network can approximate non-linear functions such as XOR. The historical challenge lay in deriving a method to train these intermediate layers.])

#serif-text()[ The solution to this theoretical impasse was popularized in 1986 by Rumelhart, Hinton, and Williams in their seminal paper _Learning representations by back-propagating errors_ @Placeholder. They demonstrated that the Chain Rule of calculus could be applied recursively to propagate the error signal from the output layer backwards through the hidden layers. This algorithm, known as Backpropagation, allowed the network to calculate the gradient of the loss function with respect to every weight in the system. Effectively, it provided a mathematical method to tell each hidden neuron exactly how much it contributed to the total error, finally solving the credit assignment problem.

Unlike Hebbian plasticity, which is local and biological, Backpropagation relies on global error signals and precise backward data flow—mechanisms effectively absent in organic tissue. Consequently, the field of @ann effectively decoupled from neuroscience. It transitioned into a branch of engineering and applied mathematics, prioritizing statistical optimization over biological realism. Paradoxically, it was this abandonment of biological fidelity that enabled the rapid scaling and performance breakthroughs that followed. ]

#pagebreak()
#mini-header()[Achievements]

#serif-text()[ With the training mechanism solved, the field exploded. The combination of Backpropagation, massive datasets, and @gpu hardware led to a "Cambrian Explosion" of neural architectures, each solving domains previously thought impossible for computers.

The revolution began in earnest with computer vision. @cnn:pl, such as AlexNet (2012) @Placeholder and later ResNet @Placeholder, introduced the idea of learning hierarchical features---detecting edges, then shapes, then objects---much like the human visual cortex. This allowed machines to classify images with superhuman accuracy.

Soon after, the focus shifted to sequence data. @rnn:pl and @lstm architectures gave machines a short-term memory, enabling breakthroughs in speech recognition and machine translation. However, the true paradigm shift occurred with the introduction of the Transformer architecture in 2017. By utilizing an "attention mechanism" to parallelize the processing of language, Transformers allowed for the training of massive @llm:pl like the @gpt.

These techniques have even transcended media generation. Deep Learning has solved fundamental scientific problems; notably, DeepMind's AlphaFold utilized these architectures to predict the 3D structure of proteins from their amino acid sequences, a 50-year-old grand challenge in biology @Placeholder. ]

#v(1em)
#mini-header()[Shortcomings]

#serif-text()[ Deep learning has achieved substantial success across various domains. However, this performance relies heavily on computational scaling. The underlying algorithms, while originally inspired by biological principles, have diverged significantly to prioritize mathematical optimization on standard hardware. By simulating neural networks on architectures not designed for them—using algorithms developed to address the specific constraints of the @mlp rather than the broader goal of efficient intelligence—this approach faces fundamental barriers that cannot be overcome merely by increasing hardware resources.

A primary limitation stems from the underlying computer architecture. Modern systems rely on the Von Neumann architecture, which physically separates the processing unit from the memory. Deep neural networks, which are defined by large matrices of synaptic weights, require constant data transfer between these components. For every token generated or inference step performed, billions of weight parameters must be fetched from memory, processed, and written back. This creates a memory bottleneck where system performance is limited not by processing speed, but by the available bandwidth to memory @Placeholder.

Concurrently, the computational cost of training state-of-the-art models is growing exponentially. Deep learning relies on dense matrix multiplications, where the number of operations scales quadratically with the network size. As models grow to encompass trillions of parameters to achieve marginal gains in performance, the hardware requirements become economically and physically unsustainable. This reliance on brute-force scaling yields diminishing returns, suggesting that the current architectural paradigm is approaching an efficiency plateau.

This architectural mismatch manifests acutely in energy consumption. In modern CMOS technology, the energy cost of moving data significantly exceeds the cost of processing it. Retrieving a single byte of data from off-chip DRAM consumes approximately three orders of magnitude more energy than performing a floating-point operation on that data @Placeholder. As highlighted in the introduction, this discrepancy has led to the proliferation of "Megawatt Models." In stark contrast, the human brain operates on approximately 20 watts. By co-locating memory and computation, biological systems manage complex multimodal processing with an energy budget comparable to a dim lightbulb.

Furthermore, backpropagation-based learning exhibits low sample efficiency compared to biological systems. Deep learning models often require millions of examples to establish robust representations, whereas biological agents demonstrate "one-shot" or "few-shot" learning capabilities. Additionally, the resulting models function as "black boxes." While empirically effective, their distributed internal representations are often opaque, making it difficult to trace the causal logic behind specific decisions or failures.

These limitations can be traced to the deviation from biological constraints. To solve the training problem, mainstream AI adopted mechanisms that are biologically implausible. Backpropagation relies on a global error signal and requires the backward pass to use the exact same synaptic weights as the forward pass (the "weight transport problem"). In biological tissue, synapses are unidirectional, and there is no known mechanism for a neuron to access the weight of a downstream synapse to calculate a gradient.

While a detailed technical analysis of these inefficiencies is presented in @mltechnicalities, the broader implication is clear: we have achieved artificial intelligence at the cost of efficiency and explainability. This realization has renewed interest in alternative architectures that align more closely with biological principles. Although the majority of research has focused on deep learning, a parallel subset of the field has continued to investigate systems that mimic the physical operation of the nervous system. ]

#v(1em)
=== Birth Of Neuromorphic

#serif-text()[ While the artificial intelligence community debated symbolic logic versus connectionism during the "AI Winter," significant developments were occurring in hardware physics. In the late 1980s at Caltech, physicist Carver Mead—a pioneer of @vlsi design—began to question the trajectory of digital computing.

Mead observed that while digital computers were becoming exponentially faster, they were also becoming less efficient in terms of energy per operation. He noted that using transistors as rigid, high-power switches to perform boolean logic was energetically wasteful compared to the biological systems they aimed to emulate.

In 1990, Mead published his seminal paper, _Neuromorphic Electronic Systems_ @Placeholder, coining the term "neuromorphic" to describe hardware that mimics the biological structure of the nervous system. His thesis proposed that rather than simulating neural equations via software on digital computers, engineers should construct physical hardware that exploits the same physical laws as the biological nervous system.

The foundational insight of the field was the physical analogy between silicon physics and ion-channel physics. In standard digital electronics, transistors are operated in "strong inversion," driven by high voltages to act as binary switches. Mead realized that a single transistor, operating in its "subthreshold" region, follows the same exponential Boltzmann statistics that govern the flow of ions through biological channels.

This realization implied that a single transistor could physically compute the non-linear functions used by biological neurons, but with significantly higher speed and lower power consumption. Consequently, synaptic functions could be implemented by single transistors rather than complex arrangements of logic gates.

To demonstrate this concept, Mead and his doctoral student Misha Mahowald developed the _Silicon Retina_ in 1991 @Placeholder. Unlike a standard camera, which captures full frames at fixed intervals (generating redundant data), the Silicon Retina operated asynchronously. It utilized analog circuits to compute spatial and temporal derivatives directly on-chip, outputting discrete "events" only when local light intensity changed.

This event-driven approach solved the redundancy problem inherent in frame-based sampling. If the scene remained static, the system transmitted no data and consumed negligible energy. This demonstrated that by aligning the hardware physics with the computational task, sensory information could be processed with a fraction of the power required by conventional digital systems.

Since the inception of neuromorphic computing, neuroscience has also advanced significantly. While Mead’s early work was based on the physical intuition of the transistor, modern neuromorphic engineering now incorporates a richer understanding of neuronal dynamics, synaptic plasticity, and network architecture. To advance the field, we must combine these foundational hardware insights with the principles of modern mechanistic neuroscience. ]
 
#pagebreak()

== Biological Principles <biologicalprinciples>

#serif-text()[ The biological brain remains the gold standard for energy-efficient, robust, and adaptive computation. Since the establishment of the Neuron Doctrine, modern neuroscience has uncovered the specific physical mechanisms that underpin this efficiency. To engineer systems that truly rival biological performance, we must transcend the "spherical cow" abstractions of early cybernetics. We cannot simply mimic the brain's output; we must emulate its internal dynamics. This requires viewing the neuron not as a static summing unit, but as it functions in reality: a complex, time-dependent, and event-driven processor.

This section provides a mechanistic overview of the nervous system, translating biological observations into the computational primitives required for neuromorphic engineering. It explores the structural hierarchy of the neuron, the physics of the action potential, and the mathematical models used to capture these dynamics in silicon. ]

#v(1em)
=== Neuron Structure & Function

#serif-text()[ In @historyanddevelopments we established the neuron as the fundamental computational unit of the brain. While it shares standard cellular machinery like mitochondria and a nucleus with other cells, it is morphologically specialized for information transmission. A neuron consists of three functional zones: ]

#box-text()[
- *The Input (Dendrites)*: A branching tree structure that collects signals from thousands of upstream neurons. This is where inputs are integrated.
- *The Integration Zone (Soma)*: The cell body where electrical potentials from the dendrites summate.
- *The Output (Axon)*: A long, cable-like structure that transmits the neuron's own signal to downstream targets.
]

#serif-text()[ The neuron exhibits a distinct morphological polarization that dictates the direction of information flow. The process begins at the "dendritic arbor", a complex branching structure that maximizes the surface area for synaptic connectivity. These dendrites serve as the primary receptor sites, where neurotransmitters binding to post-synaptic terminals induce local conductance changes. These signals propagate passively toward the soma (cell body), the neuron's central processing unit. The soma acts as an integrator, spatially and temporally summing the incoming synaptic currents. Finally, the processed signal is transmitted via the axon, a singular, elongated projection. In many vertebrate neurons, the axon is insulated by a myelin sheath, which facilitates saltatory conduction—a mechanism that allows high-speed signal propagation over long distances with minimal signal degradation. ]

#figure( image("figures/neuron.png", width:40%), caption: [The morphological structure of a biological neuron, illustrating the directional flow of information from dendritic input to axonal output.])

#serif-text()[ Functionally, the neuron operates as an electrochemical system enclosed by a cell membrane, known as the "lipid bilayer". This membrane is a thin, fatty structure that is impermeable to ions, acting as an electrical insulator. However, the fluids inside and outside the cell are conductive electrolytes. Consequently, the interaction between the insulating membrane and the conductive fluids creates a biological capacitor, capable of storing charge.

By actively pumping sodium ($"Na"^+$) out and potassium ($"K"^+$) in via the $"Na"^+$-$"K"^+$ ATPase pump, the cell maintains an electrochemical gradient across this capacitor, resulting in a stable "resting potential" of approximately $-70$ mV.

Computation occurs through the modulation of this voltage by competing synaptic inputs. excitatory inputs cause ion channels to open, allowing positive ions to influx; this reduces the negative charge (depolarization) and pushes the potential toward the firing threshold. Conversely, inhibitory inputs activate channels for negative ions (like Chloride, $"Cl"^-$), driving the potential away from the threshold (hyperpolarization). The soma integrates these opposing push and pull signals. If the aggregate membrane potential surpasses a critical threshold (approximately $-55$ mV), the system undergoes a bifurcating phase transition. Voltage-gated sodium channels cascade open, triggering an @ap—a rapid, non-linear depolarization spike that propagates down the axon. This mechanism is governed by the "all-or-nothing" principle: the output is discrete and binary, effectively filtering out sub-threshold noise. ]

#serif-text()[ Immediately following a spike, the neuron enters a "refractory period" during which ion gradients are restored, imposing a hard limit on the maximum firing frequency and ensuring the temporal separation of events.

It is important to acknowledge that the biological brain exhibits significant cellular diversity beyond this idealized model. The nervous system contains non-neuronal cells known as "glia", which provide structural support and manage energy delivery, though they are generally not considered direct participants in fast information transmission. Additionally, while the vast majority of cortical neurons communicate via uniform action potentials (spikes), certain sensory neurons utilize "graded potentials", where the signal amplitude varies continuously. However, as spiking neurons represent the dominant computational paradigm for information processing in the cortex, this thesis focuses exclusively on the spiking model as the basis for neuromorphic emulation. ]

#v(1em)
=== Action Potential & Spike Trains <actionpotentialandspiketrains>

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

#serif-text()[ This abstraction allows the post-synaptic effect to be modeled using linear systems theory. In neuron models that use this framework, the interaction is treated as instantaneous charge deposition: the arrival of a delta function $delta(t-t_f)$ imparts a discrete step-change to the post-synaptic current. This mimics the rapid opening of ion channels without requiring the computational overhead of simulating the complex voltage trajectory. ]

#figure(include("figures/spiketrain.typ"),caption:[Transformation of continuous membrane voltage (top) into a discrete spike train (bottom).])

#serif-text()[ The shift from continuous values to discrete spike trains fundamentally alters the computational paradigm, moving from spatial representations (magnitude-based) to spatio-temporal representations (time-based). ]


#v(1em)
=== Neuron Models <neuronmodels>

#serif-text()[ In the quest to simulate the brain, there exists a fundamental trade-off between biological realism and computational efficiency. At the high end of the spectrum lie conductance-based models, most notably the Hodgkin-Huxley model. This formalism describes the neuron not as a simple computational unit, but as an electrical circuit with variable resistors representing the precise, non-linear opening and closing dynamics of specific ion channels (sodium, potassium, leak) @Placeholder.

Large-scale initiatives, such as the Blue Brain Project, utilize even more granular "multi-compartment" models. These simulations treat the neuron as a complex 3D structure, discretizing the dendritic arbor and axon into hundreds of segments to model how current flows through the specific morphology of the cell @Placeholder. While invaluable for pharmacological research, these models are computationally prohibitive for large-scale neuromorphic engineering. Simulating a mere second of biological time for a small network using these equations requires supercomputing resources.

To build practical, scalable neuromorphic hardware, we must abstract these biophysical details into a phenomenological model. We seek a mathematical framework that captures the essential computational properties—integration, leakage, and thresholding—without simulating the underlying molecular physics. ]

#v(1em)
#mini-header()[ The Leaky Integrate-and-Fire (LIF) Model ]

#serif-text()[ The standard approximation used in neuromorphic engineering is the @lif model. This framework aligns perfectly with the "point process" abstraction established in the previous section, as it treats action potentials as instantaneous, discrete events. Its state is defined by a single scalar variable, the membrane potential $u(t)$. The sub-threshold dynamics are governed by a linear differential equation analogous to a simple $R C$ (Resistor-Capacitor) circuit: ]

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

#figure( kind: "eq", supplement: [Equation], caption: [The discrete firing and reset condition.], [
$ "If " u(t) > theta.alt arrow cases( "Emit spike: " S(t) arrow S(t) + delta(t), "Reset voltage: " u(t) arrow u_"reset", "Pause integration for " t in (t_f, t_f + Delta_"ref"] ) $
])<lif_reset>

#serif-text()[ This equation represents the engine of most neuromorphic algorithms. It defines a system that integrates information over time and leaks it to ensure temporal relevance. However, once $u(t)$ crosses a threshold $theta.alt$, the linearity breaks and the neuron emits a spike and $u(t)$ is manually reset to $u_"reset"$. ]

#v(1em)
#mini-header()[ The Generalized (Adaptive) LIF Model ]

#serif-text()[ While the standard @lif model is efficient, it is a one-dimensional system. Its state is strictly determined by $u(t)$. Consequently, it supports only a limited set of firing modes, primarily tonic spiking (regular firing under constant input). It cannot replicate complex non-linear behaviors observed in the cortex, such as bursting (clusters of rapid spikes followed by silence) or spike-frequency adaptation (slowing down after sustained activity).

To capture these dynamics without reverting to the heavy Hodgkin-Huxley equations, we employ the @glif model. This extends the system by introducing a second state variable, $w(t)$, often called the adaptation variable. ]

#figure( kind: "eq", supplement: [Equation], caption: [The Adaptive GLIF system. The adaptation variable $w$ provides negative feedback, enabling complex dynamics like bursting and adaptation.], [ $ tau_m (dif u)/(dif t) &= -(u - u_"rest") + R I(t) - w

tau_w (dif w)/(dif t) &= a(u - u_"rest") - w $

])<glif_eq>

#serif-text()[ In this coupled system, $w$ acts as a negative feedback loop. Every time the neuron spikes, $w$ is incremented by a constant $b$, which acts as a drag or fatigue on the membrane potential. This simple addition allows the model to exhibit sophisticated neuro-computational properties, bridging the gap between silicon efficiency and biological complexity.

The introduction of the second state variable $w$ fundamentally alters the mathematical nature of the model. While the standard @lif is a one-dimensional system that simply moves toward a threshold, the @glif constitutes a two-dimensional dynamical system. This increased dimensionality allows the model to exhibit distinct bifurcations—qualitative changes in topological behavior that occur as the input current parameter is varied.

From a dynamical systems perspective, the computational repertoire of a neuron is defined by the type of bifurcation that transitions it from a resting state (fixed point) to a spiking state (limit cycle). The @glif formulation supports both primary classes of neuronal excitability: ]

#box-text()[
- *Class I (Saddle-Node on Invariant Circle)*: In this regime, the neuron can fire at arbitrarily low frequencies. The frequency-current ($f-I$) curve is continuous, meaning the firing rate increases smoothly from zero as the input current increases. These neurons act as "Integrators," essentially converting the amplitude of the input signal into a proportional firing frequency. The standard LIF model is restricted exclusively to this class.

- *Class II (Andronov-Hopf Bifurcation)*: In this regime, the onset of repetitive firing is discontinuous. The neuron cannot fire at low rates; as the input current exceeds a critical value, the system jumps immediately to a specific non-zero firing frequency. These neurons often exhibit sub-threshold oscillations before firing, effectively acting as "Resonators." They preferentially respond to inputs matching their intrinsic resonant frequency while filtering out non-resonant signals. ]

#serif-text()[ By adjusting the coupling parameters between the membrane potential $u$ and the recovery variable $w$, the @glif model can be tuned to operate in either regime. This flexibility allows a single mathematical model to emulate diverse biological behaviors, from the simple integration of sensory transducers to the complex oscillatory synchronization found in cortical interneurons. ]

#figure(include("figures/bifurcation.typ"), caption:[The two primary firing modes determined by bifurcation dynamics. (Left) Class I Integrators show a smooth frequency response. (Right) Class II Resonators (Hopf) show a discontinuous jump and sub-threshold oscillations.])

#serif-text()[ It is natural to question whether such a mathematically reduced model can genuinely capture the behavior of biological neurons. While the @glif model discards the specific ionic mechanisms of the Hodgkin-Huxley equations, empirical validation suggests that it retains the essential computational dynamics.

In the 2008 _Quantitative Single-Neuron Modeling Competition_ @Placeholder organized by the INCF, various models were tested on their ability to predict the precise spike times of real cortical neurons recorded in vitro. Unexpectedly, simple phenomenological models like the Generalized LIF (specifically the Adaptive Exponential Integrate-and-Fire, or AdEx) outperformed highly detailed biophysical models.

The reason for this counter-intuitive success is parameter sensitivity. Complex conductance-based models have dozens of unobservable parameters (channel densities, pore open probabilities) that are difficult to tune. In contrast, the GLIF model captures the "net effect" of these mechanisms—integration, thresholding, and adaptation—using macroscopic parameters that can be robustly fitted to data @Placeholder.

As demonstrated by Izhikevich (2003), this simple system of two differential equations is capable of reproducing all known firing patterns observed in the mammalian cortex, including regular spiking, intrinsic bursting, and chattering @Placeholder. ]

#figure(include("figures/izhikevichpatterns.typ"), caption:[The Generalized LIF model is capable of reproducing the diverse firing patterns of biological cortical neurons, as categorized by Izhikevich (2003) @Placeholder.])

#serif-text()[ Consequently, for the purpose of neuromorphic engineering—where the goal is to emulate the computation (the timing of information processing) rather than the chemistry—the GLIF model represents a good trade-off between fidelity and efficiency. ]

#v(1em)
=== Neural Coding <neuralcoding>

#serif-text()[ In classical digital computing, information is represented by combining bits into richer structures, such as floating-point numbers. For instance, the luminance of a pixel is typically stored as a discrete 8-bit or 32-bit integer. Conversely, analog electronics represent values as continuous currents or voltages, offering infinite resolution within the dynamic range of the hardware.

#figure(include("figures/digitalanalogrepresentation.typ"), caption:[ Digital left analog right representation])

The biological brain occupies a unique middle ground. While neurons operate using analog membrane potentials, their communication output—the action potential—is discrete and binary. As established in @actionpotentialandspiketrains, the waveform of a spike is stereotypical; it looks like a "digital bit" in amplitude. However, unlike a digital computer which is synchronized to a rigid clock, these spikes occur in continuous time. Therefore, the information in the nervous system is not stored in the shape of the signal, but in the structure of the spike train itself.

Deciphering the "Neural Code"—the set of rules by which sensory stimuli are translated into these spike sequences—remains one of the central challenges in neuroscience. Currently, several coding schemes are hypothesized to coexist, each offering different trade-offs between latency, information density, and robustness. ]

#v(1em)
#mini-header()[ Rate Coding ]

#serif-text()[ The most traditional interpretation of neural activity is rate coding. In this paradigm, information is conveyed by the mean firing frequency of a neuron over a specific temporal window. A strong stimulus (e.g., high pressure on skin) elicits a high firing rate, while a weak stimulus results in sparse activity.

This model effectively treats the neuron as an Analog-to-Digital converter where the precise timing of individual spikes is treated as noise; only the average count carries the signal. While rate coding is robust and easily observed in motor neurons, it suffers from a fundamental latency barrier. To estimate a rate with reasonable precision, the post-synaptic neuron must integrate spikes over a significant duration (tens or hundreds of milliseconds). This contradicts the rapid reaction times (often $<100$ ms) observed in biological agents, suggesting that rate coding alone cannot account for time-critical processing. ]

#figure(include("figures/rateencoding.typ"), caption:[Rate Coding: The stimulus intensity is encoded in the frequency of the spike train. Stronger stimuli elicit more spikes per second.])

#v(1em)
#mini-header()[ Temporal Coding ]

#serif-text()[ To explain the speed of biological processing, neuromorphic engineering emphasizes temporal coding. In this regime, the precise timing of a spike carries significant information. A primary example is @ttfs coding.

In a @ttfs scheme, the intensity of a stimulus is inversely mapped to the latency of the response relative to a stimulus onset. A stronger input causes the neuron to integrate and cross the threshold faster, firing earlier than neurons receiving weak inputs. This shifts the computational model from counting spikes to a "race" between spikes.

In a network utilizing lateral inhibition (@wta), the first neuron to fire inhibits its neighbors, allowing a decision to be made as soon as the first meaningful bit of data arrives. This eliminates the need to wait for a time window to close, drastically reducing latency. Furthermore, since @ttfs coding prioritizes the strongest signals, it acts as a natural filter: the most prominent features arrive first, allowing the system to process signal over noise. ]

#figure(include("figures/temporalcoding.typ"), caption:[Temporal Coding (@ttfs): Stimulus intensity is encoded in the latency of the response. Stronger inputs ($I_1$) trigger an earlier spike ($t_1$) compared to weaker inputs ($I_2$).])

#v(1em)
#mini-header()[The Phase Ambiguity Problem]

#serif-text()[ A critical challenge in temporal coding is the need for a temporal reference frame. In Rate Coding, the "phase" (absolute timing) is irrelevant. However, in Temporal Coding, a spike at time $t$ only has meaning relative to a reference $t_0$. If the receiver does not know when the stimulus started, it cannot decode the latency.

In engineering, this is solved by a global clock or a "frame start" signal. In the brain, evidence suggests that background oscillatory rhythms (brain waves, such as theta or gamma cycles) may serve as this global reference, allowing populations of neurons to synchronize their "clocks" and decode relative timings accurately. ]

#figure(include("figures/phaseambiguity.typ"), caption:[The phase ambiguity problem in temporal encoding. Spikes occurring at the same relative phase ($phi_1$ and $phi_2$) across different oscillation cycles are mathematically indistinguishable ($phi_1 = phi_2 (mod 2pi)$). Without a mechanism to track the global cycle count, downstream neurons cannot determine whether a spike represents a delayed response to a previous stimulus or an early response to a new one.])

#v(1em)
#mini-header()[ Population & Sparse Coding ]

#serif-text()[ While single-neuron codes provide the basic signaling mechanism, the brain employs ensemble strategies to ensure robustness and precision. Crucially, population coding is not an alternative to rate or temporal coding; rather, it functions as a higher-order representation that aggregates these signals.

In population coding, variables are represented by the joint activity of a large ensemble of neurons. A classic example is found in the Primary Visual Cortex (V1), where neurons are tuned to specific edge orientations. A single neuron might respond maximally to a vertical bar ($90 degree$), but will also fire weakly for $80 degree$ or $100 degree$.

Relying on a single neuron introduces ambiguity: a weak response could indicate a perfect stimulus at low contrast, or an imperfect stimulus at high contrast. The brain resolves this by reading the population vector—the weighted average of activity across the entire local group. By combining the noisy, broad tuning curves of many neurons, the network can reconstruct the stimulus orientation with a precision far greater than that of any individual cell. ]

#figure(include("figures/tuningcurves.typ"), caption:[Population Coding in the Visual Cortex. (A) Individual neurons have broad "tuning curves" centered on a preferred orientation. (B) The precise stimulus angle is recovered by decoding the population vector sum.])

#serif-text()[ Furthermore, the brain optimizes for metabolic efficiency through sparse coding. This theory posits that neural systems minimize the number of active neurons required to represent a stimulus. At any given moment, out of billions of neurons, only a tiny fraction are firing. This stands in contrast to "dense" coding (where many units participate) or "local" coding (the hypothetical "grandmother cell" where one unique unit represents one unique object). Sparse coding strikes a mathematical balance: it creates a representation that has a high capacity for information but consumes minimal energy. ]

#figure(include("figures/populationcoding.typ"), caption:[Comparison of coding densities. Sparse coding (right) activates a minimal subset of neurons to represent a feature, maximizing energy efficiency compared to dense coding.])

#v(1em)
#mini-header()[ Coexistence of Codes ]

#serif-text()[ It was historically theorized that the brain utilizes a single universal code. However, modern evidence suggests that these schemes are not mutually exclusive but rather complementary @Placeholder. A neural circuit may utilize @ttfs for the initial rapid response (alerting the system to a change) and transition to rate coding to maintain a sustained muscle contraction or represent a static value.

Neuromorphic systems often adopt a hybrid approach: using temporal codes for the energy-efficient transmission of sparse sensory events, and rate-based readouts for interfacing with standard control systems or actuators. ]

#v(1em)
=== Neural Networks <networks>

#serif-text()[ Having established the mathematical description of the individual neuron, we now turn to the collective behavior of these units. A single neuron, regardless of its dynamical complexity, is of limited computational utility in isolation. Functional intelligence emerges only when these units are organized into specific structural topologies.

The brain is not a random mesh of connections; it is constructed from recurring architectural "motifs" that appear across various cortical areas. Understanding these motifs is essential for designing neuromorphic systems that transcend simple feed-forward processing. ]

#v(1em)
#mini-header()[Synaptic Efficacy & Weights]

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
#mini-header()[ Directionality]

#serif-text()[ Structurally, neural topologies can be categorized by the flow of information.

In sensory peripheries (such as the retina) and early processing stages, information flows unidirectionally from input to output. This topology supports rapid, reflex-like feature extraction. This configuration is known as a feed-forward network, which is mathematically equivalent to a Directed Acyclic Graph (@dag) and serves as the standard architecture for most Deep Learning @cnn:pl.

In higher cognitive areas, the dominant topology is recurrence. Neurons form feedback loops, connecting back to themselves or to distinct layers. This recurrence introduces a time component to the computation, transforming the network into a dynamical system where the current output depends not only on the input but on the network's previous state (history). ]

#figure(include("figures/connectivity.typ"), caption:[Network topologies. (A) Feed-Forward. (B) Recurrent.])

#v(1em)
#mini-header()[The Synaptic Hypothesis: Structure As Function]

#serif-text()[ It is a fundamental premise in computational neuroscience that the neuron operates largely as a generic processing unit. While distinct neuronal subtypes exist, a pyramidal neuron in the visual cortex operates on electrophysiological principles identical to those of a neuron in the motor cortex. Consequently, the functional identity of a neural circuit—the specific architecture that determines whether a network processes visual stimuli or governs motor control—is derived principally from the topology and synaptic efficacy of its interconnections.

This paradigm, known as the Synaptic Hypothesis, posits that the physical configuration of synaptic weights (the "Connectome") constitutes the substrate for all long-term memory and learned skills. Unlike artificial systems where data is distinct from the processor, biological memory is realized as a specific configuration of the structural connections themselves. Therefore, the formation of a new memory necessitates a physical alteration of the neural "hardware."

Philosophically, this architecture challenges the ontological distinction between "data" and "program." In a synaptic system, a memory is not a static artifact retrieved from a repository, but a latent computational potential. To "remember" is not to read, but to run; the retrieval of a memory is the active reconstruction of a neural state based on the compressed statistical regularities encoded in the weights. Thus, every act of memory retrieval is inherently a computational process—a simulation regenerated on demand rather than a file accessed from storage.

This dynamic nature implies that memory is labile; because retrieval is a constructive process, activating a memory trace renders it temporarily malleable, allowing for reconsolidation—the updating of old memories with new context. Furthermore, this distributed storage ensures graceful degradation; unlike a digital file that becomes unreadable if a segment is corrupted, a synaptic memory persists as a robust statistical correlation, fading in resolution rather than failing catastrophically under damage.

It must be noted, however, that the brain is not a uniform blank slate at creation. Critical functional systems, such as the visual and motor cortices, are initialized with a conserved topology during embryonic development. All humans share these fundamental circuits; postnatal learning in these regions is characterized not only by weight adjustment but by critical periods of experience-dependent pruning and refinement, rather than the construction of architecture from the ground up. ]

#v(1em)
#mini-header()[ Inhibition Patterns]

#serif-text()[ A ubiquitous micro-circuit motif in the cortex is lateral inhibition. In this configuration, an active excitatory neuron stimulates distinct inhibitory interneurons, which in turn suppress the activity of neighboring excitatory neurons. This competition engenders a @wta dynamic: as one neuron—representing a specific feature or decision—becomes active, it effectively silences its competitors. In the context of neuromorphic engineering, @wta circuits are indispensable; they provide a physical mechanism for both noise reduction, by actively suppressing weak, sub-threshold signals, and categorical decision making, enabling the circuit to autonomously select the most salient option without the need for a central processor to sort or compare values. ]

#figure(include("figures/lateralinhibition.typ"), caption:[The mechanism of lateral inhibition. (A) A highly stimulated neuron in the input layer strongly excites its corresponding output neuron while simultaneously sending lateral inhibitory signals to its immediate neighbors. (B) This architectural motif acts as a spatial filter, producing a contrast enhancement effect. A broad input stimulus (dashed blue line) is transformed into a sharper output response (solid purple line) characterized by an amplified center and suppressed surroundings (a "Mexican hat" profile), thereby sharpening signal boundaries.])

#serif-text()[ While lateral inhibition processes information in the spatial domain, Feed-Forward Inhibition (FFI) operates in the temporal domain. Structurally, this motif bifurcates an input signal into two parallel pathways: a direct excitatory route to the target neuron, and a disynaptic inhibitory route that reaches the same target with a slight synaptic delay. This architecture creates a narrow "temporal window of opportunity." Because the excitation triggers the neuron immediately before the delayed inhibition abruptly truncates the response, the neuron is prevented from integrating noise over extended durations. Consequently, FFI forces the neuron to function as a precise Coincidence Detector rather than a sluggish integrator, a dynamic that is fundamental to sound localization in the auditory cortex and fine-grain timing in the somatosensory system. ]

#figure(include("figures/feedforwardinhibition.typ"), caption:[Feed-Forward Inhibition (FFI). The input excites the target but also drives an inhibitor that shuts the target down shortly after. This creates a precise temporal window for firing.])

#serif-text()[ Distinct from the competitive nature of lateral inhibition, Feedback Inhibition functions as a local regulatory loop. In this circuit, an active excitatory neuron recruits an inhibitory interneuron, which subsequently projects back to suppress the original sender. This negative feedback loop serves two critical engineering functions. First, it provides homeostatic Gain Control, dynamically compressing the signal range to prevent neuronal saturation during high-intensity input. Second, the inherent conduction delays within the loop induce rhythmic firing in the population. This mechanism is the primary driver of Gamma frequency (30−80 Hz) oscillations, which are hypothesized to facilitate the synchronization of communication between distant cortical regions. ]

#figure(include("figures/feedbackinhibition.typ"), caption:[Feedback Inhibition. The active neuron recruits an inhibitor to suppress itself, creating a self-regulating negative feedback loop used for gain control and rhythm generation.])

#v(1em)
#mini-header()[System Example: The Visual Hierarchy]

#serif-text()[ The primate visual system serves as the archetypal biological model for neuromorphic architecture. Rather than processing images as static, monolithic frames, the cortex operates as a hierarchical cascade of feature extraction, mathematically approximating a deep Directed Acyclic Graph (@dag). This processing pipeline begins at the sensor level and progressively abstracts data through specialized stages.

Visual processing initiates in the retina, which functions not merely as a passive camera sensor but as a pre-processing neural computer. Photoreceptors connect to Retinal Ganglion Cells (RGCs) via a lateral inhibition architecture known as "Center-Surround." This configuration creates two complementary cell types: On-Center cells, which fire when a bright stimulus is surrounded by darkness, and Off-Center cells, which respond to dark stimuli on bright backgrounds. This antagonist arrangement effectively acts as a hardware-level convolution filter (specifically, a Difference of Gaussians). By discarding redundant background data and transmitting only contrast changes, the retina performs significant data compression and edge enhancement before the signal ever reaches the brain. ]

#figure(include("figures/retinalreceptivefields.typ"), caption:[Retinal Receptive Fields. (A) On-Center cell firing logic. (B) Off-Center cell firing logic. This lateral inhibition acts as an edge-enhancement filter.])

#serif-text()[ Upon reaching the Primary Visual Cortex (V1), the data undergoes a dimensional transformation from "dots of light" to "geometric primitives." In a mechanism first described by Hubel and Wiesel, the outputs of multiple aligned On/Off-Center cells converge onto single Simple Cells via strong excitatory synapses. This convergence creates a neuron acting as a band-pass Gabor filter, capable of detecting specific spatial frequencies and orientations. Consequently, a V1 neuron does not represent a pixel, but rather a concept: a vertical edge, a 45-degree slant, or a horizontal boundary. ]

#figure(include("figures/hubelwieselmodel.typ"), caption:[The Hubel & Wiesel Model. Outputs from distinct Retinal/LGN cells converge to form V1 Simple Cells, creating orientation-selective edge detectors.])

#serif-text()[ Following feature extraction in V1, the architecture bifurcates into two distinct, parallel processing streams, known as the Dual-Stream Hypothesis. The Ventral Stream ("What" Pathway) projects to the temporal lobe, functioning as a deep feed-forward network that hierarchically constructs object identity—progressing from V1 edges to V2 textures, V4 shapes, and finally IT Cortex object recognition. Conversely, the Dorsal Stream ("Where" Pathway) projects to the parietal lobe and is specialized for high temporal resolution. Rather than identifying objects, this stream calculates the optical flow and spatial coordinates required to guide motor actions. Modern neuromorphic pipelines, particularly those utilizing event cameras, mimic this biological split to separate the computationally expensive task of object recognition from the latency-critical task of motion tracking. ]

#figure(include("figures/visualstreams.typ"), caption:[The Dual-Stream Hypothesis. Visual information splits into the Ventral stream for object recognition and the Dorsal stream for spatial navigation.])

#v(1em)
#mini-header()[Macro-Circuit Motifs]

#serif-text()[ While micro-circuit motifs govern local signal processing, the brain's broader computational capabilities—such as working memory and global data integration—emerge from specific large-scale structural organizations. Although the scope of this thesis focuses on fundamental components, understanding these macro-motifs is essential for conceptualizing how neuromorphic systems can scale beyond simple pattern recognition to achieve cognitive reasoning and persistent state maintenance.

At the topological level, biological neural networks differ significantly from both regular lattices and random graphs, exhibiting instead a "Small-World" architecture. In this configuration, nodes form tightly knit local clusters (high clustering coefficient) while simultaneously maintaining sparse "long-range" connections that bridge distant clusters. For neuromorphic engineering, this topology represents a critical optimization problem: it minimizes the physical wiring cost and metabolic overhead while ensuring that the path length between any two nodes remains short. This allows for rapid global synchronization of data without the prohibitive spatial requirements of a fully connected network. ]

#serif-text()[ Within the recurrent topology described earlier, the challenge of maintaining information over time without constant external input is solved through Attractor Networks. Mathematically, these circuits function as recurrent dynamical systems where the network's energy landscape contains stable fixed points, or "basins of attraction." When neural activity enters one of these basins, it settles into a self-sustaining pattern that persists even after the stimulus is removed, effectively acting as the biological substrate for Working Memory.

This stability mechanism is what enables Content-Addressable Memory (associative memory). Information is retrieved not by address, but by "keying" the network with a partial pattern (e.g., a familiar scent). This input pushes the system state near a basin, triggering the circuit to settle into the associated attractor state and effectively re-computing the missing information. These attractors are generally categorized into two types: ]

#box-text()[
- *Discrete Attractors (Point Attractors)*: Used for categorical memory and auto-associative error correction.
- *Continuous Attractors (Line/Ring Attractors)*: Used for encoding continuous variables like spatial orientation and navigation coordinates. ]

#serif-text()[ Finally, the physical instantiation of these functions is not realized through a homogeneous mass of neurons, but through a highly modular architecture known as the Cortical Column. The neocortex is organized into discrete, repetitive units, where each column functions as a canonical microcircuit—a specialized "processing core" containing all necessary layers and inhibitory motifs to process a specific receptive field. This modularity suggests that biological intelligence scales not by inventing new, complex algorithms for every task, but by replicating a standard, versatile circuit. In hardware design, this principle is isomorphic to the "tiling" approach seen in modern neuromorphic chips, where a single optimized processing core is replicated thousands of times to achieve massive scalability and fault tolerance. ]

#v(1em)
=== Biological Learning <bio_learning>

#serif-text()[ As previously established, the functional identity of a neural circuit is not defined by a transient software state, but by its physical hardware configuration. Consequently, "learning" in a biological substrate cannot be viewed as a simple parameter optimization; it is a fundamental morphological process. If structure dictates function, then the acquisition of new skills or memories necessitates the physical restructuring of the connectome itself.

Because the brain lacks a central supervisor or global communication bus, this restructuring must be driven by Locality. A synapse can only change based on information physically available at the cleft: the activity of the pre-synaptic axon, the voltage of the post-synaptic dendrite, and the immediate neurochemical environment. Despite this constraint, the brain successfully credits specific synaptic events with outcomes that occur seconds or minutes later.

This adaptation occurs across multiple timescales and spatial resolutions via two distinct mechanisms: Structural Plasticity (the rewiring of the network topology) and Synaptic Plasticity (the modulation of connection strength). ]

#v(1em)
#mini-header()[Structural Plasticity]

#serif-text()[ While synaptic weight adjustment accounts for rapid learning and pattern recognition, the long-term storage of information and the optimization of energy efficiency are governed by structural plasticity. This mechanism involves the physical genesis (synaptogenesis) and removal (pruning) of synapses and even entire axonal branches. ]

#box-text()[
- *Synaptogenesis*: When neurons are repeatedly co-active but lack a direct connection, the brain can physically grow new dendritic spines and axonal boutons to bridge the gap. This effectively alters the network's topology, creating new pathways for information flow where none existed before.
- *Pruning*: Equally critical is the removal of redundant or noisy connections. During sleep and developmental critical periods, the brain aggressively prunes weak synapses. This "sparsification" reduces metabolic cost and increases the signal-to-noise ratio of the circuit by removing irrelevant pathways. ]

#serif-text()[In the context of the Synaptic Hypothesis, structural plasticity represents the "compiling" of temporary associations into permanent hardware architecture. ]

#figure(include("figures/structuralplasticity.typ"), caption:[Structural Plasticity. (A) Synaptogenesis: The growth of new dendritic spines to form connections. (B) Pruning: The retraction of unused connections to optimize metabolic efficiency.])

#v(1em)
#mini-header()[Synaptic Plasticity]

#serif-text()[ Once a structural connection exists, its efficacy—the magnitude of the post-synaptic response to a pre-synaptic spike—must be tuned. In biological terms, this "weight" corresponds to the amount of neurotransmitter released and the density of receptors on the receiving side. This fine-grained adjustment is governed by local learning rules. ]

#mini-header()[Hebbian Learning: Rate-Based Correlation]

#serif-text()[ The foundational axiom of biological learning was postulated by Donald Hebb in 1949. Hebb proposed that synaptic efficiency is a function of the correlated activity between two neurons. Colloquially summarized as "Neurons that fire together, wire together," this rule implies that the brain learns by detecting statistical regularities in sensory input.

Mathematically, if neuron $A$ consistently takes part in firing neuron $B$, the connection from $A$ to $B$ is strengthened. This mechanism allows the brain to perform unsupervised clustering, physically encoding associations between features that occur simultaneously in the environment (e.g., the smell of smoke and the sight of fire). ]

#v(1em)
#mini-header()[Spike-Timing-Dependent Plasticity (STDP)]

#serif-text()[ Modern neuroscience has refined Hebb’s macroscopic theory into a precise, millisecond-scale mechanism known as @stdp. Unlike rate-based models, @stdp operates on the precise timing of individual action potentials, introducing the critical element of causality.

The @stdp rule adjusts the synaptic weight based on the relative timing difference ($Delta t$) between the pre-synaptic input and the post-synaptic output: ]

#box-text()[
- *@ltp*: If the input spike arrives *before* the output spike ($Delta t > 0$), it implies the input contributed to the firing. The synapse is strengthened to reinforce this causal link.
- *@ltd*: If the input spike arrives *after* the output spike ($Delta t < 0$), the input was irrelevant to the decision. The synapse is weakened. ]

#serif-text()[ This asymmetry allows the network to self-organize, naturally filtering out random noise while reinforcing specific spatiotemporal patterns. ]

#figure(include("figures/stdpcurve.typ"), caption:[The @stdp Learning Curve. Synaptic weight change is plotted against spike timing difference. Pre-before-post timing triggers strengthening (@ltp), while post-before-pre triggers weakening (@ltd).])

#v(1em)
#mini-header()[Homeostatic Plasticity]

#serif-text()[ If Hebbian mechanisms (@ltp) were the sole drivers of plasticity, neural networks would be inherently unstable. A positive feedback loop would emerge where strengthened synapses drive higher firing rates, which in turn induce further strengthening, leading to runaway excitation (seizures). Conversely, unchecked LTD could silence a network entirely.

To maintain stability, the brain employs Homeostatic Plasticity (or Synaptic Scaling). This is a global regulatory mechanism that operates on a slower timescale (minutes to hours). It functions as a negative feedback loop: if a neuron's average firing rate exceeds a target set-point, the cell chemically downscales the strength of *all* its incoming synapses. This ensures that neurons remain within a sensitive dynamic range, preventing saturation regardless of how strong the inputs become. ]

#v(1em)
#mini-header()[Three-Factor Learning: The Path to Reinforcement]

#serif-text()[ While @stdp is excellent for learning correlations, it is agnostic to the *value* of the outcome. A circuit might learn to associate "fire" with "touching," but @stdp alone cannot encode whether this is good or bad. To learn tasks (Reinforcement Learning), the brain utilizes Neuromodulation. ]

In this "Three-Factor Rule," the synaptic update depends on:
#box-text()[
- *The Pre-synaptic activity* (The input).
- *The Post-synaptic activity* (The action).
- *A Global Reward Signal*: The presence of neuromodulators like Dopamine. ]

#serif-text()[ Even if the timing is perfect for @stdp, the synapse is only permanently strengthened if a neuromodulator arrives shortly after, signaling a "reward" or "novelty." This bridges the gap between local synaptic events and global behavioral goals. ]

#pagebreak()

== Technical Details Of Machine Learning <mltechnicalities>

#serif-text()[ This chapter delineates the technical foundations of modern artificial intelligence, contrasting the established paradigms of @dl with the emerging principles of Neuromorphic Engineering. We begin by analyzing the algorithmic architecture of standard Deep Learning, identifying the computational bottlenecks and energy inefficiencies inherent in its reliance on dense matrix multiplication and backpropagation. Subsequently, we introduce the technical framework of neuromorphic engineering, demonstrating how the translation of the biological principles discussed in the previous section—such as sparsity, event-driven processing, and local learning—can yield systems that vastly outperform conventional models in terms of energy efficiency and latency. ]

#serif-text()[ A critical distinction must be drawn between biological plausibility and bio-inspired engineering. From an engineering perspective, the primary objective is functional utility; a mechanism need not faithfully replicate biological reality to be valuable. It is important to recognize that evolution is a "satisficing" process—selecting for traits that ensure survival rather than finding mathematically optimal solutions. Therefore, an engineer may treat the brain merely as a source of heuristic inspiration rather than a blueprint to be copied dogmatically. However, the pursuit of biologically plausible systems remains vital. Not only does it offer potential distinct advantages in robustness and adaptability, but the creation of synthetic systems that respect biological constraints serves as a powerful verification tool for neuroscience, bridging the gap between artificial construction and biological understanding. ]

#v(1em)
=== Optimization

#serif-text()[ To rigorously analyze intelligent systems, we must first abstract the learning process from its biological roots into a formal mathematical optimization problem. In this framework, "learning" is not treated as an emergent property of organic tissue, but as a search problem within a high-dimensional vector space.

Mathematical optimazation is the selection of a "best candiate" with regard to some defined criteria. A simple optimasation problem is finding the minimum of a one-dimentional function $f : RR -> RR$ here the "best canditade" is the lowest real number in the domain of the function. The criteria is "the lowest posible number $x in RR$". If the function is convex it has a and smooth it has a well defiend "global minimum" and the function can be differentiated to find the global minimum. Any quantity that can be mesasured by a criteria can be optimized even functions themselves can be optimised.

Fundamentally, an artificial intelligence model operates as a function approximator. We assume the existence of an unknown underlying function $f: X -> Y$ that perfectly maps inputs to their target outputs, for example such a function could be one that perfectly describes the earth's wether pattern taking into account known (and unkown) physics as well as all the particles with their mass and velocity in the atmosphere. This function may be fully deterministic or stocastic, in any case such a function is yet to be discoverd and the best we can do is to take guess at a much simpler function that still makes usefull predictions even if they are wrong to some degree. By adding tunable parameters to this function we can construct a family of hypothesis functions $f_bold(theta) (bold(x))$ Here, $bold(theta) in RR^d$ represents the state of the system—a vector containing all tunable parameters, such as synaptic weights, biases, or time constants. The dimensionality $d$ of this space corresponds to the degrees of freedom of the model, and the precise configuration of $bold(theta)$ determines the system's behavior.

To guide the search for the optimal parameters $hat(bold(theta))$, one must quantify the divergence between the model's predictions and the ground truth. We define a scalar Loss Function $cal(L)(hat(bold(y)), bold(y))$ that evaluates the error on a single data point, such as the Squared Error for regression or Cross-Entropy for classification. However, optimizing for a single example is insufficient for generalization. Instead, we seek to minimize the Empirical Risk (or Cost Function) $J(bold(theta))$, defined as the average loss over a dataset of size $N$:

$ J(bold(theta)) = 1/N sum_(i=1)^N cal(L)lr(( f(bold(x)_i, bold(theta)), bold(y)_i),size: #200%) $

Consequently, the learning task is reduced to finding the global minimizer of this cost function:

$ hat(bold(theta)) = "argmin"_(bold(theta)) space J(bold(theta)) $

Geometrically, the cost function $J(bold(theta))$ induces a complex structure known as the Optimization Landscape or Error Surface. In modern non-linear systems, this landscape is rarely convex; it is characterized by a multitude of local minima, saddle points, and plateaus. Navigating this non-Euclidean topology to find a low-energy state is the central challenge of AI engineering.

Since closed-form solutions for $hat(bold(theta))$ are generally intractable for complex non-linear functions, we rely on iterative optimization algorithms, principally Gradient Descent. This method relies on the mathematical principle that the gradient vector $nabla_(bold(theta)) J(bold(theta))$—the vector of partial derivatives with respect to all parameters—points in the direction of the steepest ascent of the function. To minimize error, the system must update its state in the direction opposite to the gradient. The update rule for a generic iteration $t$ is given by:

$ bold(theta)_(t+1) arrow.l bold(theta)_t - eta nabla_(bold(theta)) J(bold(theta)_t) $

In this equation, $eta$ (eta) represents the Learning Rate, a hyperparameter that governs the step size of the update. This formulation encapsulates the fundamental loop of modern AI: evaluate the current state, compute the gradient of the error, and incrementally adjust the parameters to descend the error surface toward convergence. ]

#figure(include("figures/gradientdecent.typ"), caption:[The Optimization Landscape. The system seeks to traverse the high-dimensional surface defined by $J(bold(theta))$ to find the global minimum $bold(theta)^*$, using the gradient $nabla J$ as a navigational compass.])

#serif-text()[ Altoguh gradient decent works remarkably well for its simplicity it has limitations. As already mentioned the method can get stuck in local minium. Furthermore the speed and effeciency may at times be slow. especcially for loss landscaped that appear flat. Perhaps the most crucial limitation is that the loss function must be differentialble, otherwise we cannot obtain a gradient. This becomes very important for optimasation of neuromorhpic systems. ]

#v(1em)
#mini-header[Unsupervised Optimization]

#serif-text()[ The optimization framework defined previously describes Supervised Learning, where the system is guided by explicit target labels $bold(y)$. However, relying solely on labeled data is biologically implausible and engineering-wise inefficient. The vast majority of sensory data received by an intelligent agent is unlabeled; the retina receives photons, not pixel-wise classifications.

To handle this, we employ Unsupervised Learning. In this regime, the dataset consists only of input vectors $X = {bold(x)_1, ..., bold(x)_N}$. The optimization objective shifts from minimizing prediction error to minimizing representation error or energy.

Mathematically, the goal is often to learn the underlying probability distribution $P(bold(x))$ of the data or to discover a lower-dimensional manifold $cal(Z)$ that efficiently captures the structure of $cal(X)$. A common formulation is the minimization of Reconstruction Loss (as seen in Autoencoders), where the system attempts to compress the input into a latent code $bold(z)$ and then reconstruct it:

$ J(bold(theta)) = 1/N sum_(i=1)^N || bold(x)_i - f_"decode"(f_"encode"(bold(x)_i; bold(theta))) ||^2 $

Alternatively, in energy-based models (which closely resemble physical thermodynamic systems), the optimization seeks to find a configuration of parameters that minimizes the "energy" of plausible data configurations while maximizing the energy of implausible ones.

This distinction is critical for Neuromorphic Engineering. Biological plasticity rules, such as those discussed in the following sections, are predominantly unsupervised. They function by detecting statistical correlations in the input stream (e.g., "edges generally occur in continuous lines") to build internal representations of the world without external supervision. ]


#v(1em)
#mini-header()[Stochastic vs. Batch Optimization]

#serif-text()[ The formulation above describes Batch Gradient Descent, where the gradient is computed over the entire dataset $N$ before a single update is made. For modern high-dimensional datasets, this is computationally prohibitive. Furthermore, biological learning does not wait to experience "all of life" before adapting; it learns online.

To address this, modern AI employs Stochastic Gradient Descent (SGD) or Mini-Batch Gradient Descent. Instead of computing the exact gradient over $N$, the gradient is approximated using a small random subset (a batch) of data $B << N$:

$ bold(theta)_(t+1) arrow.l bold(theta)_t - eta 1/(|B|) sum_((x,y) in B) nabla_(bold(theta)) cal(L)(f(bold(x)), bold(y)) $

This introduces noise into the optimization trajectory. Paradoxically, this noise is beneficial: it prevents the system from getting stuck in shallow local minima and saddle points, allowing the optimization process to "jitter" its way toward more robust solutions. This mimics the noisy, event-driven updates seen in biological synaptic plasticity. ]

#v(1em)
#mini-header()[Generalization and The Bias-Variance Tradeoff]

#serif-text()[ Strictly minimizing the empirical cost $J(bold(theta))$ carries a risk. If the model capacity ($d$) is too large relative to the data size ($N$), the system may simply "memorize" the training examples, including their noise, rather than learning the underlying function $f$. To understand this failure mode, we decompose the generalization error into three components, known as the Bias-Variance Decomposition: ]

$ "Error" = "Bias"^2 + "Variance" + "Irreducible Error" $

#box-text()[
- *Bias (Underfitting)*: The error introduced by approximating a real-world problem with a simplified model (e.g., trying to fit a curve with a straight line). High bias causes the model to miss relevant relations between features and target outputs.
- *Variance (Overfitting)*: The error introduced by the model's sensitivity to small fluctuations in the training set. A high-variance model captures the random noise in the training data rather than the intended outputs. ]

#serif-text()[ This creates a fundamental tension: increasing model complexity decreases bias but increases variance. The objective of optimization is not zero training error, but finding the "sweet spot" in this tradeoff where the total generalization error is minimized. ]

#figure(include("figures/biasvariance.typ"), caption:[The Bias-Variance Tradeoff. As model complexity increases, bias (underfitting) decreases while variance (overfitting) increases. The optimal model exists at the trough of the total error curve.])

#serif-text()[ To manage this tradeoff, optimization often includes Regularization terms (such as $L_1$ or $L_2$ penalties) that artificially constrain the complexity of the hypothesis space. In biological systems, this regularization is naturally enforced by metabolic constraints—the brain aggressively prunes weak connections to maintain a sparse, energy-efficient topology, effectively reducing variance by limiting the hardware available to overfit noise. ]

#v(1em)
=== Deep Learning Constituents

#serif-text()[ In the context of the optimization framework established previously, Artificial Neural Networks (ANNs) constitute a specific class of hypothesis functions formed by the hierarchical composition of simple, non-linear modules.

While the "Perceptron" introduced in the history chapter served as the atomic unit, modern Deep Learning aggregates these units into high-dimensional layers. Mathematically, a Deep Neural Network with $L$ layers is expressed not as a singular equation, but as a composite function $f_bold(theta)(bold(x))$ mapping an input $bold(x)$ to an output $bold(y)$ through a chain of nested operations:

$ bold(y) = f_L ( ... f_2 ( f_1 ( bold(x) ) ) ) $

This depth is not arbitrary; it allows the system to learn hierarchical representations. The initial layers might detect simple edges or frequencies, while deeper layers recombine these primitives to recognize complex semantic concepts like "faces" or "syntax." ]



#v(1em)
#mini-header()[The Forward Pass: Affine Transformations]

#serif-text()[ The computation of these layers during inference is known as the Forward Pass. For a standard Multi-Layer Perceptron (MLP), each layer $l$ performs two distinct mathematical operations:

1.  Affine Transformation: The input vector $bold(a)^((l-1))$ is multiplied by a weight matrix $bold(W)^((l))$ and shifted by a bias vector $bold(b)^((l))$. This operation is linear and represents a rotation and scaling of the data manifold.
2.  Non-Linear Activation: The result $bold(z)^((l))$ is passed through a non-linear function $sigma(dot)$.

$ bold(z)^((l)) = bold(W)^((l)) bold(a)^((l-1)) + bold(b)^((l)) $
$ bold(a)^((l)) = sigma(bold(z)^((l))) $

This seemingly simple structure contains the "Universal Approximation" capability of neural networks. However, the power of the network is entirely dependent on the choice of the non-linearity $sigma(dot)$. ]

#v(1em)
#mini-header()[Activation Dynamics and Gradient Flow]

#serif-text()[ If a network consisted only of linear transformations (just the $bold(W)x + bold(b)$ part), the entire deep stack would mathematically collapse into a single linear matrix multiplication. The activation function $sigma(dot)$ introduces the necessary non-linearity to model complex real-world data.

However, the choice of $sigma(dot)$ dictates not just expressivity, but trainability. During optimization, the gradient must flow backwards through these functions.

Sigmoid / Tanh (The Old Paradigm): Historically, functions like the Sigmoid were used because they mimic the firing rate of a biological neuron (saturating at 0 and 1). However, their derivatives are strictly less than 1 (max 0.25). As gradients propagate back through many layers, these small numbers multiply, causing the signal to decay exponentially to zero. This is the Vanishing Gradient Problem.
ReLU (The Modern Standard): To enable deep learning, modern networks employ the Rectified Linear Unit, $f(x) = max(0, x)$. Its derivative is either 0 or 1. This "gating" property preserves the magnitude of the gradient, allowing error signals to travel through deep structures without vanishing. ]



#figure(include("figures/activations.typ"), caption:[Activation Functions. The Sigmoid (left) saturates, killing gradients. The ReLU (right) preserves gradient magnitude for positive inputs.])

#v(1em)
#mini-header()[The Backward Pass: Computational Graphs]

#serif-text()[ To update the parameters $bold(W)$ and $bold(b)$, the system must attribute the total error $J(bold(theta))$ to specific weights. This is achieved via Backpropagation.

Conceptually, Backpropagation is simply the recursive application of the Chain Rule of calculus. If we view the network as a computational graph, the gradient for a weight $w$ in layer $l$ is computed by propagating the error $delta$ from the layer above:

$ (partial cal(L)) / (partial w) = underbrace((partial cal(L)) / (partial a), "Error from above") dot underbrace((partial a) / (partial z), "Activation derivative") dot underbrace((partial z) / (partial w), "Input value") $

In modern frameworks (like PyTorch or TensorFlow), this is implemented via Automatic Differentiation (AutoDiff). The system builds a dynamic graph of operations during the forward pass and traverses it in reverse to compute exact gradients. ]

#v(1em)
#mini-header()[Deep Learining Computations: Dense Matrix Operations]

#serif-text()[ While the equations above describe the behavior of individual neurons, modern Deep Learning does not compute them one by one. To achieve the throughput required for training, these operations are vectorized. The affine transformation for an entire layer is executed as a Dense Matrix Multiplication (GEMM):

$ bold(Z) = bold(W) times bold(A) + bold(b) $

Where $bold(W)$ is a matrix of dimensions $N_"out" times N_"in"$. Crucially, the backward pass (gradient computation) also reduces to matrix multiplication, typically involving the transpose of the weight matrix $bold(W)^T$.

This mathematical reality is the defining characteristic of modern AI hardware usage. A deep network is effectively a sequence of massive matrix multiplications. While this structure allows for efficient parallelization on SIMD (Single Instruction, Multiple Data) hardware like GPUs, it creates a massive dependency on memory bandwidth. The entire weight matrix $bold(W)$—which can reach gigabytes in size for modern Transformers—must be loaded into the processor registers for every single inference step.

This structural reliance on moving massive dense matrices provides the context for the primary bottleneck of modern AI, discussed in the following section. ]

#figure(include("figures/matrixmath.typ"), caption:[Deep Learning as Matrix Multiplication. Both forward and backward passes rely on dense matrix-vector products, necessitating high-bandwidth memory access.])

#v(1em)
=== Why Is Deep Learning Inefficient?

#serif-text()[ While the matrix-centric formulation of Deep Learning enables high-throughput parallelization on GPUs, it fundamentally conflicts with the physical constraints of modern computing hardware. As models scale to billions of parameters, the primary bottleneck shifts from algorithmic capability to physical realizability. This inefficiency manifests in three distinct engineering dimensions: data movement, dense computation, and global synchronization. ]

#v(1em)
#mini-header()[The Von Neumann Bottleneck & Data Movement]

#serif-text()[ The most significant physical limitation is the Von Neumann Architecture, which physically separates the Processing Unit from the Memory Unit. For Deep Learning, this is catastrophic. A neural network is defined by its weight matrix $bold(W)$. To perform a single inference step, the processor must fetch the entire weight matrix from off-chip DRAM to on-chip registers, perform the calculation, and write results back.

According to data from Horowitz and Dally @Placeholder, fetching a 32-bit value from off-chip DRAM consumes approximately 640 pJ, whereas performing the floating-point addition on that value consumes only 0.1 pJ. The system burns 99.9% of its energy moving data to the calculator, and only 0.1% actually doing the calculation. ]

#figure(include("figures/vonneuman.typ"), caption:[The Von Neumann Bottleneck. The separation of memory and compute forces massive energy expenditure on data transport.])

#v(1em)
#mini-header()[Dense Processing of Sparse Data]

#serif-text()[ Standard Deep Learning implementations rely on Dense Matrix Multiplication (GEMM). This approach is algorithmically rigid: it executes the same number of operations regardless of the data content.

Real-world data is often sparse (containing many zeros), and the ReLU activation function naturally produces activation maps where 50-80% of values are zero. However, a standard GPU is "blind" to this sparsity. It will dutifully fetch a zero from memory, load it into a register, and multiply it by a weight ($0 times w = 0$), consuming energy and cycles to produce a null result.

In a biological or neuromorphic system, a "zero" is simply the absence of an event. Nothing is transmitted, and absolutely no energy is consumed. Deep Learning's inability to exploit this silence represents a massive structural inefficiency. ]

#v(1em)
#mini-header()[The High Cost of Synchrony]

#serif-text()[ Deep Learning hardware is typically Synchronous, meaning every component operates in lockstep with a global clock. This introduces a severe overhead known as the Clock Distribution Penalty.

To ensure that billions of transistors switch at the exact same moment, the chip must drive a high-frequency clock signal across the entire silicon die. Charging and discharging the capacitive wires of this clock tree occurs billions of times per second, regardless of whether the chip is doing useful work. In many high-performance processors, the clock network alone can consume 30% to 40% of the total power budget.

Furthermore, this global synchronization enforces a "worst-case" latency. If one part of the matrix multiplication finishes early, it must sit idle and wait for the slowest part to finish before the next clock cycle can begin. ]

#v(1em)
#mini-header()[Backpropagation and Global Dependencies]

#serif-text()[ Finally, the learning algorithm itself—Backpropagation—imposes severe constraints on memory and latency. Backpropagation is non-local in both time and space.

To update a specific weight $w_(i j)$ in the first layer of a deep network, the system cannot act immediately. It must:
1.  Wait for the Forward Pass: The input must propagate through the entire network to the output layer to generate a prediction.
2.  Calculate Global Error: The system computes the loss $J(bold(theta))$ based on the global output.
3.  Wait for the Backward Pass: The error gradient must be propagated all the way back from the output to the input.

This creates a Locking Problem. The activations of every intermediate layer must be stored in high-speed memory (VRAM) for the duration of the entire pass, preventing that memory from being reused. This memory footprint grows linearly with network depth, often limiting the size of models that can be trained. Additionally, the weight update is dependent on the global state of the network, meaning a local synapse cannot adapt to local changes instantly; it is shackled to the global error loop. ]

#v(1em)
=== Principles of Neuromorphic Engineering

#serif-text()[ To address the fundamental inefficiencies of the Von Neumann architecture, we turn to the paradigm of Neuromorphic Engineering. The term, coined by physicist Carver Mead in the late 1980s, is derived from the Greek roots _neuro_ (relating to nerves or the nervous system) and _morphe_ (meaning form or shape).

Literally translating to "taking the form of the brain," the term was not intended to describe software simulations of neural networks. Rather, it described a specific hardware design philosophy: the construction of electronic circuits that utilize the inherent physics of silicon to mimic the biophysics of neural tissue.

Mead's foundational insight was that the physical equations governing the flow of ions through biological channels (Boltzmann statistics) are mathematically identical to the equations governing the flow of electrons through a transistor operating in its "subthreshold" (weak inversion) region.

$ I_("channel") prop e^(kappa V_g / U_T) space ( "Transistor" ) <--> I_("membrane") prop e^(V_m / V_T) space ( "Neuron" ) $

Therefore, "Neuromorphic" does not simply mean "AI inspired by the brain." It specifically refers to systems that replicate the physical topology and analog dynamics of the nervous system to achieve computation. While modern interpretations have expanded to include digital implementations (like Intel's Loihi), the core definition remains rooted in mimicking the brain's structural "form"—specifically its parallelism, connectivity, and local processing—rather than just its mathematical outputs. ]

// #figure(include("figures/subthreshold.typ"), caption:[The Physical Isomorphism. Carver Mead demonstrated that the exponential relationship between voltage and current in a subthreshold transistor matches the ion flow in biological membranes, allowing silicon to compute neural functions natively without digital logic.])

#serif-text()[ The inefficiencies described above—the memory wall, the energy cost of dense processing, and the overhead of global synchronization—are not fundamental limits of computation. Rather, they are artifacts of the Von Neumann architecture. As established in the _History & Developments_ section, physicist Carver Mead identified this divergence as early as 1990, proposing that to achieve the efficiency of the brain, we must adopt the physics of the brain.

Neuromorphic Engineering is the translation of the Biological Principles discussed in the previous chapter into silicon hardware. It replaces the rigid, clock-driven logic of standard computing with the adaptive, event-driven dynamics of neural tissue. This approach rests on three architectural pillars that directly address the bottlenecks of Deep Learning: ]

#v(1em)
#mini-header()[Co-location of Memory and Compute (The Synaptic Principle)]

#serif-text()[ To dismantle the Von Neumann bottleneck, neuromorphic architectures eliminate the separation between the processor and the memory. In the biological brain, there is no separate "RAM" module; memory is stored in the synaptic weights themselves, right where the processing (integration of current) occurs.

Neuromorphic chips replicate this by distributing memory across the silicon die. Each artificial neuron possesses its own local memory to store its state and synaptic weights. By processing data *in situ*, the massive energy cost of shuttling weights back and forth is eliminated. This is the engineering realization of the Synaptic Plasticity mechanisms discussed earlier: computation and storage are physically inseparable. ]

#v(1em)
#mini-header()[Event-Driven Asynchrony (The Action Potential Principle)]

#serif-text()[ To address the "Dense Processing of Sparse Data" and the "Clock Distribution Penalty," neuromorphic systems abandon the global clock. Instead, they operate asynchronously, driven strictly by the arrival of data.

This mimics the All-or-None Law of the biological neuron. Just as a neuron remains quiescent until its membrane potential reaches a threshold, a neuromorphic circuit consumes negligible power until an event (a spike) arrives. If a part of the network is not currently processing information, it effectively shuts down. This "activity-gating" ensures that power consumption scales linearly with the complexity of the task, rather than the size of the network—a critical advantage for processing sparse real-world sensory data. ]

#v(1em)
#mini-header()[Sparse Communication (The Spike Principle)]

#serif-text()[ Finally, to solve the bandwidth issue, neuromorphic systems utilize Spikes for communication. Unlike the 32-bit floating-point numbers used in Deep Learning, a spike is a binary event that carries no payload other than its source address and its timing.

This corresponds to the biological principle of Temporal Coding. Information is not encoded in the complex magnitude of a signal, but in the precise timing of events (Inter-Spike Intervals or Time-to-First-Spike). This allows the system to compress high-dimensional information into sparse, energy-efficient pulse trains, drastically reducing the bandwidth required to transmit information between neurons. ]

#v(1em)
#mini-header()[Conclusion of the Framework]

#serif-text()[ By inverting the Von Neumann paradigm, Neuromorphic Engineering offers a path to artificial intelligence that operates within the energy envelope of biological systems. The following sections detail the specific algorithmic implementation of these principles: the Spiking Neural Network (SNN).

Neuromorhic engineering and neuromorphic computing is any system that mimics or takes advantage of key mechanisms of the nervous system. The great promise of neuromorphic is to act as a successor to deep learing and take on tasks which needs greater inteligne ability to adapt without breaking the energy and resource budget. As mentioned Carver Mead pionered this feild and coined the term. But neuroscince has gone a long way since and we can draw more inpiration.

The unifying mechanism that all neuromorhic systems use is the spike or event just like the nervous system coverd in @biologicalprinciples. A neuromorphic camera can use the change in luminosity of a pixel to send out an event, this way only changes in the scene are recorded and the camera becomes super effecient.

Neuromorphic computers can also use the fact that summation of currents in the neuron is analog and can be computed very effeciently using analog summing circuits.

Neuromorhic ideology is in creating algrithms directly embedded in hardware ]

#v(1em)
=== Learning In Neuromorphic Systems

#serif-text()[ In the optimazation and deep learing section we saw that training deep neural networks can be achevied with gradient decent this starts to break down when we try to apply it to neuromor

loss function needs to influence the synaptic weigts.
weights does not know anything about a global signal (execpt for feedback dopamine)
weights have to

Consequently, standard backpropagation cannot be directly applied to @snn:pl. Gradients calculated using the chain rule become zero or undefined at the spiking neurons, preventing error signals from flowing backward through the network to update the weights effectively. This incompatibility represents a substantial obstacle, as it seemingly precludes the use of the highly successful and well-understood gradient-based optimization toolkit that underpins much of modern @ai.

To maintain this critical regime, modern neuromorphic chips implement homeostatic plasticity—algorithms that automatically scale synaptic weights to keep the mean firing rate within a target range, ensuring signals can propagate through deep layers without fading out or exploding.

Surrogate Gradients: A popular approach involves using a "surrogate" function during the backward pass of training. While the forward pass uses the discontinuous spike generation, the backward pass replaces the step function's derivative with a smooth, differentiable approximation (e.g., a fast sigmoid or a clipped linear function). This allows backpropagation-like algorithms (often termed "spatio-temporal backpropagation" or similar) to estimate gradients and train deep SNNs, albeit with approximations. ]

#v(1em)
=== Neuromorphic Hardware Techniques

#serif-text()[ In contrast to deep learing hardware. neuromorphic hardware is not constrained to the von neuman architechture and thus the bottleneck
adress event representation make reference to macro-motif section ]

#figure( include("figures/inmemory.typ"), caption: [In-Memory Computing via a Crossbar Array. Unlike von Neumann architectures, memory and computation are physically co-located. Input voltages (V, representing activations) are applied to the wordlines. The memory elements at the junctions hold programmable conductances (G, representing synaptic weights). Multiplication is naturally performed at each junction by Ohm's Law (I=V×G), and the resulting currents are summed along the bitlines via Kirchhoff's Current Law (∑I). This allows dense matrix-vector multiplications to occur in a single analog time step with zero data transport cost.])

#serif-text()[ crossbar array looks like a neural network! ]

#figure(include("figures/inmemoryhierarcy.typ"), caption:[Architectural Comparison. (Left) The Von Neumann architecture separates memory and compute, creating a bottleneck. (Right) The Neuromorphic architecture co-locates them, mimicking the distributed topology of biological neural networks.])
#pagebreak()
== Neuromorphic State Of The Art

#serif-text()[ Disentangling core computational mechanisms from biological implementation details is a major ongoing challenge in neuroscience and neuromorphic engineering. Some complex molecular processes might be essential for learning or adaptation, while others might primarily serve metabolic or structural roles not directly involved in the instantaneous computation being modeled. The principles of neuromorphic computing, born from Carver Mead's vision and informed by modern neuroscience, have matured from theoretical concepts into a vibrant field of applied research. This progress is best seen in two key areas: the development of specialized, brain-inspired hardware and the creation of sophisticated software frameworks for simulating and deploying spiking neural networks (SNNs). ]


#v(1em)
=== Applied Neural Codes <appliedneuralcodes>

#serif-text()[ A central challenge in neuromorphic engineering is encoding information into spikes. The most "standard" method is Rate Coding (frequency of spikes = intensity), but this is slow and energy-inefficient. To solve this, Simon Thorpe proposed Rank Order Coding. Thorpe observed that the human visual system processes images far too quickly for neurons to average spikes over time. Instead, he proposed that information is encoded in the order of firing. The most strongly activated neurons fire first.]

#box-text()[ The "N-of-M" Strategy: To implement this efficiently in hardware, engineers often use an "N-of-M" coding scheme.
- M (Population): A large pool of potential neurons (e.g., 1000).
- N (Active): Only the first N neurons (e.g., 50) to spike are transmitted.
- Mechanism: Once N spikes are received, the system inhibits the rest. This guarantees extreme sparsity (low power) and filters out "noise" (late spikes). ]

#serif-text()[ This approach transforms time into a priority queue. A downstream neuron does not wait for a "frame" to finish; it begins computing as soon as the earliest, most salient spikes arrive. ]

#v(1em)
=== Learing

#serif-text()[ The most biologically plausible learning algorithm is @stdp. Unlike Deep Learning, which updates weights based on a global error calculated at the output, @stdp updates weights based on local causality between two connected neurons.

Causal (LTP): If the input spike (pre) arrives before the output spike (post), the synapse is strengthened (Long-Term Potentiation). The input "caused" the firing.

Acausal (LTD): If the input spike arrives after the output spike, the synapse is weakened (Long-Term Depression). The input was irrelevant to the decision.

@stdp allows networks to self-organize and detect repeating patterns in data without labeled supervision. However, on its own, it struggles to reach the high accuracy of modern supervised classifiers. ]

#v(1em)
#mini-header()[Surrogate Gradients]

#serif-text()[ To achieve "State-of-the-Art" (SOTA) performance on complex tasks (like ImageNet or Speech Recognition), modern neuromorphic engineers often hybridize biology with Deep Learning.

The core problem is that spikes are non-differentiable (a step function has zero derivative everywhere), which breaks standard Backpropagation. The solution is Surrogate Gradient Learning.

Forward Pass: The hardware uses the true, crisp spiking physics (non-differentiable).

Backward Pass: The learning algorithm substitutes the spike with a smooth "surrogate" function (like a sigmoid) to calculate gradients.

This allows us to train SNNs using powerful optimizers (like Adam) and frameworks (like PyTorch), transferring the trained weights to the neuromorphic chip for efficient inference. ]

#v(1em)
=== Complex Dynamical Models

#serif-text()[ Neuromorphic is not just about making processors but also about building machines and models to better understand the biology

Beyond these standard algorithms, there is a rich landscape of more complex theoretical models. While we will not utilize them in this specific implementation, they represent the frontier of the field:

Reservoir Computing (Liquid State Machines): Using a chaotic, randomly connected "tank" of neurons to project inputs into high-dimensional space before a simple readout layer.

Equilibrium Propagation: A physics-based learning rule that relaxes a network to an energy minimum, avoiding the need for a separate backward pass.

Attractor Networks: Recurrent networks (Ring or Line attractors) that maintain stable states (memories) even in the absence of input, crucial for spatial navigation and working memory. ]

#v(1em)
=== Neuromorphic Sensors

#serif-text()[ 2.1. The Paradigm Shift: From Frame-Based to Event-Based Sensing

Traditional sensory acquisition systems, particularly in computer vision and audio processing, have historically relied on the von Neumann architecture's separation of memory and processing, coupled with a clock-driven sampling approach. In this conventional paradigm, sensors capture the state of the entire environment at fixed temporal intervals—generating frames or samples regardless of the scene's activity. While effective for static analysis, this method generates redundant data streams that impose significant latency, bandwidth, and power consumption penalties, particularly in high-speed or sparse-signal environments.

In contrast, neuromorphic engineering, a field pioneered by Carver Mead in the late 1980s, seeks to emulate the biological principles of the mammalian nervous system. Neuromorphic sensors abandon the global shutter and fixed clock in favor of asynchronous, event-driven acquisition. Information is encoded not as absolute values at a fixed rate, but as a sparse stream of "events" or "spikes" triggered only when a significant change in the physical stimulus occurs. This bio-inspired approach offers theoretical and practical advantages in terms of temporal resolution, dynamic range, and energy efficiency (pJ/event), effectively shifting the computational burden from raw data processing to sparse event management.

2.2. Neuromorphic Vision Sensors (DVS)

The most mature realization of this paradigm is the Dynamic Vision Sensor (DVS), often referred to as the event camera. Unlike standard Active Pixel Sensors (APS) that integrate photon counts over a fixed exposure time, each pixel in a DVS operates independently and asynchronously.

The fundamental operation of a DVS pixel involves continuous monitoring of the log-intensity of the incident light. An event e is generated at time t when the change in logarithmic intensity exceeds a preset threshold θ:

Δln(I)=ln(I(t))−ln(I(t−Δt))≥±θ

where I(t) is the photocurrent at time t. This mechanism yields a stream of events characterized by a tuple (x,y,t,p), representing the pixel coordinates, the microsecond-resolution timestamp, and the polarity of the brightness change (ON or OFF).

Recent literature highlights three key advantages of this architecture:

    Temporal Resolution: Event cameras achieve effective frame rates equivalent to several kilohertz, with latencies in the microsecond range, making them ideal for high-speed robotics and ballistics tracking.

    Dynamic Range (DR): Because individual pixels do not share a global exposure time, DVS pixels do not saturate easily. Modern neuromorphic sensors boast dynamic ranges exceeding 120 dB, compared to the ∼60 dB typical of standard industrial cameras, allowing robust operation in scenes with simultaneous extreme brightness and darkness.

    Data Sparsity: In static scenes, a DVS generates near-zero output, drastically reducing power consumption and downstream processing requirements compared to frame-based cameras that output constant data regardless of content.

2.3. Auditory and Tactile Modalities

While vision sensors dominate the field, the principles of neuromorphic sensing extend to other modalities, mirroring the specialized transduction mechanisms of the biological cochlea and skin.

2.3.1. Silicon Cochlea

Neuromorphic auditory sensors, or "silicon cochleas," emulate the basilar membrane's hydrodynamics. Instead of performing a global Fourier Transform (FFT) on a sampled audio waveform, these sensors utilize a cascade of analog band-pass filters. Each filter channel operates independently, generating spikes when the energy in its specific frequency band exceeds a threshold. This architecture provides high-fidelity temporal information crucial for tasks such as sound source localization and separation, often achieving lower latency and power consumption than DSP-based solutions.
2.3.2. Neuromorphic Tactile Sensing

Electronic skins (e-skins) and neuromorphic tactile sensors are an emerging frontier, designed to provide robots with high-frequency feedback for manipulation tasks. Recent advances utilize triboelectric, piezoelectric, or piezoresistive materials coupled with event-based readout circuits. These sensors encode pressure, vibration, and shear force changes as asynchronous spikes, allowing for the rapid detection of slip events—a critical capability for stable grasping that mimics the fast-adapting mechanoreceptors (e.g., Meissner corpuscles) in human skin.

2.4. Processing Events: Spiking Neural Networks (SNNs)

The asynchronous nature of neuromorphic sensors necessitates a departure from standard Convolutional Neural Networks (CNNs), which are optimized for dense, synchronous matrix multiplications. The natural processing counterpart for event streams is the Spiking Neural Network (SNN).

In an SNN, artificial neurons maintain an internal membrane potential that integrates incoming spikes over time; the neuron "fires" an output spike only when this potential crosses a threshold. This compatibility has driven the development of specialized neuromorphic hardware accelerators, such as Intel’s Loihi and SynSense’s Dynap-CNN, which support massive parallelism and local synaptic plasticity. These chips enable edge-native learning and inference with power budgets in the milliwatt range, significantly lower than standard GPU-accelerated embedded systems ]

#v(1em)
=== Neuromorphic Computers

#serif-text()[ The primary goal of neuromorphic hardware is to escape the von Neumann bottleneck and emulate the power efficiency and massive parallelism of the brain. Two landmark systems define the state of the art:

IBM TrueNorth: A prominent early example, TrueNorth is a fully digital, real-time, event-driven chip. It consists of 4,096 "neurosynaptic cores," collectively housing one million digital neurons and 256 million synapses. Its architecture is explicitly non-von Neumann; processing and memory are tightly integrated within each core. TrueNorth's key achievement is its staggering power efficiency: it can perform complex SNN inference tasks (like real-time video object detection) while consuming only tens of milliwatts---orders of magnitude less than a CPU or GPU performing a similar task. However, its architecture is largely fixed, making it a powerful "inference engine" but less flexible for researching novel, on-chip learning rules.

Intel Loihi (and Loihi 2): Intel's line of neuromorphic research chips, starting with Loihi in 2017, represents a significant step towards flexible, on-chip learning. Like TrueNorth, Loihi is an asynchronous, event-driven digital chip, but with a key difference: it features programmable "learning engines" within each of its 128 neuromorphic cores. This allows researchers to implement and test dynamic learning rules, such as @stdp and its variants, directly on the hardware in real-time. The second generation, Loihi 2, further refines this with greater scalability, improved performance, and more advanced, programmable neuron models, positioning it as a leading platform for cutting-edge neuromorphic algorithm research.

Neuromorphic sensors ...] 


#v(1em)
=== Simulation and Software Frameworks <simulationandsoftwareframeworks>

#serif-text()[ Before algorithms can be deployed on specialized hardware, they must be designed, tested, and validated. This is the role of SNN simulators, which function as the "TensorFlow" or "PyTorch" of the neuromorphic world.

Brian: A highly flexible and popular SNN simulator used extensively in the computational neuroscience community. Its strength lies in its intuitive syntax, which allows researchers to define neuron models and network rules directly as a set of mathematical equations (e.g., the differential equations of a Leaky Integrate-and-Fire neuron). This makes it an ideal tool for exploring the detailed dynamics of biologically realistic models.

Nengo: A powerful, high-level framework that functions as a "neural compiler." Nengo is built on a strong theoretical foundation (the Neural Engineering Framework) that allows users to define complex computations and dynamical systems in high-level Python code. Nengo then "compiles" this functional description into an equivalent SNN. Its key advantage is its backend-agnostic nature; the same Nengo-defined network can be run on a standard CPU, a GPU, or deployed directly to neuromorphic hardware like Intel's Loihi. ]

#pagebreak()

== Research Gaps

#serif-text()[ Despite this immense progress in hardware and software, a fundamental challenge remains, creating a critical research gap: the training problem.

Mainstream deep learning has a powerful, universal tool: backpropagation. Neuromorphic computing does not yet have a clear equivalent. While these systems exist, they still struggle with finding an efficient, scalable, and powerful learning algorithm that is both biologically plausible and computationally effective. This gap manifests in several ways:

- Limited Supervised Learning
"Local" rules like @stdp are fundamentally unsupervised. They are excellent for finding patterns and correlations but struggle with complex, task-driven "supervised" problems (e.g., "classify this audio signal into one of ten specific words").

- The Conversion Compromise
A popular workaround is to first train a conventional, non-spiking @ann using backpropagation, and then "convert" its weights to an @snn for efficient inference. This method, while practical, is a compromise. It discards the rich temporal dynamics SNNs are capable of and does not represent true "neuromorphic learning."

- The Surrogate Gradient Challenge
The firing of a spiking neuron is a non-differentiable event, which makes it incompatible with standard backpropagation. New methods, like "surrogate gradient" learning, attempt to approximate this spike event with a smooth function to enable gradient-based learning, but this is an area of intense and ongoing research.

This thesis confronts this central challenge: How to effectively and efficiently train spiking neural networks for complex, real-world temporal tasks. While hardware like Loihi provides the platform for on-chip learning, it still requires a robust and scalable algorithm. The existing approaches of ANN-to-SNN conversion or simple Hebbian rules are insufficient.

This thesis proposes a method to bridge this gap by... (...for example: "...developing a novel, event-driven surrogate gradient algorithm capable of training deep @snn:pl directly in the temporal domain," or "...introducing a hybrid learning rule that combines the efficiency of @stdp with the task-driven power of error-based feedback," or "...proposing a new architecture for temporal credit assignment that is both hardware-friendly and scalable.") ]

#pagebreak()

= Method <method>

#serif-text()[ This chapter details implementations of neuromorphic methods to address the fundamental limitations of standard deep learning described in @biologicalprinciples. The approach presented here aligns with the constraints of biological substrates: sparsity, asynchrony, and locality. We propose a neuromorphic architecture designed to maximize energy efficiency and computational robustness ]

#v(2em)
== Data

#serif-text()[]

#figure( include("figures/dataexample.typ"), caption: [In-memory])
#figure( include("figures/dataexamplenoise.typ"), caption: [In-memory])


#pagebreak()

#v(2em)
== Information Representation

#serif-text()[ The choise of a coding shceme puts key constraints on the design any processing system as it lays the founation for the flow of information. In @neuralcoding and @appliedneuralcodes we dicussed several candidate neural codes that are both biologically plausible and have been used in previous neuromorphic systems. The neural code plays a large role in determening the effecieny of the system. Many neuromorphic systems use rate code as it is easy to translate values it is straight forward. any float value can be encoded as a rate code and integrate and fire neurons work well with this encoding. An IF neuron using rate code can reduce the multiply and add operations with just add operations. each spike arriving at a synapse adds its weight to the total. It is easy to see that for a rate code a higher rate input means more of that input contributing to the total sum. Using a rate code we can convert multiply and accumulate operations to just a series of accumulate operations by discretetizing the input values. The main reason for why this thesis will not use a rate code is that it is relativly ineficient and slow compared to a temporal code. In a temporal code only one spike is required to establish its value in relation to other synapses, making a single event much more dense in information. This is especially important for systems that do not have the connection density of a biological brain and must use shared buses that can be congested if too many spikes are present on the bus. Secondly to determine the value of a rate code you have to take an average of multiple spikes, imposing a delay. Using a @ttfs encoding requires more sophisticated neuron models than a simple integrate and fire, it need more bookkeeping to keep track of the relative arival of incoming spikes and as mentioned in @neuralcoding and as we will explain by using a temporal encoding we need to handle the phase abmiguity problem.
A neural code should have the following characteristics: ]

#box-text()[
  - Fast
  - Effecient in terms of neurons used
  - Able to encode a wide range of stimuli
  - Robust to noise
]

#serif-text()[ Visual tasks which is what the application this thesis focuses on ]
#box-text()[
  1. a population code is used in conjunction with a temporal code
  2. Population code is used alone (eg individual order within a population does not matter)
]

#serif-text()[ That beeing said and as discussed in @neuralcoding combining neural codes and using the right one for the job is also an important aspect to consider, there is not a clear consenus on which code is better it is a matter of the task at hand. As we also have talked about, Combining differnt codes toghether can make for a powerful representation of information. This thesis focuses on visaul recognition and a TTFS combined with population code is the representation of choise. The popultion code will represent a distinct pattern in an image. This can be set up to detect lines and primitve shapes, deeper layers will use populations of primitve shapes to detect more complex features like circles and boxes and so on. ]

#v(1em)
=== Encoding <encoding>

#serif-text()[ To convert a float value (like a pixel value from an image to a @ttfs encoding we first need to find the range of the pixels from minimum to maximum value. Then we need to decide on what featere of interest we wish to represent, a natural choise is pixel luminance but others such as contrast or hue or other features using other color spaces could be used. For our purposes we will not be using contrast but since our image is engineered for this task we can skip the contrast extraction that needs to be in place for real world images and poduce images that has clear distinct lines present in the raw pixel data (since the dimensions are small a line in the image will typically only be a few pixels wide so there would be no difference if we took the contrast of pixels or the luminance direclty).. If the image is black and white it is straight forward to extract the luminance as most often the pixel luminace is stored directly. Once we have ound the range say minimum is 0 and max is 255 we can construct an event queue where the pixel intensity determines its place in the queue and its assoiated delay. The conversion mapping can be linar where each pixel gets its luminace multiplied by a scalar other usefull mappings can be logarithmic or exponential. ]

#figure(
kind:"algo",
caption: [Linear intensity to dely encoding],
supplement: [Algorithm],
mono-text(pseudocode-list(hooks:.5em, indentation:1em, booktabs:true)[
procedure intensity_to_delay_encoding(image, T_max=100, T_min=0): #h(1fr)
  + normalized_image = normalize(image)
  + spike_times = T_max - (T_max - T_min) \* normalized_image
  + return spike_times
]))

#figure(
kind:"algo",
caption: [Logarithmic intensity to dely encoding],
supplement: [Algorithm],
mono-text(pseudocode-list(hooks:.5em, indentation:1em, booktabs:true)[
procedure intensity_to_delay_encoding(image, T_max=100, T_min=0): #h(1fr)
  + normalized_image = normalize(image)
  + spike_times = T_max - (T_max - T_min) \* normalized_image
  + return spike_times
]))

#serif-text()[ In deeper layers the meaning of the information may be obscured as is common with deep neural networks, however temporal encoding should if the neuron model is designed with this sceme in mind should produce shorter delays for greater values if a deeper layer neuron reacts to a box and its neigbour in the same layer reacts to a circle and the image is a rounded box (but more box than circle) then the neuron responind to a box should emit a spike faster than the neuron responding to a circle. ]

#v(1em)
=== Decoding

#serif-text()[ Decoding the neural code requres an adequate neuron model that uses the neural code to run useful computations. As discussed in @biologicalprinciples a neuron should have the following charecteristcs: ]

#box-text()[
  - Acuumulate spikes in some form (integrate)
  - Fire when some treshold has been reached
  - Leak the potential over time
]


#serif-text()[ We have seen that glif neuron is bio plausible and such a neuron is simple and can be realized easily on hardware the way. Using a temporal code the neuron has to differentiate between inputs in the time domain otherwise it cannot decode the information and cannot do useful work. Earlier inputs are encoded with larger values and should have a greater portion of summation going on inside the neuron. The neurons need to be desigend to work with the chosen encoding either at individual neuron level or population level

-If order matters (temporal code) the neuron must handle it
Evidence for bio-plauibility can be found from eq 1 where the R(u) is dependent on the membrane potential 

make the input exponentially weaker in proportion to the threshold. If the treshold is zero then the incoming spike does not get suppressed if the therhold is very close to max then the incoming spike gets more suppressed. ] 


#figure( include("figures/thresholdsensitive.typ"), caption: [Neuron model where new incomming spikes have an exponential decaying influence on the potential])

#serif-text()[ A counter starts when the first spike arrives

Needs a global reset or local like we talked about in the biological section about phase ambiguity ]

#figure(
kind:"algo",
caption: [Logarithmic intensity to dely encoding],
supplement: [Algorithm],
mono-text(pseudocode-list(hooks:.5em, indentation:1em, booktabs:true)[
integrate_and_fire(excitatory, inhibitory, threshold) -> integer:
  + excitatory_spikes = [] #h(1fr)
  + if inhibitory is None:
    + inhibitory_spikes = []
  + else:
    + inhibitory_spikes = []
  + all_spikes = excitatory_spikes + inhibitory_spikes
  + if not all_spikes:
    + return None
  + all_spikes.sort(key=lambda x: x[0])
  + integrated_potential = 0.0
  + firing_time = None
  + for time, spike_type in all_spikes:
    + integrated_potential += spike_type
    + integrated_potential = max(0, integrated_potential)
    + if integrated_potential >= threshold:
      + firing_time = time
      + break
  + return firing_time
]))


#serif-text()[ In a time to first spike scheme of we care about the order (the relative values since information is stored in time and order) we have to use weights and a neuron model that distinguish between inputs arriving earlier than others. I present a scheme where the first neuron that arrives starts a linear count where the slope of the counter is the weight additional inputs will increase or decrease the slope according to their weight. We can see that neurons arriving earlier will get more time to increase the counter and thus will carry a higher value. If the counter reaches a threshold the neuron will fire. The astute will notice that in this scheme the neuron will fire even for the smallest stimulus since the counter will count up a non zero value and eventually reach the threshold, to mitigate this we can simply say that if the counter is too slow the neuron will not fire we will see later that this scheme satisfies the criteria above.

The problem with this decoding is for strong stimuli we would ideally make the neuron respond immediately and fire, but it has to wait until the counter has reached the threshold to fix this we can also add the weight of the input directly to the potential while also starting a counter. Now if early strong inputs arrive they will fill up the potential and make the neuron fire almost immediately. Small inputs wil take some time  ]

#figure( include("figures/neuronmodel.typ"), caption: [Proposed simplifed layout of a SNN. The neurons are connected with hirearcical busses that allow for the network to be configured as a _small world network_] )


recall this equation. @ttfs model should mathematically behave the same
$ I_i(t) = sum_j w_(i j) dot S_j(t) $

#serif-text()[ Leaky integrate and fire models seem the best bet, however complex dynamics like exponential decay and analog weights and potentials seem excessive, we might do without. Binary weights 1 for excitatory and and 0 for inhibitory. Stronger weights can be modeled with multiple parallel synapses

Another way which is also based on relative firing order of single spikes could be a passcode encoding. Such an encoding could work by having neurons only react to a sequence. It has an internal state machine of sorts and will only advance to the next state if recives the correct input in the correct order. This encoding does only care about relative order not relative timings. ]

#pagebreak()

== Network Architechture

#serif-text()[ As the methods will be fitted to visual stimuli. Network topologies for these tasks are widely studdied and proved to be effective the topolygy we will be using is similar to @cnn topology whiich is also inspired by the visual cortex in mamals. The idea is that early layers will pick up on very simple features like lines and curves that form eg around an object. The next layer might use the representation of lines and curves from the layer before to represent more complex shaped like boxes and circles. Furhter down the network more complex and abstract features can be represented this way. We will focus on fairly shallow networks with three layers to capture the simple shapes like boxes and circles present in the data. Both the @cnn and the @snn will have the same architechture and number of parameters so that weights can be shared and the idea is that comparisons should be more fair. ] 

#figure( include("figures/architechture.typ"), caption: [Network arhcitechture for the task at hand])

#serif-text()[ Using the same architechtire for @cnn and @snn and sharing weights is not mathemathically eqiuvalent but sharing weights is still usefull. The key distinction is that the @snn uses lateral inhibition the cnn does nececarly do, altough max pooling works in the same way.

The input layers match the input dimension

use 3x3 filters since features are not that big

use 3 channels, each channel is used for a spesicic local feature, use wta/max pooling to choose the most prominent feature in a local region (lateral inhibition. A local feature like vertical diagonal or horisontal line is "chosen" in that region) ]

#pagebreak()

== Proposed Learning Algorithms

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

#pagebreak()

== Simulation

#serif-text()[ We have discussed the benefits of co algorithm design and designing new specialed computer hardware that run neuromorphic algorithms directly on the substrate via gates or analaog elements and exotic new materials. However developing biologically inspired computers and algorithms on cheap and avaliable CPU and GPU hardware is a great way to quicly iterate and test

Altough many simulation and software packages exists as outlines in @simulationandsoftwareframeworks. The methods in this thesis has been tested using custom simulaton software for full control ]


#figure(include("figures/simulatorarch.typ"),caption:[Simulator architechture block diagram])

#serif-text()[ The simulator runs an event loop Spikes are pushed to a heap The simulator can run both on CPU and GPU, When running on CPU the spikes are pushed to a heap, when running on GPU spikes are pushed to a adress event bus the algorithms presented in this thesis are highly paralizable It is built from the ground up using the Vulkan API ]

#figure( include("figures/eventloop.typ"), caption: [In-memory])

// #figure(image("figures/screenshot.png"), caption: [Neuron model where new incomming spikes have an exponential decaying influence on the potential])

#pagebreak()

== Metrics

#serif-text()[ To test and verify the methods we need a way to measure the performance for classification tasks accuracy recall and . is often used. This works great for supervised learning. For unsupervised learing ... ]

// #figure(include("figures/confusionmatrix.typ"),caption:[Simulator architechture block diagram])

#serif-text()[ Accuracy and recall measures effectivness but as the goal of this thesis is to improve efficiency we need a way to mesaure the resource usage. There are direct ways to do this like measure power draw and data usage however we do not have the resources to set up such a test rig. Another way is the measure and theroize about number of operations as an indirect measure of resoource usage. This has several accuracy issues but can give an estimate. The first issue is that this is not an apples to apples comparison ]

#pagebreak()

= Results <results>


== Inference
#serif-text()[
#lorem(100)

#figure(include("figures/network.typ"),caption:[Neural network before learing])

#figure(include("figures/network.typ"),caption:[Neural network during learning])

#figure(include("figures/network.typ"),caption:[Neural network after learning])

#lorem(100)

#lorem(100)
]

== Training

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

#serif-text()[
In this section we discuss the

In the engineered test enviroment we used pixel intensity directly and for this toy example it works quite well since the feeatures we are interesting in are eesy to read in a low resoluton simple enviroment. However for real world vision tasks pixel intensity is not the best features to feed into the algorithm, it is better to use local contrast since that is what determines object outlines and other interesting elements of an image. It is possible to use an SNN to compute local contrast from pixel intsity, in fact that was part of the development of the algorithm. For a @ttfs aproach it gets very difficult to do this effeciently since to get the normalized contrast (true contrast indepenendt of luminance) you have to wait for all signals to arrive even the complete absense of light completly wasting the potential of @ttfs. To fix this a dedicated chip that outputs @ttfs contrast directly can be used. A normal difference of lumince gives you a absolute measure of contrast where differences of brighter pixels gives a larger contrast where differences between two darker pixels gives a lower contrast. Contrast should be relative (a ratio) so by converting luminace to log-space first and then taking difference will give a true contrast independent of luminance intensity

We see a trade-off between the ability to learn and speed of learing and forgetting. Synaptic plasticitiy must be tuned in order for the right learing enviroment to form

representing contrast is difficult with a ttfs scheme, better to use special sensors and pass the encoded contrast in ttfs to a neuromophic processor

#v(1em)
=== Perceptron equivalence

#serif-text()[ The perceptron equation can be obtained with a ttfs using the inverse of firing times ]

#figure( kind: "eq", supplement: [Equation], caption: [Weigthed sum], [
$ T = sum w/t $
])

#serif-text()[ In a time to first spike scheme of we care about the order (the relative values since information is stored in time and order) we have to use weights and a neuron model that distinguish between inputs arriving earlier than others. I present a scheme where the first neuron that arrives starts a linear count where the slope of the counter is the weight additional inputs will increase or decrease the slope according to their weight. We can see that neurons arriving earlier will get more time to increase the counter and thus will carry a higher value. If the counter reaches a threshold the neuron will fire. The astute will notice that in this scheme the neuron will fire even for the smallest stimulus since the counter will count up a non zero value and eventually reach the threshold, to mitigate this we can simply say that if the counter is too slow the neuron will not fire we will see later that this scheme satisfies the criteria above.

The problem with this decoding is for strong stimuli we would ideally make the neuron respond immediately and fire, but it has to wait until the counter has reached the threshold to fix this we can also add the weight of the input directly to the potential while also starting a counter. Now if early strong inputs arrive they will fill up the potential and make the neuron fire almost immediately. Small inputs wil take some time  ]

recall this equation. @ttfs model should mathematically behave the same
$ I_i(t) = sum_j w_(i j) dot S_j(t) $

#serif-text()[ Leaky integrate and fire models seem the best bet, however complex dynamics like exponential decay and analog weights and potentials seem excessive, we might do without. Binary weights 1 for excitatory and and 0 for inhibitory. Stronger weights can be modeled with multiple parallel synapses

Another way which is also based on relative firing order of single spikes could be a passcode encoding. Such an encoding could work by having neurons only react to a sequence. It has an internal state machine of sorts and will only advance to the next state if recives the correct input in the correct order. This encoding does only care about relative order not relative timings. ]

easy with rate code

need global information to normalize

using ttf is difficult

using negative image - still problem with absolute contrast

can be done locally with speial sensors and using logarithmic intensity to delay ]

#lorem(150)

#lorem(150)

#lorem(100)
#lorem(100)

#lorem(100)

#pagebreak()
== Future Work

#lorem(150)

#lorem(150)
#lorem(100)


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
