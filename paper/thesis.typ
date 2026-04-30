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

// #text(size: 9pt, weight: "medium")[ #h(1fr) Wordcount: #total-words ]

#pagebreak()

#align(center)[
#block(width:100%)[
#align(left)[
#text(weight:"semibold",size:16pt,[ACKNOWLEDGEMENTS & DECLARATIONS])

#serif-text()[ I would like to thank my supervisors and the very kind and helpful community at the ROBIN and NANO research groups at the Department of Informatics.


The repository containing all source code including simulation software, source code for this document and its figures can be found at https://github.com/nammenam/neuromorphics.git ]

#v(1em)
#mini-header()[Declaration of the use of generative artificial intelligence]

#serif-text()[ In this scientific work, generative @ai has been used. All data and personal information have been processed in accordance with the University of Oslo's regulations, and I, as the author of the document, take full responsibility for its content, claims, and references. An overview of the use of generative @ai is provided below.

The service Gemini, developed by Google @team_gemini_2025, has been used to improve the content of the thesis. Sections of text like a subsection, paragraph or source code for figures was given to the model along with prompts to make the language free of errors and more professional. The text underwent multiple iterations using refined prompts to ensure structural coherence and academic tone. In the case for figures the model was used to speed up the development of the figures with the model generating numbers for positioning elements. The final result was cut out, fact-checked, and partly rewritten by the author.
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

In recent years, great strides have been made towards this goal. Deep Learning, which utilizes multilayered neural networks, has exceeded previous performance benchmarks. These systems have demonstrated high proficiency in tasks previously limited to human capability. For example, AlphaFold has addressed complex problems in protein folding @jumper_highly_2021, reinforcement learning agents have mastered the complexity of games such as Go @silver_mastering_2016, and Large Language Models have shown capabilities in text generation that approach human fluency. Consequently, @ai is increasingly viewed as a general-purpose technology that may influence societal infrastructure.

However, despite these advances, there are significant limitations to the current approach. The success of modern deep learning relies heavily on scaling, which involves increasing data volume and computational power. This strategy is approaching physical and economic boundaries. Training state-of-the-art models consumes substantial energy and results in a large carbon footprint @strubell_energy_2019. Although specialized hardware allows for more efficient computations, the underlying architecture and algorithms impose an intrinsic limit on scalability independent of the underlying hardware. Furthermore, the requirement for massive datasets presents challenges in sourcing and curation. Additionally, evidence suggests that this scaling approach yields diminishing returns. Models often function as statistical correlation engines; they lack common-sense reasoning, struggle with out-of-distribution generalization, and are prone to brittle failure modes @geirhos_shortcut_2020.

These limitations are evident when comparing artificial systems to biological intelligence. The human brain demonstrates that high-level intelligence is possible without massive energy consumption or dataset sizes. The brain operates on approximately 20 watts @laughlin_energy_2001. With this limited energy budget, it manages biological functions, processes real-time multi-sensory data, and performs abstract reasoning. In contrast, deep learning models require GPU clusters with significantly higher power requirements to match a fraction of these capabilities. There is also a discrepancy in learning efficiency. Deep learning models are sample-inefficient, often requiring vast numbers of examples to learn a representation. Biological systems, however, are capable of "one-shot" or "few-shot" learning and can acquire new information without catastrophic forgetting. This suggests the inefficiency of current @ai is a paradigmatic issue rather than just an engineering problem.

The proposed direction for addressing these issues involves biological inspiration in both hardware and algorithm design, specifically Neuromorphic Computing. This field attempts to engineer computer architectures that mimic the biological structure of the nervous system. Unlike traditional @ai, which runs as software on general-purpose hardware, neuromorphic engineering aims to align the algorithm with the physical substrate. It moves away from clock-driven processing toward asynchronous, event-driven systems. In this paradigm, information is encoded as sparse, discrete events or "@spike:pl". Similar to biological neurons, a neuromorphic processor consumes minimal energy when inactive, processing information only when triggered. This approach is being pursued by both industrial groups, such as Intel’s Loihi @davies_loihi_2018, and academic projects like SpiNNaker @furber_spinnaker_2014. These systems represent a shift from calculation-based machines to those capable of real-time adaptation.

Although neuromorphic systems achieve optimal performance on co-designed platforms---where the algorithm is embedded directly into the hardware---there is significant value in executing neuromorphic algorithms on traditional von Neumann architectures. In this thesis, we explore biologically inspired algorithms deployed on traditional CPU and GPU hardware. We examine how event-driven, biologically plausible computation can address limitations in scalability, data efficiency, and energy consumption, even when simulated on standard processors. We present approaches for efficient information coding and learning algorithms inspired by neural mechanisms. Concretely, this thesis aims to: ]

#box-text()[
*Sparse Efficient Computing:* Investigate whether biologically inspired sparse,
event-driven algorithms can reduce the computational footprint of visual
classification when simulated on standard hardware.

*Neuron Model Evaluation:* Identify which temporal integration dynamics are
compatible with @ttfs rank-order decoding across systematic threshold regimes.

*Inference Via Weight Transfer:* Quantify the accuracy penalty of zero-shot
@ann\-to-@snn weight transfer under @ttfs encoding, isolating the cost of
transitioning from static activations to event-driven spike integration.

*Native Unsupervised Learning:* Determine whether local @stdp can extract
meaningful geometric features from visual input without labeled data or global
error signals.
]

#serif-text()[ The succeeding chapter lays the historical and theoretical foundation, covering early neuroscience and the development of artificial neural networks based on simple models of the brain. Following this, we review relevant modern neuroscience literature, extracting key concepts that will inform the methodology. We also provide concise overview on machine learning concepts and frameworks. Finally, we detail the implementation of these principles in a neuromorphic context and evaluate their performance against standard benchmarks. ]

#pagebreak()

= History & Developments <s.history>

#serif-text()[ Historically, the understanding of neural tissue was dominated by the reticular theory, which claimed that the brain consisted of a continuous, fused network of nerve fibers. This paradigm was fundamentally challenged by the work of Santiago Ramón y Cajal. Through the application of novel staining techniques, Cajal established the neuron doctrine, demonstrating that the nervous system is composed of discrete, individual cells @glickstein_golgi_2006. Building on these findings, Heinrich Wilhelm Gottfried Von Waldeyer-Hartz proposed the "Neuron Doctrine" and coined the term "neurons" to describe these dicrete cells @waldeyer_ueber_1891. Subsequent analysis using electron microscopy has provided irrefutable validation of this discrete cellular structure.

The conceptualization of the brain as a collection of discrete units facilitated the development of mathematical models describing neural function. In 1943, Warren McCulloch and Walter Pitts published A Logical Calculus of the Ideas Immanent in Nervous Activity, introducing the first formal model of the neuron.

The McCulloch-Pitts (M-P) neuron abstracted biological complexity into a binary decision device governed by the following logic: ]

#box-text()[
*Inputs*: The neuron receives multiple binary inputs, weighted as either excitatory or inhibitory.

*Summation*: The unit calculates the linear sum of these weighted inputs.

*Thresholding*: If the aggregate sum exceeds a fixed threshold, the neuron outputs a 1 (firing); otherwise, it outputs a 0 (silence).
]

#serif-text()[ McCulloch and Pitts demonstrated that networks of these units could theoretically compute any logical operation (AND, OR, NOT) @mcculloch_logical_1943. This abstraction established the foundational link between biological processes and digital logic, suggesting that neural function could be replicated in electronic hardware. Consequently, the M-P neuron serves as the common ancestor for both computational neuroscience and artificial intelligence.

However, despite its theoretical utility, the original M-P model presented significant functional limitations. The connectivity was static, requiring circuits to be manually designed rather than learned. Furthermore, the restriction to binary weights precluded the modeling of graded signal intensity, preventing the system from capturing the nuance of real-world sensory input.

In 1949, Donald Hebb addressed the critical issue of plasticity in his work The Organization of Behavior. He proposed a theoretical mechanism for synaptic modification, now known as Hebbian learning, which provided a biological basis for how neural networks could adapt over time. Hebb postulated: ]

#box-text()[ "Let us assume that the persistence or repetition of a reverberatory activity (or "trace") tends to induce lasting cellular changes that add to its stability. ... When an axon of cell A is near enough to excite a cell B and repeatedly or persistently takes part in firing it, some growth process or metabolic change takes place in one or both cells such that A’s efficiency, as one of the cells firing B, is increased" @hebb_organization_2002. ]

#serif-text()[ This principle is colloquially summarized as "neurons that fire together, wire together" @shatz_developing_1992. Crucially, this describes a local and decentralized learning rule; a synapse does not require a global error signal or external supervision to adjust. It requires only the correlation between the pre-synaptic input and the post-synaptic output. The convergence of the M-P architectural model and the Hebbian plasticity framework established the prerequisite conditions for the development of modern neural networks. ]

#v(2em)
== The Perceptron <s.perceptron>

#serif-text()[ In 1957, Frank Rosenblatt advanced these theoretical concepts by engineering the Perceptron @rosenblatt_perceptron_1958. The "Mark I Perceptron" was a hardware implementation of the neural model, distinguished by a crucial innovation: a weight-adjustment mechanism based on Hebbian principles. Rosenblatt introduced the perceptron learning rule, an iterative algorithm capable of minimizing error automatically. The system processed an input pattern (e.g., a pixelated character) and produced a binary classification. When the output deviated from the target, the algorithm adjusted the weights proportional to the error: strengthening connections that should have contributed to a correct firing and weakening those that led to false positives. ]

#figure(include("figures/perceptron.typ"),caption:[The perceptron model. Inputs $x_i$ are multiplied by weights $w_i$ and summed. If the linear combination $sum x_i w_i$ exceeds the bias $b$, the neuron activates. ])

#serif-text()[ Consequently, the Perceptron was capable of converging on a solution for any problem where the data was linearly separable. This success generated significant enthusiasm, with contemporary reports suggesting that such machines would soon mimic human consciousness @noauthor_new_1958.

These expectations were abruptly tempered by theoretical limitations. In 1969, Marvin Minsky and Seymour Papert published Perceptrons, a rigorous mathematical analysis of the architecture @minsky_perceptrons_1988. They demonstrated that a single-layer perceptron is fundamentally a linear classifier. While capable of learning operations like AND or OR, it is mathematically incapable of solving the XOR (Exclusive OR) problem. In the XOR case, the classes cannot be separated by a single hyperplane. This proof highlighted a severe boundary on the utility of single-layer networks for complex, non-linear tasks. ]

#figure(include("figures/gates.typ"),caption:[The XOR problem. Unlike AND/OR, the data points for XOR cannot be separated by a single linear boundary.])

#serif-text()[ The publication of Perceptrons coincided with a significant reduction in neural network research funding, a period retrospectively termed the "First AI Winter". It is worth noting that Minsky and Papert acknowledged that a @mlp, a network stacking multiple layers of neurons, could theoretically solve the XOR problem by creating complex, non-linear decision boundaries.

However, a critical algorithmic gap remained: the "credit assignment problem". While researchers knew that hidden layers could represent complex features, there was no known method to propagate error signals back through the layers to adjust the weights of hidden neurons effectively. Rosenblatt’s rule was mathematically valid only for the output layer. The field remained stagnant until a method for training multi-layer networks could be formalized. ]

#v(2em)
== Deep Learning <s.deeplearningintro>

#serif-text()[ The critique presented by Minsky and Papert precipitated a contraction in funding; despite this, theoretical inquiry persisted. It was widely hypothesized that the limitations of the single perceptron could be overcome by a @mlp. By organizing neurons (single perceptrons) into hierarchical layers, the network could theoretically perform successive non-linear transformations on the input space, enabling the formation of complex decision boundaries. The primary impediment was not the architecture itself, but the absence of a viable learning algorithm.

In a single-layer perceptron, error attribution is immediate: if the output deviates from the target, the error is directly derived from the weights of the output layer. However, in a multi-layer architecture, quantifying the contribution of a specific neuron within the "hidden" layers to the final output error presents a significant challenge. This is formally known as the Credit Assignment Problem @minsky_steps_1961, and it remained the central theoretical obstacle for over a decade. ]

#figure(include("figures/network.typ"),caption:[A @mlp. By inserting "hidden layers" between input and output, the network can approximate non-linear functions such as XOR. The historical challenge lay in deriving a method to train these intermediate layers.])

#serif-text()[ The solution to this theoretical impasse was popularized in 1986 by Rumelhart, Hinton, and Williams in their seminal paper _Learning representations by back-propagating errors_ @rumelhart_learning_1986. They demonstrated that the Chain Rule of calculus could be applied recursively to propagate the error signal from the output layer backwards through the hidden layers. This algorithm, known as Backpropagation, allowed the network to calculate the gradient of the loss function with respect to every weight in the system. Effectively, it provided a mathematical method to tell each hidden neuron exactly how much it contributed to the total error, finally solving the credit assignment problem.

Unlike Hebbian plasticity, which is local and biological, Backpropagation relies on global error signals and precise backward data flow---mechanisms effectively absent in organic tissue. Consequently, the field of @ann effectively decoupled from neuroscience. It transitioned into a branch of engineering and applied mathematics, prioritizing statistical optimization over biological realism. Paradoxically, it was this abandonment of biological fidelity that enabled the rapid scaling and performance breakthroughs that followed. ]

#v(1em)
=== Achievements

#serif-text()[ With the training mechanism solved, the field exploded. The combination of Backpropagation, massive datasets, and GPU hardware led to a led to a rapid diversification of neural architectures, each solving domains previously thought impossible for computers.

The revolution began in earnest with computer vision. @cnn:pl, such as AlexNet (2012) @krizhevsky_imagenet_2017 and later ResNet @he_deep_2016, introduced the idea of learning hierarchical features---detecting edges, then shapes, then objects---much like the human visual cortex. This allowed machines to classify images with superhuman accuracy.

Soon after, the focus shifted to sequence data. @rnn:pl and @lstm architectures gave machines a short-term memory, enabling breakthroughs in speech recognition and machine translation. However, the true paradigm shift occurred with the introduction of the Transformer architecture in 2017 @vaswani_attention_2023. By utilizing an "attention mechanism" to parallelize the processing of language, Transformers allowed for the training of massive @llm:pl like the @gpt.

These techniques have even transcended media generation. Deep Learning has solved fundamental scientific problems; notably, DeepMind's AlphaFold utilized these architectures to predict the 3D structure of proteins from their amino acid sequences, a 50-year-old grand challenge in biology @jumper_highly_2021 ]

#v(1em)
=== Shortcomings

#serif-text()[ Deep learning's reliance on computational scaling masks fundamental inefficiencies in both its hardware implementation and underlying algorithms. By simulating biological concepts on digital architectures not designed for them, the current paradigm is approaching physical and economic limits.

A primary limitation is the Von Neumann architecture, which physically separates processing units from memory. Deep neural networks, defined by massive matrices of synaptic weights, necessitate constant data transfer. For every inference step, billions of parameters must be fetched from off-chip DRAM, processed, and written back. This creates a severe memory bottleneck where system performance is bounded by bandwidth rather than processing speed @sze_efficient_2017.

Consequently, the energy cost of moving data significantly exceeds the cost of computation itself. Retrieving a single byte from DRAM consumes approximately three orders of magnitude more energy than performing a floating-point operation @horowitz_11_2014. Compounding this hardware friction, the dense matrix multiplications required for training scale quadratically with network size, making the pursuit of trillion-parameter models increasingly unsustainable.

Furthermore, the optimization algorithms driving this scale are fundamentally incompatible with physical biological systems. Backpropagation, while mathematically elegant, relies on a global error signal and suffers from the "weight transport problem"---the requirement that the backward pass utilizes the exact same synaptic weights as the forward pass. In organic tissue, synapses are unidirectional, and there is no known mechanism for a neuron to access the exact weight of a downstream synapse to calculate a gradient.

While a detailed technical analysis of these inefficiencies is presented in @s.technicaldetailsofml, the central issue is clear: modern AI prioritizes statistical optimization over physical realism. Overcoming the limitations of the Von Neumann bottleneck and backpropagation requires a paradigm shift toward architectures that inherently co-locate memory and computation. ]

#v(2em)
== Birth Of Neuromorphic <s.birthneuromorphic>

#serif-text()[ While the artificial intelligence community debated symbolic logic versus connectionism during the "AI Winter," significant developments were occurring in hardware physics. In the late 1980s at Caltech, physicist Carver Mead---a pioneer of @vlsi design---began to question the trajectory of digital computing.

Mead observed that while digital computers were becoming exponentially faster, they were also becoming less efficient in terms of energy per operation. He noted that using transistors as rigid, high-power switches to perform boolean logic was energetically wasteful compared to the biological systems they aimed to emulate.

In 1990, Mead published his seminal paper, _Neuromorphic Electronic Systems_ @mead_neuromorphic_1990, coining the term "neuromorphic" to describe hardware that mimics the biological structure of the nervous system. His thesis proposed that rather than simulating neural equations via software on digital computers, engineers should construct physical hardware that exploits the same physical laws as the biological nervous system.

The foundational insight of the field was the physical analogy between silicon physics and ion-channel physics. In standard digital electronics, transistors are operated in "strong inversion," driven by high voltages to act as binary switches. Mead realized that a single transistor, operating in its "subthreshold" region, follows the same exponential Boltzmann statistics that govern the flow of ions through biological channels.

This realization implied that a single transistor could physically compute the non-linear functions used by biological neurons, but with significantly higher speed and lower power consumption. Consequently, synaptic functions could be implemented by single transistors rather than complex arrangements of logic gates.

To demonstrate this concept, Mead and his doctoral student Misha Mahowald developed the _Silicon Retina_ in 1991 @mahowald_silicon_1991. Unlike a standard camera, which captures full frames at fixed intervals (generating redundant data), the Silicon Retina operated asynchronously. It utilized analog circuits to compute spatial and temporal derivatives directly on-chip, outputting discrete "events" only when local light intensity changed.

This event-driven approach solved the redundancy problem inherent in frame-based sampling. If the scene remained static, the system transmitted no data and consumed negligible energy. This demonstrated that by aligning the hardware physics with the computational task, sensory information could be processed with a fraction of the power required by conventional digital systems. ]

#v(3em)
#line(length:100%)
#v(3em)
#serif-text()[ Since the inception of neuromorphic computing, neuroscience has also advanced significantly. While Mead’s early work was based on the physical intuition of the transistor, modern neuromorphic engineering now incorporates a richer understanding of neuronal dynamics, synaptic plasticity, and network architecture. To advance the field, we must combine these foundational hardware insights with the principles of modern mechanistic neuroscience. ]

#pagebreak()

= Biological Principles <s.biological>

#serif-text()[ The biological brain remains the gold standard for energy-efficient, robust, and adaptive computation. Since the establishment of the Neuron Doctrine, modern neuroscience has uncovered the specific physical mechanisms that underpin this efficiency. To engineer systems that truly rival biological performance, we must transcend the highly simplified abstractions of early cybernetics. We cannot simply mimic the brain's output; we must emulate its internal dynamics. This requires viewing the neuron not as a static summing unit, but as it functions in reality: a complex, time-dependent, and event-driven processor.

This section provides a mechanistic overview of the nervous system, translating biological observations into the computational primitives required for neuromorphic engineering. It explores the structural hierarchy of the neuron, the physics of the action potential, and the mathematical models used to capture these dynamics in silicon. ]

#v(2em)
== Neuron Structure & Function <s.neuronstructure>

#serif-text()[ In @s.history we established the neuron as the fundamental computational unit of the brain. While it shares standard cellular machinery like mitochondria and a nucleus with other cells, it is morphologically specialized for information transmission. A neuron consists of three functional zones: ]

#box-text()[
*The Input (Dendrites)*: A branching tree structure that collects signals from thousands of upstream neurons. This is where inputs are integrated.

*The Integration Zone (Soma)*: The cell body where electrical potentials from the dendrites summate.

*The Output (Axon)*: A long, cable-like structure that transmits the neuron's own signal to downstream targets.
]

#serif-text()[ The neuron exhibits a distinct morphological polarization that dictates the direction of information flow. The process begins at the "dendritic arbor", a complex branching structure that maximizes the surface area for synaptic connectivity. These dendrites serve as the primary receptor sites, where neurotransmitters binding to post-synaptic terminals induce local conductance changes. These signals propagate passively toward the soma (cell body), the neuron's central processing unit. The soma acts as an integrator, spatially and temporally summing the incoming synaptic currents. Finally, the processed signal is transmitted via the axon, a singular, elongated projection. In many vertebrate neurons, the axon is insulated by a myelin sheath, which facilitates saltatory conduction---a mechanism that allows high-speed signal propagation over long distances with minimal signal degradation. ]

#figure( image("figures/neuron.png", width:60%), caption: [Image of a neuron, we the soma, axon and the densrites @getz_high-resolution_2022])

#serif-text()[ Functionally, the neuron operates as an electrochemical system enclosed by a cell membrane, known as the "lipid bilayer". This membrane is a thin, fatty structure that is impermeable to ions, acting as an electrical insulator. However, the fluids inside and outside the cell are conductive electrolytes. Consequently, the interaction between the insulating membrane and the conductive fluids creates a biological capacitor, capable of storing charge.

By actively pumping sodium ($"Na"^+$) out and potassium ($"K"^+$) in via the $"Na"^+$-$"K"^+$ ATPase pump, the cell maintains an electrochemical gradient across this capacitor, resulting in a stable "resting potential" of approximately $-70$ mV.

Computation occurs through the modulation of this voltage by competing synaptic inputs. Excitatory inputs cause ion channels to open, allowing positive ions to influx; this reduces the negative charge (depolarization) and pushes the potential toward the firing threshold. Conversely, inhibitory inputs activate channels for negative ions (like Chloride, $"Cl"^-$), driving the potential away from the threshold (hyperpolarization). The soma integrates these opposing push and pull signals. If the aggregate membrane potential surpasses a critical threshold (approximately $-55$ mV), the system undergoes a bifurcating phase transition. Voltage-gated sodium channels cascade open, triggering an @actionpotential\---a rapid, non-linear depolarization spike that propagates down the axon. This mechanism is governed by the "all-or-nothing" principle: the output is discrete and binary, effectively filtering out sub-threshold noise. ]

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

#serif-text()[ Under this formalism, the output of a neuron is modeled not as a continuous signal, but as a "@spiketrain"---a temporal sequence of these Dirac impulses @gerstner_neuronal_2014. For a neuron emitting $N$ spikes at times ${t^((1)), t^((2)), ..., t^((N))}$, the output signal $S(t)$ is defined as: ]

#figure( kind: "eq", supplement: [Equation], caption: [A spike train represented as a sum of Dirac delta functions.], [ $ S(t) = sum_(f=1)^(N) delta(t - t^((f))) $
])<spike_train>

#serif-text()[ This abstraction allows the post-synaptic effect to be modeled using linear systems theory. In neuron models that use this framework, the interaction is treated as instantaneous charge deposition: the arrival of a delta function $delta(t-t_f)$ imparts a discrete step-change to the post-synaptic current. This mimics the rapid opening of ion channels without requiring the computational overhead of simulating the complex voltage trajectory. The shift from continuous values to discrete spike trains fundamentally alters the computational paradigm, moving from spatial representations (magnitude-based) to spatio-temporal representations (time-based). ]

#figure(include("figures/spiketrain.typ"),caption:[Transformation of continuous membrane voltage (top) into a discrete spike train (bottom). The membrane voltage trace is a recording from the L5 dorsal rootlet of a rat using a multiple electrode array @metcalfe_action_2020], placement: auto)

#v(2em)
== Neuron Models <s.neuronmodels>

#serif-text()[ In the quest to simulate the brain, there exists a fundamental trade-off between biological realism and computational efficiency. At the high end of the spectrum lie conductance-based models, most notably the Hodgkin-Huxley model. This formalism describes the neuron not as a simple computational unit, but as an electrical circuit with variable resistors representing the precise, non-linear opening and closing dynamics of specific ion channels (sodium, potassium, leak) @hodgkin_quantitative_1952.

Large-scale initiatives, such as the Blue Brain Project, utilize even more granular "multi-compartment" models. These simulations treat the neuron as a complex 3D structure, discretizing the dendritic arbor and axon into hundreds of segments to model how current flows through the specific morphology of the cell @markram_blue_2006. While invaluable for pharmacological research, these models are computationally prohibitive for large-scale neuromorphic engineering. Simulating a mere second of biological time for a small network using these equations requires supercomputing resources.

To build practical, scalable neuromorphic hardware, we must abstract these biophysical details into a phenomenological model. We seek a mathematical framework that captures the essential computational properties---integration, leakage, and thresholding---without simulating the underlying molecular physics. ]

#v(1em)
=== The @lif Model <s.biolif>

#serif-text()[ The standard approximation used in neuromorphic engineering is the @lif model @gerstner_neuronal_2014. This framework aligns perfectly with the "point process" abstraction established in the previous section, as it treats action potentials as instantaneous, discrete events. Its state is defined by a single scalar variable, the membrane potential $u(t)$. The sub-threshold dynamics are governed by a linear differential equation analogous to a simple $R C$ (Resistor-Capacitor) circuit.]

#figure(include("figures/lifcircuit.typ"), caption:[Electronic circuit modeling the dynamics of the LIF])

#figure( kind: "eq", supplement: [Equation], caption: [The LIF differential equation. The change in voltage is driven by the leak (decay to rest) and the input current. $tau_m = R C$ is known as the time constant and dictates how fast the membrane potential drains], $ tau_m​(dif u)/(dif t)=−(u−u_"rest")+R I(t) $)<lif_eq>

#serif-text()[ Where $tau_m$ is the membrane time constant (determining how fast the neuron "forgets"), $u_"rest"$ is the resting potential, $R$ is the membrane resistance, and $I(t)$ is the input current.

Connecting this to the spike train abstraction derived in the previous section, the input current I(t) is not continuous. It is a sequence of discrete events arriving from pre-synaptic neurons $j$ with weight $w_j$. Mathematically, this is modeled as a sum of Dirac delta functions: ]

#figure( kind: "eq", supplement: [Equation], caption: [Synaptic input modeled as a weighted sum of Dirac delta functions.], $ I(t)=sum j w_j sum f delta(t−t_j(f)) $)<lif_input>

#serif-text()[ Because the differential equation is linear below the threshold, we can solve it analytically. The membrane potential $u(t)$ becomes a convolution of the input spike train with the system's impulse response (a decaying exponential kernel). This means the potential at any moment is simply the sum of the decaying traces of all past spikes: ]

#figure( kind: "eq", supplement: [Equation], caption: [The analytical solution for the membrane potential. The current voltage is the superposition of all past inputs, decayed by time constant $tau_m$.], $ u(t)=u_"rest"+sum j w_j sum f exp(−(t−t_j(f))/tau_m) $)<lif_sol>

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

To capture these dynamics without reverting to the computationally heavy Hodgkin-Huxley equations, we employ the @glif model @gerstner_neuronal_2014. This extends the system by introducing a second state variable, $w(t)$, representing cellular adaptation. ]

#figure( kind: "eq", supplement: [Equation], caption: [The Adaptive GLIF system. The adaptation variable $w$ provides negative feedback, enabling complex dynamics like bursting and adaptation.], [
$ tau_m (dif u)/(dif t) &= -(u - u_"rest") + R I(t) - w \
  tau_w (dif w)/(dif t) &= a(u - u_"rest") - w $
])<glif_eq>

#serif-text()[ In this coupled system, $w$ provides a negative feedback loop. Every time the neuron spikes, $w$ increments by a constant $b$, acting as a physiological drag on the membrane potential. By adjusting the coupling parameters between $u$ and $w$, this two-dimensional system can be tuned to emulate the full spectrum of biological firing patterns.

It is natural to question whether such a mathematically reduced model can genuinely capture the behavior of biological neurons. While the @glif model discards the specific ionic mechanisms of the Hodgkin-Huxley equations, empirical validation demonstrates that it retains superior computational dynamics for large-scale modeling.

In the 2008 _Quantitative Single-Neuron Modeling Competition_ organized by the INCF, phenomenological models like the @glif (specifically the Adaptive Exponential Integrate-and-Fire) unexpectedly outperformed highly detailed biophysical models in predicting the precise spike times of real cortical neurons.

This counter-intuitive success is due to parameter sensitivity. Complex conductance-based models have dozens of unobservable parameters that are difficult to tune. In contrast, the @glif model captures the "net effect" of these mechanisms using macroscopic parameters that can be robustly fitted to data. As demonstrated by Izhikevich (2003) @izhikevich_simple_2003, this simple system of two differential equations is capable of reproducing all known firing patterns observed in the mammalian cortex @gerstner_how_2009. We leverage this state-dependent adaptation to implement Model D (@s.decoding), evaluating its effectiveness in temporal sequence decoding. ]

#figure(include("figures/izhikevichpatterns.typ"), caption:[The @glif model is capable of reproducing the diverse firing patterns of biological cortical neurons, as categorized by Izhikevich (2003) @izhikevich_simple_2003.], placement: auto)

#serif-text()[ Consequently, for the purpose of neuromorphic engineering, the @glif model represents the optimal trade-off between biological fidelity and computational efficiency. ]

#v(2em)
== Neural Coding <s.neuralcoding>

#serif-text()[ In classical digital computing, information is represented by combining bits into richer structures, such as floating-point or integer numbers. For instance, the luminance of a pixel is typically stored as a discrete 8-bit or 32-bit integer. Conversely, analog electronics represent values as continuous currents or voltages, offering infinite resolution within the dynamic range of the hardware. ]

#figure(include("figures/digitalanalogrepresentation.typ"), caption:[ Digital left analog right representation])

#serif-text()[ The biological brain occupies a unique middle ground. While neurons operate using analog membrane potentials, their communication output---the action potential---is discrete and binary. As established in @s.actionpotential, the waveform of a spike is stereotypical; it looks like a "digital bit" in amplitude. However, unlike a digital computer which is synchronized to a rigid clock, these spikes occur in continuous time. Therefore, the information in the nervous system is not stored in the shape of the signal, but in the structure of the spike train itself.

Deciphering the "Neural Code"---the set of rules by which sensory stimuli are translated into these spike sequences---remains one of the central challenges in neuroscience. Currently, several coding schemes are hypothesized to coexist, each offering different trade-offs between latency, information density, and robustness. ]

#v(1em)
=== Rate Coding <s.ratecoding>

#serif-text()[ The most traditional interpretation of neural activity is rate coding. In this paradigm, information is conveyed by the mean firing frequency of a neuron over a specific temporal window. A strong stimulus (e.g., high pressure on skin) elicits a high firing rate, while a weak stimulus results in sparse activity.

This model effectively treats the neuron as an Analog-to-Digital converter where the precise timing of individual spikes is treated as noise; only the average count carries the signal. While rate coding is robust and easily observed in motor neurons, it suffers from a fundamental latency barrier. To estimate a rate with reasonable precision, the post-synaptic neuron must integrate spikes over a significant duration (tens or hundreds of milliseconds). This contradicts the rapid reaction times (often $<100$ ms) observed in biological agents, suggesting that rate coding alone cannot account for time-critical processing @thorpe_speed_1996. ]

#figure(include("figures/rateencoding.typ"), caption:[Rate Coding: The stimulus intensity is encoded in the frequency of the spike train. Stronger stimuli elicit more spikes per second.])

#v(1em)
=== Temporal Coding <s.temporalcoding>

#serif-text()[ To explain the speed of biological processing, neuromorphic engineering emphasizes temporal coding. In this regime, the precise timing of a spike carries significant information. A primary example is @ttfs coding.

In a @ttfs scheme, the intensity of a stimulus is inversely mapped to the latency of the response relative to a stimulus onset @rullen_rate_2001. A stronger input causes the neuron to integrate and cross the threshold faster, firing earlier than neurons receiving weak inputs. This shifts the computational model from counting spikes to a "race" between spikes.

In a network utilizing lateral inhibition (@wta), the first neuron to fire inhibits its neighbors, allowing a decision to be made as soon as the first meaningful bit of data arrives. This eliminates the need to wait for a time window to close, drastically reducing latency. Furthermore, since @ttfs coding prioritizes the strongest signals, it acts as a natural filter: the most prominent features arrive first, allowing the system to process signal over noise. This @ttfs scheme forms the primary encoding modality for our MNIST experiments, as detailed in @s.encoding. ]

#figure(include("figures/temporalcoding.typ"), caption:[Temporal Codiing (@ttfs): Stimulus intensity is encoded in the latency of the response. Stronger inputs ($I_1$) trigger an earlier spike ($t_1$) compared to weaker inputs ($I_2$).])

#v(1em)
=== The Phase Ambiguity Problem <s.phaseambiguity>

#serif-text()[ A critical challenge in temporal coding is the need for a temporal reference frame. In Rate Coding, the "phase" (absolute timing) is irrelevant. However, in Temporal Coding, a spike at time $t$ only has meaning relative to a reference $t_0$. If the receiver does not know when the stimulus started, it cannot decode the latency.

In engineering, this is solved by a global clock or a "frame start" signal. In the brain, evidence suggests that background oscillatory rhythms (brain waves, such as theta or gamma cycles) may serve as this global reference, allowing populations of neurons to synchronize their "clocks" and decode relative timings accurately @basso_gamma_2016. ]

#figure(include("figures/phaseambiguity.typ"), caption:[The phase ambiguity problem in temporal encoding. Spikes occurring at the same relative phase ($phi_1$ and $phi_2$) across different oscillation cycles are mathematically indistinguishable ($phi_1 = phi_2 (mod 2pi)$). Without a mechanism to track the global cycle count, downstream neurons cannot determine whether a spike represents a delayed response to a previous stimulus or an early response to a new one.])

#v(1em)
=== Population & Sparse Coding <s.populationcoding>

#serif-text()[
While single-neuron codes provide the basic signaling mechanism, the brain employs ensemble strategies to ensure robustness and precision. In population coding, variables are represented by the joint activity of a large group of neurons, each with broad, overlapping tuning curves. A classic example is found in the Primary Visual Cortex (V1), where orientation-selective neurons each respond maximally to a preferred angle but also fire weakly for nearby orientations. By reading the weighted population vector across the group, the network reconstructs the stimulus with far greater precision than any individual cell could provide alone.
The brain further optimizes for metabolic efficiency through sparse coding, where only a small fraction of neurons are active at any moment. This strikes a mathematical balance between representational capacity and energy cost, and is naturally enforced by lateral inhibition circuits that suppress weaker, competing responses. ]

#v(1em)
=== Coexistence of Codes <s.coexistenceofcodes>

#serif-text()[ These schemes are not mutually exclusive but complementary. A circuit may use @ttfs for a rapid initial response---alerting the system to a salient change---before transitioning to rate-based activity for sustained processing. Neuromorphic systems often adopt this hybrid approach, using temporal codes for energy-efficient sparse event transmission and rate-based readouts for interfacing with downstream control systems. This thesis follows the same principle, using @ttfs encoding for the transmission of visual features combined with a population-level representation at the hidden layer. ]

#v(2em)
== Neural Networks <s.neuralnetworks>

#serif-text()[ Having established the mathematical description of the individual neuron, we now turn to the collective behavior of these units. A single neuron, regardless of its dynamical complexity, is of limited computational utility in isolation. Functional intelligence emerges only when these units are organized into specific structural topologies. We implement these topologies in our experimental framework, as detailed in @s.network.

The brain is not a random mesh of connections; it is constructed from recurring architectural "motifs" that appear across various cortical areas. Understanding these motifs is essential for designing neuromorphic systems that transcend simple feed-forward processing. ]

#v(1em)
=== Synaptic Efficacy & Weights <s.synapticefficacy>

#serif-text()[ Before analyzing the structural topology of networks, we must define the fundamental unit of connectivity: the synapse. In the biological brain, neurons do not touch; they are separated by a microscopic gap known as the synaptic cleft. Communication across this gap is chemical, mediated by the release of neurotransmitters.

The efficiency of this transmission---determined by factors such as the amount of neurotransmitter released and the number of post-synaptic receptors---is abstracted in mathematical models as the synaptic weight ($w$).

In the @snn formalism, the weight represents a scaling factor for the incoming spike. When a pre-synaptic neuron $j$ fires a spike at time $t_j$, it induces a @psc in neuron $i$ scaled by the weight $w_(i j)$. Mathematically, the total synaptic input $I(t)$ is the weighted sum of all incoming spike trains: ]

#figure( kind: "eq", supplement: [Equation], caption: [The synaptic input current as a weighted sum of incoming impulses.], [
$ I_i(t) = sum_j w_(i j) dot S_j(t) $
])<synaptic_input>

#serif-text()[ Synaptic weights determine not just the magnitude but also if the synapse is excitatory or inhibitory. ]
#box-text()[
*Excitatory Synapses ($w > 0$):* These depolarize the target neuron, pushing its membrane potential closer to the firing threshold (e.g., Glutamate synapses).

*Inhibitory Synapses ($w < 0$):* These hyperpolarize the target neuron, pushing the potential away from the threshold (e.g., GABA synapses). ]

#serif-text()[ A fundamental constraint in biological networks, known as Dale's Principle, states that a neuron performs the same chemical action at all of its synaptic outputs @eccles_cholinergic_1954. This means a neuron is strictly excitatory or strictly inhibitory; it cannot send positive signals to one neighbor and negative signals to another. While standard @ann:pl often violate this rule for mathematical convenience (allowing weights to flip signs during training), bio-plausible neuromorphic architectures often enforce this constraint to mimic the distinct populations of Pyramidal (excitatory) and Interneuron (inhibitory) cells found in the cortex. ]

#serif-text()[ The network must maintain a precise Excitation-Inhibition (E/I) Balance. The brain operates at a critical point of instability: ]

#box-text()[
*Excess Excitation* leads to runaway feedback loops (analogous to epileptic seizures).

*Excess Inhibition* leads to signal extinction (quiescence). ]

#v(1em)
=== Directionality <s.directionality>

#serif-text()[ Structurally, neural topologies can be categorized by the flow of information.

In sensory peripheries (such as the retina) and early processing stages, information flows unidirectionally from input to output. This topology supports rapid, reflex-like feature extraction. This configuration is known as a feed-forward network, which is mathematically equivalent to a Directed Acyclic Graph (@dag) and serves as the standard architecture for most Deep Learning @cnn:pl.

In higher cognitive areas, the dominant topology is recurrence. Neurons form feedback loops, connecting back to themselves or to distinct layers. This recurrence introduces a time component to the computation, transforming the network into a dynamical system where the current output depends not only on the input but on the network's previous state (history). ]

#figure(include("figures/connectivity.typ"), caption:[Network topologies. (A) Feed-Forward. (B) Recurrent.])

#v(1em)
=== Synaptic Hypothesis: Structure As Function <s.synaptichypothesis>

#serif-text()[ A foundational premise in neuromorphic engineering, derived from biological observation, is that the neuron operates largely as a generic processing unit. The functional identity of a neural circuit---whether it processes visual stimuli or governs motor control---is determined principally by the topology and efficacy of its synaptic interconnections.

This paradigm, known as the Synaptic Hypothesis, posits that the physical configuration of synaptic weights constitutes the substrate for all computation and memory. Unlike traditional Von Neumann architectures, where data is retrieved from a distinct memory module and processed in a central CPU, biological systems eliminate the distinction between "data" and "program." Memory is not a static artifact, but a latent computational potential distributed across the network's structural graph. Consequently, learning in a neuromorphic system is realized through the physical alteration of these synaptic weights, ensuring robust, decentralized processing that is inherently resistant to localized hardware failure (graceful degradation). ]

#v(1em)
=== Inhibition Patterns <s.inhibitionpatterns>

#serif-text()[ A ubiquitous micro-circuit motif in the cortex is lateral inhibition. In this configuration, an active excitatory neuron stimulates distinct inhibitory interneurons, which in turn suppress the activity of neighboring excitatory neurons. This competition engenders a @wta dynamic: as one neuron---representing a specific feature or decision---becomes active, it effectively silences its competitors. In the context of neuromorphic engineering, @wta circuits are indispensable; they provide a physical mechanism for both noise reduction, by actively suppressing weak, sub-threshold signals, and categorical decision making, enabling the circuit to autonomously select the most salient option without the need for a central processor to sort or compare values. We utilize this @wta motif in our output layer to enforce categorical decision-making. ]

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
*Synaptogenesis*: When neurons are repeatedly co-active but lack a direct connection, the brain can physically grow new dendritic spines and axonal boutons to bridge the gap. This effectively alters the network's topology, creating new pathways for information flow where none existed before.

*Pruning*: Equally critical is the removal of redundant or noisy connections. During sleep and developmental critical periods, the brain aggressively prunes weak synapses. This "sparsification" reduces metabolic cost and increases the signal-to-noise ratio of the circuit by removing irrelevant pathways. ]

#serif-text()[In the context of the Synaptic Hypothesis, structural plasticity represents the "compiling" of temporary associations into permanent hardware architecture. ]

#v(1em)
=== Synaptic Plasticity <s.synapticplasticity>

#serif-text()[ Once a structural connection exists, its efficacy---the magnitude of the post-synaptic response to a pre-synaptic spike---must be tuned. In biological terms, this "weight" corresponds to the amount of neurotransmitter released and the density of receptors on the receiving side. This fine-grained adjustment is governed by local learning rules. ]

#v(1em)
=== Hebbian Learning: Rate-Based Correlation <s.hebbianlearning>

#serif-text()[ The foundational axiom of biological learning was postulated by Donald Hebb in 1949. Hebb proposed that synaptic efficiency is a function of the correlated activity between two neurons. Colloquially summarized as "Neurons that fire together, wire together," this rule implies that the brain learns by detecting statistical regularities in sensory input.

Mathematically, if neuron $A$ consistently takes part in firing neuron $B$, the connection from $A$ to $B$ is strengthened. This mechanism allows the brain to perform unsupervised clustering, physically encoding associations between features that occur simultaneously in the environment (e.g., the smell of smoke and the sight of fire). ]

#v(1em)
=== Spike-Timing-Dependent Plasticity (STDP) <s.stdp>

#serif-text()[ Modern neuroscience has refined Hebb’s macroscopic theory into a precise, millisecond-scale mechanism known as @stdp. Unlike rate-based models, @stdp operates on the precise timing of individual action potentials, introducing the critical element of causality.

The @stdp rule adjusts the synaptic weight based on the relative timing difference ($Delta t$) between the pre-synaptic input and the post-synaptic output: ]

#box-text()[
*@ltp*: If the input spike arrives *before* the output spike ($Delta t > 0$), it implies the input contributed to the firing. The synapse is strengthened to reinforce this causal link.

*@ltd*: If the input spike arrives *after* the output spike ($Delta t < 0$), the input was irrelevant to the decision. The synapse is weakened. ]

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

Fundamentally, a deep learning model operates as a function approximator. We assume the existence of an unknown underlying function $f: X arrow Y$ that perfectly maps inputs to their target outputs. Since this true function is unknown, we construct a family of hypothesis functions $f_bold(theta)(bold(x))$ to approximate it. Here, $bold(theta) in RR^d$ represents the state of the system---a vector containing all tunable parameters, such as synaptic weights or biases. The dimensionality $d$ corresponds to the degrees of freedom of the model.

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

#serif-text()[ Strictly minimizing the empirical cost carries the risk of overfitting---the model memorizes training data including noise rather than learning the underlying function. In biological systems this is naturally regulated by metabolic constraints; the brain prunes weak connections to maintain a sparse topology, effectively trading model complexity for generalization. In artificial systems this is managed via explicit regularization penalties added to the cost function. ]

#v(1em)
=== Unsupervised Learning <s.unsupervised>

#serif-text()[ While supervised learning relies on labeled targets, biological systems predominantly learn via Unsupervised Learning. In this regime, the dataset consists only of input vectors $X = {bold(x)_1, ..., bold(x)_N}$. The optimization objective shifts from minimizing prediction error to minimizing representation error.

Mathematically, the goal is often to discover a lower-dimensional manifold that efficiently captures the structure of the data. A common formulation is the minimization of Reconstruction Loss, where the system attempts to compress the input into a latent code and reconstruct it:

$ J(bold(theta)) = 1/N sum_(i=1)^N || bold(x)_i - f_"decode"(f_"encode"(bold(x)_i; bold(theta))) ||^2 $

Alternatively, the system may optimize for clustering density or distances between feature centroids. The distinction between supervised and unsupervised learning is critical for Neuromorphic Engineering, as biological plasticity rules (like @stdp) are unsupervised, functioning by detecting statistical correlations in the input stream to build internal representations without external labels. ]

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
*Local Connectivity:* Neurons only process a small, local receptive field, analogous to the biological visual cortex.

*Weight Sharing:* The exact same kernel is applied across the entire image, drastically reducing the number of tunable parameters and establishing translation invariance (a feature learned in one corner of an image can be recognized anywhere else).]

#serif-text()[While CNNs are the standard baseline for spatial processing, they remain fundamentally synchronous and frame-based, evaluating the entire image structure in dense mathematical passes regardless of local activity. ]

#v(2em)
== Why Is Deep Learning Inefficient? <s.whyisdlinefficient>

#serif-text()[ While the matrix-centric formulation of Deep Learning enables high-throughput parallelization on GPUs, it fundamentally conflicts with the physical constraints of modern computing hardware. As models scale to billions of parameters, the primary bottleneck shifts from algorithmic capability to physical realizability. This inefficiency manifests in four distinct engineering dimensions: ]

#v(1em)
=== The Von Neumann Bottleneck & Data Movement <s.vonneumanbottleneck>

#serif-text()[ The most significant physical limitation is the Von Neumann Architecture, which physically separates the Processing Unit from the Memory Unit. To perform a single inference step, the processor must fetch the entire weight matrix from off-chip DRAM to on-chip registers, perform the calculation, and write the results back.

According to Horowitz and Dally @horowitz_11_2014, fetching a 32-bit value from off-chip DRAM consumes approximately 640 pJ, whereas performing a floating-point addition on that value consumes only 0.1 pJ. The system expends 99.9% of its energy transporting data, and only 0.1% actually computing. ]

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
*Co-location of Memory and Compute (The Synaptic Principle):* Neuromorphic architectures eliminate the Von Neumann bottleneck by distributing memory across the silicon die. Each artificial neuron stores its own state and synaptic weights locally, processing data *in situ* to eliminate the energy cost of shuttling data.

*Event-Driven Asynchrony (The Action Potential Principle):* Neuromorphic systems abandon the global clock. They operate asynchronously, driven strictly by the arrival of data. If a part of the network is not processing information, it consumes negligible power, ensuring energy scales linearly with task complexity rather than network size.

*Sparse Communication (The Spike Principle):* Neuromorphic systems utilize binary Spikes for communication. Information is encoded in the precise timing of events rather than complex magnitudes, drastically reducing the bandwidth required to transmit information between neurons. ]

#v(2em)
== Training Spiking Networks <s.trainingsnn>

#serif-text()[ While the physical architecture of neuromorphic systems is highly efficient, training these networks presents a fundamental mathematical challenge. Standard deep learning relies on gradient descent, but backpropagation cannot be directly applied to native @snn:pl.

In a spiking network, the neuron's activation function is a discontinuous step function (the Dirac delta event threshold). The derivative of this function is zero everywhere except at the exact moment of the spike, where it is undefined. Consequently, gradients calculated using the chain rule immediately drop to zero---known as the "Dead Neuron" problem---preventing error signals from flowing backward through the network to update the weights.

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


#pagebreak()

= Related Works <s.relatedworks>

#serif-text()[ The shift towards neuromorphic computing is largely driven by the urgent need for energy-efficient machine intelligence. Recent literature highlights a rapidly expanding landscape of opportunities for neuromorphic algorithms, moving beyond isolated benchmarks toward real-world applications. The methodologies implemented in this thesis---specifically temporal coding, sparsity, and local learning---intersect directly with several active areas of state-of-the-art research. ]

#v(1em)
=== Time-Based Coding and Biological Plausibility

#serif-text()[ The foundational premise of our @ttfs architecture is that precise spike timing carries
significant information. This is powerfully supported by recent biological evidence; in vivo
recordings from the human cortex confirm that neuronal sequences during population bursts
explicitly encode information through their temporal order @xie_neuronal_2024.

Translating these biological temporal sequences into functional artificial networks has been a
major focus of recent engineering efforts. Han and Roy have demonstrated that deep spiking
neural networks can achieve extreme energy efficiency specifically by leveraging time-based
coding paradigms @roy_towards_2019. Concurrently, theorists are continually refining how
biological constraints---such as excitation-inhibition balance and adaptive homeostatic
currents---can be integrated into leaky integrate-and-fire networks to achieve optimal
efficient coding.

However, a persistent open question in the field concerns the compatibility of standard neuron
models with the strict priority ordering required by @ttfs decoding. As demonstrated in Phase I
of this thesis, classical Leaky Integrate-and-Fire dynamics are fundamentally misaligned with
@ttfs principles: the exponential leak mechanism actively penalizes early, high-salience spikes,
causing the neuron to implicitly favor discordant sequences. This motivates the development of
the momentum-based and state-discounting neuron models evaluated herein, which are designed to
mathematically align integration dynamics with the "strongest-first" priority of temporal coding. ]

#v(1em)
=== Hardware Acceleration and Sparsity

#serif-text()[ A core theoretical advantage of the @ttfs encoding utilized in our experiments is spatial and
temporal sparsity. However, realizing these efficiency gains requires bridging the gap between
software simulation and physical deployment. Recent SOTA implementations have focused heavily
on exploiting this sparsity directly on customized hardware. For example, Sommer et al.
@sommer_efficient_2022 proposed novel memory interlacing and hardware acceleration architectures
for sparsely active convolutional @snn:pl on @fpga:pl, achieving significant speedups by
ensuring processing time scales directly with the number of spikes.

Furthermore, event-driven algorithms are increasingly being paired with native event-based
sensors. Applications such as spiking optical flow estimation from dynamic vision sensors have
been successfully deployed on specialized neuromorphic substrates like IBM's TrueNorth
@haessig_spiking_2017, proving the viability of low-power, asynchronous processing in dynamic
environments.

A critical finding of this thesis is that these efficiency gains cannot be realized by simply
simulating sparsity on conventional GPU hardware. While the @ttfs @snn achieves an 85.2%
reduction in theoretical synaptic operations compared to the ANN baseline, GPU architectures
optimized for dense matrix multiplication continue to execute the underlying floating-point
operations for quiescent neurons masked to zero. This "Sparsity Paradox" reinforces the
broader finding of the hardware acceleration literature: energy efficiency in neuromorphic
algorithms is contingent on deployment on native event-driven substrates such as Intel Loihi
or IBM TrueNorth, not simulation on von Neumann processors. ]

#v(1em)
=== ANN-to-SNN Conversion

#serif-text()[ A significant portion of contemporary neuromorphic engineering focuses on bridging the gap
between classical Deep Learning and spiking hardware via offline training. Foundational work
by Rueckauer et al. demonstrated that continuous-valued networks could be losslessly converted
into @snn:pl by carefully scaling weights and normalizing spiking thresholds
@rueckauer_conversion_2017. However, the vast majority of these conversion methodologies rely
on Rate Coding, requiring hundreds of simulation ticks to approximate continuous activation
values via temporal averaging.

Applying zero-shot weight transfer directly to a @ttfs encoding, as explored in Phase II of
this thesis, represents a distinct and more sensitive challenge: quantization errors impact
temporal priority rather than just average frequency, and the absence of a rate-averaging
window means the network must reach a correct decision from the very first spike wavefront.
Despite this constraint, the momentum-based Model C architecture achieves 94.50% top-1
accuracy on MNIST with a mean decision latency of 8.4 ticks, demonstrating that the spatial
hierarchies learned by gradient descent are recoverable from a purely temporal spike ordering
without any intermediate retraining or fine-tuning. ]

#v(1em)
=== Unsupervised STDP and Competitive Learning

#serif-text()[ The application of unsupervised Spike-Timing-Dependent Plasticity (@stdp) to benchmark
datasets like MNIST serves as the standard baseline for evaluating biologically plausible
learning. The foundational architecture for this approach was established by Diehl and Cook
(2015), who demonstrated that a spiking network utilizing @stdp, lateral inhibition, and
adaptive homeostatic thresholds could achieve 95% accuracy on MNIST without any labels
@diehl_unsupervised_2015. However, it is critical to note that their result relied on a
massive spatial population of 6,400 hidden neurons and extended rate-based integration windows
to form robust, overlapping receptive fields.

Phase III of this thesis evaluates these same principles under substantially more severe
constraints: a 128-neuron architecture operating under strict @ttfs single-spike encoding,
where each neuron fires at most once per saccade and spatial redundancy is eliminated. Under
these conditions, the network achieves 44.3% top-1 accuracy---a result that, while far below
the rate-coded ceiling, nonetheless demonstrates successful unsupervised geometric clustering
without any global error signal. The performance gap relative to Diehl and Cook is therefore
not a failure of the @stdp rule itself, but a direct and quantifiable consequence of removing
the spatial redundancy on which rate-coded representations depend. This provides a controlled
data point for understanding the interaction between network scale, temporal encoding
precision, and the representational capacity of local learning rules. ]

#pagebreak()

= Method <s.method>

#serif-text()[ This chapter details the specific implementations of the neuromorphic architectures proposed to address the limitations of standard deep learning. Aligning with the biological constraints of sparsity, asynchrony, and locality established in previous chapters, we outline the construction and evaluation of a @snn.

To empirically validate the theoretical advantages of neuromorphic algorithms, we evaluate the system on a benchmark image classification task. The experiment is bifurcated into three distinct phases:]

#box-text()[
*Phase I---Neuron Model Evaluation:* Evaluating the decoding efficiency and accuracy of different simulated spiking models.

*Phase II---Inference Via Weight Transfer:* Evaluating the zero-shot performance of these @snn:pl initialized with weights directly mapped from a classically trained Artificial Neural Network (ANN).

*Phase III---Native Unsupervised Learning:* Training the @snn from scratch utilizing local @stdp.
]

#v(2em)
== Dataset & Pre-processing <s.dataset>

#serif-text()[ To benchmark these algorithms, we require a dataset that necessitates the extraction of complex spatial features but remains computationally tractable for rapid experimental iteration. We utilize the MNIST database of handwritten digits @lecun_gradient-based_1998.

The dataset consists of a training set of 60,000 examples and a test set of 10,000 examples of digits (0-9). Each instance is a $28 times 28$ pixel grayscale image. While standard deep learning models routinely score over 90% accuracy on this task, making it largely a solved problem in classical AI, its well-understood feature space makes it an ideal, isolated baseline. Because the spatial hierarchy of MNIST is relatively shallow, it allows us to evaluate the efficacy of neuromorphic learning rules without the confounding variables introduced by massive, multi-layered convolutional architectures.

Crucially, the MNIST images are pre-processed by the dataset creators to be size-normalized and centered within the pixel grid using the center of mass of the pixels. This spatial alignment is a vital prerequisite for our chosen network topology. Unlike @cnn:pl, which slide localized filters across an image, the @fcn utilized in this thesis lacks translation invariance. If a digit were shifted several pixels off-center, the @fcn would perceive it as an entirely novel pattern. The pre-centered nature of MNIST mitigates this limitation, ensuring that the network can reliably map specific geometric strokes to specific input neurons.

#figure( image("figures/mnist_grid.png", width:100%), caption: [Sample of the MNIST dataset. The 28x28 images are normalized and flattened into 1D vectors before being translated into temporal spike events.])

Furthermore, the dataset exhibits a high degree of spatial sparsity. In a typical MNIST image, the vast majority of pixels represent the empty background. From a neuromorphic engineering perspective, this sparsity is highly advantageous. As established in the theoretical framework, event-driven systems expend energy strictly when events occur. A sparse input array ensures that the majority of input neurons remain quiescent, minimizing bus congestion and validating the energy-efficiency claims of the proposed @snn.

Before the raw images can be converted into temporal spike trains, they must undergo standard spatial pre-processing to ensure compatibility with the network's mathematical boundaries. This consists of two primary transformations: ]

#box-text()[
*Normalization*: Raw pixel intensities in the MNIST dataset range from $0$ (pure black) to $255$ (pure white). To stabilize the learning algorithms and ensure consistent weight scaling, these values are strictly normalized to a continuous float range of $p_i in [0.0, 1.0]$.

*Flattening*: Because this thesis utilizes a Fully Connected Network (FCN) to facilitate direct weight transfer, the 2D spatial structure of the images must be unrolled. Each $28 times 28$ matrix is flattened into a 1-dimensional vector of $784$ elements. ]

#serif-text()[ Consequently, every individual image is presented to the system as a discrete array of $784$ normalized intensities. In the classical Artificial Neural Network (ANN), these continuous values are fed directly into the input neurons. However, because @snn:pl operate exclusively on discrete events, these normalized values must be passed through a temporal encoding algorithm before inference or learning can begin. ]

#figure( image("figures/mnist_histogram.png"), caption: [Sample of the MNIST dataset. The 28x28 images are normalized and flattened into 1D vectors before being translated into temporal spike events.])

#v(2em)
== Network Architecture <s.network>

#serif-text()[ To facilitate a direct, one-to-one mapping of synaptic weights from the @ann to the @snn, both models must share an identical macroscopic topology. That is the motivation for why this implementation utilizes a @fcn, also known as a @mlp, rather than a @cnn.

While @cnn:pl are the standard baseline for vision tasks due to their spatial inductive biases, transferring convolutional kernels into a spiking substrate introduces significant mapping complexities---specifically the need to physically unroll and duplicate shared weights across the spiking array. An @fcn provides a straightforward, mathematically transparent architecture for cleanly evaluating direct weight transfer and @stdp without confounding architectural variables.

The network is structured as a shallow hierarchy to capture the primitive geometric features of the dataset. Let $N_l$ denote the number of neurons in layer $l$. The formal architecture is defined as follows: ]

#box-text()[
*Input Layer ($L_0$):* The $28 times 28$ pixel grayscale images are flattened into a 1D vector, requiring $N_0 = 784$ input neurons.

*Hidden Layer ($L_1$):* A fully connected intermediate layer consisting of $N_1 = 128$ neurons. The synaptic connections are defined by the weight matrix $W^((1)) in bb(R)^(N_1 times N_0)$.

*Output Layer ($L_2$):* $N_2 = 10$ neurons, corresponding directly to the categorical digit classes (0 through 9). The connections from the hidden layer are defined by the weight matrix $W^((2)) in bb(R)^(N_2 times N_1)$.
]

#figure( include("figures/architecture.typ"), caption: [Network architecture. The 28x28 images are flattened and passed through a Fully Connected Network. Both the @ann and @snn share this identical macroscopic topology.])

#v(1em)
=== Weight Initialization And Biases <s.weightinit>

#serif-text()[ For the baseline @ann, the weight matrices are initialized using standard PyTorch defaults to ensure stable variance during the initial forward passes.

A critical architectural consideration in @ann\-to-@snn conversion is the handling of bias vectors ($b^((l))$). While standard ANNs utilize biases to shift activation functions, representing a static continuous bias in a spiking network requires either injecting a constant background current into the neuron or actively modifying the neuron's firing threshold. To maintain pure, event-driven sparsity and isolate the performance of the synaptic weights, explicit bias terms were omitted entirely from both architectures. ]

#v(1em)
=== ANN-SNN Compatibility <s.ann-snncompatibility>

#serif-text()[ Despite the macroscopic symmetry required for weight sharing, the microscopic dynamics of the two networks differ fundamentally. The offline @ann utilizes standard continuous activation functions (specifically ReLU) to compute smooth gradients during backpropagation.

In contrast, the @snn replaces these continuous functions with Integrate-and-Fire neurons governed by a strict voltage threshold. This simulates the biological "all-or-nothing" action potential, acting as a hard step function. Furthermore, the spiking architecture utilizes @wta at the output layer, and during training @wta is used at the hidden layer as well. As the network integrates evidence over time, the first output neuron to reach its threshold heavily suppresses its competitors, forcing a definitive categorical decision and actively filtering sub-threshold noise. ]

#v(1em)
#mini-header()[@wta Vs Lateral inhibition]

#serif-text()[ In biological neural networks, competitive learning is typically enforced via dense recurrent collateral connections that provide continuous lateral inhibition. When a neuron fires, it physically suppresses its neighbors' membrane voltages, forcing the network to specialize and preventing multiple neurons from learning the identical feature.

However, replicating continuous lateral voltage suppression in a discrete-time software simulation introduces significant computational overhead and chaotic voltage fluctuations during the forward integration pass. To maintain a clean, deterministic forward pass while still enforcing strict competitive specialization, this implementation completely bypasses continuous lateral inhibition in favor of an algorithmic @wta mechanism.

Instead of exchanging inhibitory synaptic blasts during the temporal saccade, the hidden neurons integrate their membrane potentials entirely independently. The competition is resolved strictly post-hoc or at the exact moment of the first spike: the first neuron to cross its threshold triggers a strict @wta condition, claiming the feature and instantly suppressing all competitors from learning or firing further. This algorithmic abstraction perfectly replicates the functional goal of lateral inhibition---ensuring feature decorrelation---while vastly simplifying the network topology and guaranteeing simulation stability. ]

#v(2em)
== SNN Information Representation <s.informationrepresentation>

#serif-text()[ The choice of neural code lays the foundation for information flow and dictates the efficiency of the entire system. While Rate Coding (encoding pixel intensity as spike frequency) is straightforward and simple to implement with standard Integrate-and-Fire neurons, it is inefficient compared to @ttfs. Rate codes require integration over extended time windows to calculate an average, introducing latency and saturating the network bus with redundant spikes. Furthermore, on digital hardware rate coding imposes additional stress on the system due to rapid switching which is very bad for transistor power draw and bus congestion.

To maximize energy efficiency and processing speed, this implementation utilizes a @ttfs temporal encoding @delorme_face_2001. In this regime, a single spike carries the information payload. A high-intensity (bright) pixel triggers an early spike, while a low-intensity (dark) pixel triggers a late spike. This compresses the spatial information into a highly sparse, priority-driven queue; downstream neurons begin processing as soon as the most salient features arrive, without waiting for an entire frame to integrate.

As noted in @s.neuralcoding, temporal codes suffer from Phase Ambiguity---downstream neurons need a reference "clock" to decode latency. To resolve this without relying on a rigid, global system clock, we simulate the biological concept of a saccade (the rapid movement of the eye to fixate on a target). The initial presentation of the image acts as a synchronized global event ($t_0$). All subsequent input spikes are evaluated relative to this saccade onset, providing a natural, biologically plausible temporal reference frame. ]


#v(1em)
=== Encoding <s.encoding>

#serif-text()[ Following the @ttfs principles described in @s.temporalcoding, we convert the continuous pixel intensities of the MNIST dataset into discrete spike trains. For a given input image, we extract the luminance of each pixel and normalize it to a bounded range, where $p_i in [0, 1]$. $1$ representing maximum intensity and $0$ representing the background).

We implement a single, highly sparse linear conversion mapping to evaluate latency dynamics. The pixel intensity $p_i in [0.0, 1.0]$ is inverted such that brighter pixels correspond to shorter delays. In our implementation, the maximum delay permitted for a valid pixel is 32 ticks, despite the full saccade window being 64 ticks. Furthermore, to aggressively enforce input sparsity, any pixel with an intensity below 0.1 is treated as background noise and discarded (assigned a delay of infinity, meaning it never fires).

#figure( kind: "eq", supplement: [Equation], caption: [Intensity-to-Delay Encoding Implementation], [
$ t_i = cases(
  "round"((1.0 - p_i) dot 32) & "if " p_i >= 0.1,
  infinity & "if " p_i < 0.1
) $
])

Under this mapping, the brightest pixels fire immediately near $t=0$, transmitting the most critical structural features of the digit first, while background pixels are entirely suppressed. ]


#v(1em)
=== Decoding With Neuron Models <s.decoding>

#serif-text()[ Decoding the temporal information generated by the @ttfs encoding requires a neuron model capable of accumulating discrete events. However, a fundamental engineering tension exists between biological fidelity, computational efficiency, and how a model dynamically reacts to temporal density.

To systematically evaluate these trade-offs, this thesis implements and benchmarks four distinct neuron models. These models range from simple arithmetic accumulators requiring global synchronization to complex, fully asynchronous dynamic systems. By adjusting the threshold bounds during Phase I experiments, we evaluate how each distinct mathematical approach processes incoming spikes and updates its membrane potential $V_m(t)$ when an event arrives at time $t$, carrying a synaptic weight $w_i$. ]

#v(1em)
#mini-header()[Model A: The Simple Window Integrator (Standard IF)]

#serif-text()[ The most computationally lightweight approach is the standard Integrate-and-Fire (IF) model without any leak or decay mechanisms. In this paradigm, the neuron acts as a pure arithmetic accumulator during the simulation window. ]

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

*Temporal Dynamics:* Static. While highly efficient, this model's integration is purely additive. Its temporal dynamics are rigidly tied to threshold calibration; if the threshold is low, it fires prematurely based solely on early magnitude accumulation, remaining blind to the wider temporal distribution of the spike train.

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

#serif-text()[ Model B---the @lif model is covered in great detail in @s.biolif. Model B fires only if the incoming spike train has a sufficient density of spikes. A sparse spike train cannot overcome the exponentially decaying membrane potential. ]

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

*Temporal Dynamics:* Decay-driven. The exponential leak provides a basic temporal filter that naturally favors spikes arriving in rapid succession. However, its effectiveness is highly dependent on both threshold constraints and the time constant $tau_m$. If the threshold is set too high relative to the input density, the signal degrades before threshold crossing, preventing firing entirely.

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

#serif-text()[ To explore dynamic accumulation without the computational overhead of continuous exponential kernels, Model C utilizes a linear time-dependent accumulator combined with a strict hard reset timer ($T_"window" = 10$ ticks). The arrival of a spike increments both the instantaneous potential $u(t)$ and the integration gradient $I(t)$. However, if the neuron does not reach the threshold within the coincidence window of the initial spike, the timer expires and both state variables are aggressively wiped to 0.0. This is achieved by modeling the membrane potential $u(t)$ as a system of coupled differential equations driven by a sequence of Dirac delta impulses:

  $ dot(I)(t) = sum_i w_i delta(t - t_i) $
  $ dot(u)(t) = I(t) + sum_i w_i delta(t - t_i) $

This formulation ensures that an afferent spike at time $t_i$ with weight $w_i$ exerts a dual influence: it induces an immediate translocation of the potential state while simultaneously incrementing the rate of change for all subsequent integration intervals. Because early spikes establish the baseline slope $I(t)$ for the remainder of the simulation window, they exert a disproportionate momentum on the time-to-threshold. ]

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
*Computational Complexity:* Moderate. While the model successfully replaces power-intensive transcendental operations with simple floating-point additions and a single multiplication per spike event ($I dot Delta t$), it introduces state-management overhead. The system must actively track and decrement the coincidence timers for all currently stimulated neurons, requiring additional conditional logic and memory access to aggressively zero the state variables upon expiration.

*Temporal Dynamics:* Highly reactive. The quadratic-like growth of the potential relative to spike arrival times means earlier spikes violently accelerate the trajectory toward the threshold. By shifting the threshold bounds, we can observe how this momentum-driven integration drastically alters firing latency compared to standard static accumulators.

*Synchronization:* This model requires a rigorous global synchronization protocol (e.g., a saccade-driven clock). In the absence of a periodic global reset to zero the state variables $I(t)$ and $u(t)$, the linear integration would diverge toward hardware saturation limits.
]

#figure(
  include("figures/rampmodel.typ"),
  caption: [Evolution of state variables in Model C. This demonstrates the resulting piecewise linear membrane potential $u(t)$. Note how earlier spikes increase the integration gradient, accelerating the trajectory toward the firing threshold $theta$.]
)

#figure(
  kind: "algo",
  caption: [Model C: Discrete State Update Algorithm with Coincidence Window],
  supplement: [Algorithm],
  mono-text(pseudocode-list(hooks: .5em, indentation: 1em, booktabs: true)[
    linearRampNeuron(S, W, theta, T_window) -> t_fire: #h(1fr)
      + // Let S be the ordered set of spikes $(t_i, w_i)$
      + $u arrow.l 0.0$
      + $I arrow.l 0.0$
      + $t_"prev" arrow.l 0.0$
      + $t_"start" arrow.l infinity$
      +
      + for each $(t_i, w_i) in S$:
        + // If the hard reset timer expired before this spike arrived
        + if $(t_i - t_"start") >= T_"window"$:
          + $u arrow.l 0.0$
          + $I arrow.l 0.0$
          + $Delta t arrow.l 0$
        + else:
          + $Delta t arrow.l t_i - t_"prev"$
        +
        + // If in a resting state, this spike starts the coincidence timer
        + if $u == 0.0$:
          + $t_"start" arrow.l t_i$
        +
        + // Apply momentum integration and incoming weight
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

#serif-text()[ Drawing inspiration from the adaptation mechanisms of the @glif model (@s.glif), Model D introduces a state-dependent penalty to incoming spikes. Rather than penalizing spikes based on the passage of time, this model penalizes spikes based on the current internal state of the neuron.

In this paradigm, the increase in membrane potential is inversely proportional to the current potential itself. When a spike arrives at a neuron in its resting state ($u = 0$), there is no discount, and the full synaptic weight is added. However, if a spike arrives when the neuron is already close to the firing threshold $theta$, its impact is exponentially discounted. ]

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

*Temporal Dynamics:* State-dependent. This mechanism creates a unique temporal compression. The earliest spikes contribute their absolute maximum weight, while late arrivals encounter a partially filled potential and are severely dampened. Testing this model across varying thresholds reveals an asymptotic behavior where the neuron may never fire if the initial temporal momentum is insufficient to overcome the compounding discount.

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
#mini-header()[Thresholding]

#serif-text()[ To rigorously evaluate these distinct temporal dynamics, Phase I of our methodology introduces a variable thresholding benchmark. Rather than relying on a single fixed threshold to measure performance, we systematically adjust the membrane threshold ($theta$) relative to the total sum of the incoming synaptic weights ($sum w_i$).

If $theta$ is configured significantly lower than $sum w_i$, models naturally reach early saturation, firing prematurely based strictly on the magnitude of early-arriving spikes. By sweeping the threshold from low to high (approaching the total weight sum), we can empirically demonstrate how the different internal mechanics---perfect integration, leaky decay, momentum accumulation, and state-discounting---respond to the exact same temporal patterns under varying saturation constraints. ]

#v(2em)
== Training Methodologies <s.training>

#serif-text()[ Following the optimization principles established in @s.optimization, our training methodology combines weight-based synaptic plasticity with topological structural plasticity. In a neuromorphic context, learning is not merely parameter tuning but a physical self-organization of the network. We implement a dual-stage learning process: unsupervised feature discovery via local plasticity rules, followed by a global structural optimization phase.

The core of our learning objective is the detection of spatiotemporal sequences. Recent in vivo recordings from the human cortex confirm that population bursting relies heavily on the temporal order of spikes to encode and categorize information @xie_neuronal_2024. Consequently, in our TTFS paradigm, the relative arrival order of spikes defines the pattern identity.. We hypothesize that a neuron successfully learns a pattern (e.g., sequence $A arrow B arrow C$) when it adjusts its internal thresholds to fire as soon as the first sufficient evidence ($A arrow B$) arrives.

However, this early firing introduces a prediction risk: if the neuron fires based on a prefix ($A arrow B$) but the expected suffix ($C$) fails to arrive, the synaptic efficacy must be penalized. This mimics the biological concept of error-driven learning without a global supervisor; the neuron "predicts" the completion of a learned pattern and self-corrects based on subsequent local evidence. If the predicted input is absent, the weights associated with the prefix are slightly decayed, preventing the network from becoming overly sensitive to incomplete or noisy patterns. ]

#v(1em)
=== Weight Transfer <s.weighttransfer_method>

#serif-text()[ Following the theory of offline training and mapping discussed in @s.weighttransfer_theory, the baseline methodology for translating a conventionally trained @ann into a spiking architecture relies on direct parameter mapping. To maintain strict structural symmetry and perfectly isolate the effects of the temporal encoding, the floating-point weight matrices ($W^((1))$ and $W^((2))$) are transferred one-to-one from the ANN to the SNN: ]

#figure(
  kind: "eq",
  supplement: [Equation],
  caption: [Direct Zero-Shot Weight Transfer],
  [
    $ W_"SNN"^{(l)} = W_"ANN"^{(l)} $
  ]
)

#serif-text()[ Crucially, this transfer is performed "zero-shot"—meaning there is no intermediate retraining, no fine-tuning in the spiking domain, and no discrete quantization of the parameters. The SNN loads the exact FP32 continuous weights optimized by the ANN's gradient descent.

Because the SNN replaces continuous dot-product activations with discrete, time-dependent spike accumulation, continuous activation scales do not naturally align with spiking thresholds. The primary methodological challenge is aligning the dynamic range of the transferred FP32 weights with the voltage integration parameters of Model C. Rather than artificially scaling or altering the synaptic weights to fit a hardcoded threshold, the firing thresholds ($theta$) and momentum parameters of the SNN are dynamically tuned to accommodate the natural variance of the FP32 weights.

By porting these weights directly to the GPU without quantization, the simulator establishes a pure baseline to measure the *Temporal Penalty*—the isolated loss of classification accuracy incurred strictly by transitioning from static continuous activations to time-to-first-spike (TTFS) momentum integration. ]

#v(1em)
=== TTFS STDP Inspired Learning Rule <s.ttfsstdp>

#serif-text()[ Our learning rule adapts the @stdp mechanism from @s.stdp to the @ttfs domain. In standard continuous networks, synaptic weights are updated via global error gradients. In the @stdp paradigm, a synapse $w_(i j)$ connecting a pre-synaptic neuron $i$ to a post-synaptic neuron $j$ is updated based strictly on the temporal difference between their respective firing times.

Let $t_i$ denote the spike time of the input neuron, and $t_j$ denote the spike time of the output neuron. The relative arrival time is defined as $Delta t = t_j - t_i$. Because this architecture utilizes a strict @ttfs encoding where neurons fire at most once per @saccade, the classical continuous @stdp curve is adapted into a discrete, deterministic update rule: ]

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

#serif-text()[ To prevent runaway synaptic growth or catastrophic sign-flipping, the updated weights are strictly clamped to a positive physical range $w_(i j) in [0, W_"max"]$. ]


=== Vectorized Coincidence Detection

#serif-text()[ Traditional biological @stdp relies on continuous exponential decay kernels to modulate synaptic plasticity. However, deploying continuous exponentials in discrete-time software simulations introduces severe computational bottlenecks and memory overhead. To resolve this, Phase III implements a pragmatic, hardware-optimized "Coincidence Detector" variant of STDP.

Rather than calculating exact exponential time-deltas, the update rule acts as a vectorized boolean filter based strictly on the firing time of the winning neuron ($t_"win"$): ]

#box-text()[
*Long-Term Potentiation (LTP):* Any afferent synapse that delivered a spike before or exactly at the decision time ($t_i \le t_"win"$) is deemed causal to the victory. Its weight is incremented by a fixed magnitude ($+A_+$).
*Long-Term Depression (LTD):* Any afferent synapse that spiked late ($t_i > t_"win"$) or failed to spike at all is classified as irrelevant or background noise. Its weight is decremented by a fixed magnitude ($-A_-$).
]

#serif-text()[ This approach mirrors the approximations used in digital neuromorphic ASICs, replacing transcendental operations with simple additive logical masks. All weights are subsequently clamped to a physical hardware range ($W_"min", W_"max"$) to prevent sign-flipping or unbounded integer overflow. ]

#figure(
  kind: "algo",
  caption: [The Vectorized STDP Weight Update Function],
  supplement: [Algorithm],
  mono-text(pseudocode-list(hooks: .5em, indentation: 1em, booktabs: true)[
    applyVectorizedSTDP(T_"pre", t_"win", W, A_+, A_-, W_"max") -> W_"new": #h(1fr)
      + // T_pre is the vector of all input spike times
      + // t_win is the exact time the winning neuron fired
      +
      + // Create boolean masks (Vectorized Coincidence)
      + $"mask"_"ltp" arrow.l (T_"pre" <= t_"win") and (T_"pre" != infinity)$
      + $"mask"_"ltd" arrow.l (T_"pre" > t_"win") or (T_"pre" == infinity)$
      +
      + // Apply flat constants via masks
      + $W["mask"_"ltp"] arrow.l W["mask"_"ltp"] + A_+$
      + $W["mask"_"ltd"] arrow.l W["mask"_"ltd"] - A_-$
      +
      + // Enforce physical hardware constraints
      + $W arrow.l max(0.0, min(W, W_"max"))$
      +
      + return $W$
  ])
) <a.vectorizedstdp>

#v(1em)
=== Competitive Specialization (Post-Hoc Hard-WTA)

#serif-text()[ If coincidence detection is applied to all neurons simultaneously, the network suffers from "catastrophic averaging"---all neurons attempt to learn the most dominant spatial feature, resulting in identical, redundant weights rather than distinct feature clustering.

To force specialization without disrupting the continuous integration dynamics of the forward pass, the network enforces strict Hard @wta dynamics retroactively as a learning mask.

Upon the conclusion of the saccade, the @stdp algorithm evaluates the precise timestamp of every spike. Only the single mathematically absolute first neuron to fire ($t_"min"$) is granted permission to undergo synaptic plasticity. All competing neurons are retroactively vetoed, and their weights remain frozen for that iteration. By applying WTA post-hoc, the network isolates the learning updates without introducing chaotic voltage fluctuations during the forward inference pass. If one neuron claims a specific structural pattern (e.g., a diagonal stroke), its peers are forced to adapt to secondary patterns to win future competitions. ]

#figure(
  kind: "algo",
  caption: [Post-Hoc Winner-Takes-All (WTA) Selection],
  supplement: [Algorithm],
  mono-text(pseudocode-list(hooks: .5em, indentation: 1em, booktabs: true)[
    applyPostHocWTA($T_"post"$) -> ($n_"winner"$, $t_"win"$): #h(1fr)
      + // T_post is the array of spike times for all hidden neurons
      + // Unfired neurons are represented by infinity
      +
      + // Find the absolute earliest spike time
      + $t_"win" arrow.l min(T_"post")$
      +
      + if $t_"win" == infinity$:
        + return $("null", infinity)$ // DNF: No neurons fired
      +
      + // Identify all neurons that fired at the exact winning tick
      + $N_"tied" arrow.l "findIndices"(T_"post" == t_"win")$
      +
      + // Tie-breaker: Randomly select one neuron to be the sole winner
      + $n_"winner" arrow.l "randomChoice"(N_"tied")$
      +
      + // All other neurons are retroactively vetoed from learning
      + return $(n_"winner", t_"win")$
  ])
)

#v(1em)
=== Homeostatic Threshold Adaptation

#serif-text()[ Relying strictly on WTA competition introduces a secondary risk: "dead" neurons. A small subset of hyper-responsive neurons might win every competition, preventing the rest of the layer from ever learning. To counteract this and ensure a distributed feature representation, the network employs a strict Homeostatic Threshold Adaptation loop mimicking the biological mechanisms detailed in @s.homeostatic.

Following every image presentation, the system regulates both the firing thresholds and the synaptic distributions of the hidden layer: ]

#box-text()[
*Global Threshold Decay:* The adaptive thresholds ($theta_"adaptive"$) of all hidden neurons are multiplied by a decay factor ($theta_"decay" = 0.90$). This gradually makes quiescent, "dead" neurons more sensitive over time.
*Winner Penalty:* The single neuron that won the WTA competition receives a massive additive penalty to its adaptive threshold ($theta_"plus" = 600.0$). This physically prevents it from dominating subsequent saccades, ensuring it only fires again when its highly specialized feature is prominently displayed.
*Synaptic Normalization:* The synaptic weights of the winning neuron are scaled to maintain a specific target $L_1$ norm ($K_"target"$). This ensures that the total synaptic drive of the neuron remains bounded, regardless of how many LTP updates it receives.
]

#serif-text()[ To execute unsupervised feature extraction on the MNIST dataset, the @stdp rule is embedded within the saccade simulation loop. The network is initialized with randomized synaptic weights $W tilde cal(U)(0, 1)$.

During the training phase, an image is presented, and the network executes a forward pass. To maintain network stability and prevent a single neuron from dominating the receptive field, we pair the @a.vectorizedstdp rule with Hard @wta dynamics and Homeostatic Threshold Adaptation. Only the single earliest firing neuron is permitted to learn. Following an @a.vectorizedstdp update, the adaptive thresholds of all hidden neurons naturally decay ($theta_"decay" = 0.90$). However, the single winning neuron receives a massive threshold penalty ($theta_"plus" = 600.0$), mathematically suppressing it in future iterations and forcing competing neurons to specialize in distinct, non-overlapping geometric primitives. At the conclusion of the 64-tick saccade, the simulator halts, compares the timestamp arrays of the hidden and output layers, and computes the @stdp weight updates in parallel before the next image is presented. ]

#v(2em)
== Evaluation Metrics

#serif-text()[ To rigorously validate the proposed @snn, the evaluation framework must bridge two distinct domains: classification effectiveness (decoding accuracy) and computational efficiency (hardware-agnostic resource usage). Because the networks undergo both supervised transfer (Phase II) and unsupervised native learning (Phase III), specialized metrics are required to capture the evolution of the internal representations. ]

#v(1em)
=== Effectiveness and Classification Performance

#serif-text()[ For Phase II (Zero-Shot Weight Transfer), performance is measured against the standard 10,000-image MNIST test set. Since the labels are pre-defined by the source ANN, effectiveness is quantified using standard statistical tools: ]

#box-text()[
*Top-1 Accuracy:* The percentage of images where the first output neuron to fire via the @wta mechanism matches the ground-truth label.

*Confusion Matrices:* Utilized to visualize the classification distribution and identify specific morphological overlaps (e.g., '4' vs. '9'). These matrices help determine if certain geometric features are disproportionately misclassified after the translation from static weights to discrete temporal decoding.
]

#serif-text()[ Evaluating Phase III (Unsupervised STDP) requires a shift in methodology. Because the network learns without labels, output neurons do not inherently map to categorical digits; they map to geometric clusters. To generate a comparable accuracy metric, we employ a *Post-Hoc Labeling* strategy: ]

#box-text()[
*Freezing:* Synaptic plasticity is disabled (learning rate set to zero) after the STDP training phase to prevent further weight drift.

*Assignment:* A subset of the labeled training data is passed through the network. Each output neuron is permanently assigned the label of the digit class that most frequently triggered it to fire.

*Testing:* The standard 10,000-image test set is passed through the "labeled" network to calculate the final Top-1 accuracy.
]

#v(1em)
=== Computational Efficiency (Hardware Proxies)

#serif-text()[ While the primary goal of neuromorphic engineering is immense energy reduction, evaluating true physical power draw ($J/"inference"$) is restricted by the use of PyTorch-based software simulators running on standard GPUs. Because these platforms incur heavy overhead simulating temporal event loops on von Neumann hardware, we abandon direct power measurements in favor of hardware-agnostic proxy metrics universally recognized in SNN literature: ]

#v(1em)
#mini-header()[Sparsity and Synaptic Operations (SyOPs)]

#serif-text()[ In a standard @ann, every forward pass requires a fixed number of @mac operations. In an @snn, computation is driven by discrete events. Because @if neurons do not require multiplication (a spike simply triggers the addition of its weight to the post-synaptic potential), @mac:pl are replaced by simpler @syops.

The total computational cost for a single 64-tick @saccade is estimated as: ]

#figure(
  kind: "eq",
  supplement: [Equation],
  caption: [SNN Operational Cost Proxy],
  [ $ "Total SyOPs" = sum_{l=1}^{L} N_"spikes"^{(l)} dot F_"out"^{(l)} $ ]
)

#serif-text()[ Where $N_"spikes"$ is the total spikes emitted in layer $l$ and $F_"out"$ is the fan-out of the neurons. By comparing the SNN SyOPs against the fixed MAC count of the baseline ANN, we derive a theoretical energy efficiency ratio. ]

#v(1em)
#mini-header()[Temporal Latency Metrics]

#serif-text()[ To evaluate the efficacy of the Time-to-First-Spike (TTFS) encoding, we measure the *Time-to-Decision Latency*. This is defined as the exact simulation tick $t in [0, 64)$ at which the WTA output layer reaches a decision. A lower average latency indicates a superior temporal decoder that successfully prioritizes salient information, allowing the system to theoretically power down early in the simulation window. ]

#v(2em)
== Experiment Setup and Evaluation Phases

#serif-text()[ To systematically evaluate the proposed @snn architectures, the experimental framework is structured as a progressive pipeline, moving from isolated mathematical unit tests to full-scale visual classification. This approach isolates three critical neuromorphic variables: *temporal integration dynamics*, *parameter robustness during quantization*, and *unsupervised feature emergence*. ]

=== SNN Simulation Engine

#serif-text()[ The primary testbed is a custom-built, discrete-time SNN simulator implemented in PyTorch. Unlike standard Artificial Neural Networks (ANNs) which process data in a single static "glance," the @snn processes data through a dynamic temporal @saccade lasting $T_"max" = 64$ ticks, requiring state variables to be tracked across the temporal dimension. ]

#figure(
  include("figures/softwarearch.typ"),
  caption: [Software architecture and data flow of the @snn simulator.]
)

#serif-text()[ To ensure reproducibility across the network-scale evaluations (Phases II and III), the baseline hardware-proxy parameters are standardized as follows. *(Note: Phase I overrides $V_"th"$ to perform targeted threshold sweeps).* ]

#table(
  columns: (1fr, 1fr, 2fr),
  inset: 10pt,
  align: horizon,
  [*Parameter*], [*Value*], [*Context*],
  [$T_"max"$], [64], [Total simulation ticks per image],
  [Input Cap], [32], [Maximum delay for lowest intensity pixel],
  [$V_"th_C"$], [600.0 / 180.0], [Baseline thresholds for Model C (180.0 utilized in Phase 3 inference)],
  [Homeostasis], [$theta_"decay" = 0.90, theta_"plus" = 600.0$], [Adaptive threshold limits applied during STDP],
  [Optimizer], [Adam @kingma_adam_2017], [Source optimizer for Phase II Baseline],
)

#v(1em)
=== Evaluation Phases

#serif-text()[ The experiment is executed in three logical phases, building in complexity from an isolated single neuron to a fully self-organizing network. ]

#mini-header()[Phase I: Temporal Dynamics and Threshold Sweeps]
#serif-text()[
  This phase serves as a functional unit test for the isolated neuron models described in @s.decoding. Rather than utilizing a single static threshold, the models are subjected to synthetic spike trains across a spectrum of threshold bounds:
]
#box-text()[
*Saturation Regime (Low Threshold):* Evaluates susceptibility to false positives when the total incoming weight vastly exceeds the threshold.

*Critical Regime (Balanced Threshold):* Evaluates sequence decoding when the threshold strictly requires nearly all pattern weights to coordinate.

*Deficit Regime (High Threshold):* Evaluates resilience when the base pattern is insufficient, testing if momentum or leak mechanics can leverage noise to force a spike.
]

#mini-header()[Phase II: The ANN Baseline and Zero-Shot Transfer]
#serif-text()[ Phase II benchmarks the "SNN-as-Accelerator" hypothesis. First, an Artificial Neural Network (ANN) baseline is established (using a standard $784 -> 128 -> 10$ MLP without biases) to provide an "ideal" accuracy ceiling. Learned $"FP32"$ weights are then strictly quantized to an $"INT8"$ range and transferred directly into the SNN. We measure the *Accuracy Decay Rate* ($Delta_"Acc" = "Acc"_"ANN" - "Acc"_"SNN"$) to quantify the information lost during this static-to-temporal translation. ]

#mini-header()[Phase III: Native Unsupervised Learning]
#serif-text()[ The final phase evaluates neuromorphic self-organization. Discarding all pre-trained weights, the network is initialized randomly and exposed to the MNIST dataset strictly via unsupervised @stdp. ]

#box-text()[
*Lateral Inhibition:* A @wta mechanism is enforced; the first output neuron to fire suppresses all competitors, driving the network toward distinct feature clustering.

*Homeostasis:* Synaptic weights are normalized after each stimulus ($sum w_j = K$) to prevent single-neuron dominance and ensure a distributed feature representation across the hidden layer.
]

#serif-text()[ Because @stdp is fundamentally unsupervised, the @snn output neurons organically form distinct feature clusters rather than mapping to pre-defined human labels (0-9). To quantify classification accuracy, a post-hoc assignment protocol is utilized. Following the unsupervised training phase, a distinct subset of the training data is passed through the frozen network. A frequency matrix $M$ of size $N_"out" times 10$ tracks the ground-truth label of each image that causes output neuron $j$ to win the @wta competition. The assigned label for neuron $j$ is then defined as $L_j = "argmax"(M_j)$. ]

#pagebreak()

= Results <c.results>

#serif-text()[ This chapter presents the empirical data gathered from the three evaluation phases. The results progress from isolated single-neuron temporal dynamics to full-network zero-shot translation, and finally to unsupervised self-organization. ]

#v(2em)
== Phase I: Temporal Dynamics Of Neuron Models <s.res_phase1>

#serif-text()[ Phase I evaluated the isolated response of the four neuron models to permuted temporal patterns (Concordant vs. Discordant) across three distinct threshold constraints: Saturation (Low), Critical (Balanced), and Deficit (High). The base spatial weight sum for all trials was $300.0$. The resulting output spike latencies are summarized in @tbl:phase1_results and visualized in @fig:phase1_composite. ]

#figure(
  table(
    columns: (1.8fr, 1fr, 1fr, 1fr, 1fr),
    inset: 8pt,
    align: center,
    [*Model*], [*Regime*], [*Threshold ($theta$)*], [*Concordant Spike ($t$)*], [*Discordant Spike ($t$)*],
    [Model A (Simple IF)], [Saturation], [150.0], [6], [14],
    [Model A (Simple IF)], [Critical], [290.0], [18], [18],
    [Model A (Simple IF)], [Deficit], [310.0], [DNF], [DNF],
    [Model B (Standard LIF)], [Saturation], [140.0], [6], [14],
    [Model B (Standard LIF)], [Critical], [240.0], [DNF], [18],
    [Model B (Standard LIF)], [Deficit], [310.0], [DNF], [DNF],
    [Model C (Linear Ramp)], [Saturation], [500.0], [4], [9],
    [Model C (Linear Ramp)], [Critical], [1800.0], [8], [17],
    [Model C (Linear Ramp)], [Deficit], [2500.0], [DNF], [DNF],
    [Model D (State Discount)], [Saturation], [100.0], [6], [14],
    [Model D (State Discount)], [Critical], [220.0], [10], [DNF],
    [Model D (State Discount)], [Deficit], [310.0], [DNF], [DNF],
  ),
  caption: [Output spike latencies across varying threshold regimes. DNF indicates the neuron Did Not Fire within the $T_"max" = 64$ saccade window.],
  kind: "table",
  supplement: [Table]
) <tbl:phase1_results>

#serif-text()[ Under the Low (Saturation) threshold regime, all models successfully fired early. Predictably, when the strongest inputs arrived first (Concordant), all models reached the threshold significantly faster than when the strongest inputs arrived last.

However, when thresholds were critically tuned to perfectly match the total spatial weight of the spike train (Balanced), model behaviors diverged significantly. Model A (Simple IF) demonstrated complete temporal blindness, firing at $t=18$ regardless of the input order. Model B (Standard LIF) acted as a late-spike coincidence detector; the early spikes decayed before the threshold could be reached, meaning it only fired when the strongest input came last ($t=18$). Conversely, both Model C (Linear Ramp) and Model D (State Discount) demonstrated strong preference for early stimuli, firing faster when the strongest inputs arrived first.

Finally, under the Deficit regime (where the threshold was set strictly higher than the total weight of the stimuli), all architectures failed to fire. For Model C specifically, the temporal momentum was insufficient to overcome the high threshold before the 10-tick coincidence timer expired, aggressively zeroing the state variables. ]

#figure(
  image("figures/phase1_composite_sweep.png", width: 100%),
  caption: [Input spike distributions and resulting temporal dynamics across the three threshold regimes.],
) <fig:phase1_composite>

#box-text()[ Because Phase II and Phase III require an architecture capable of processing variable input densities while maintaining strict feature discriminability, the selected neuron model must demonstrate distinct temporal separability under critical constraints. As demonstrated in @tbl:phase1_results, Model C (Current-Accumulating Linear Ramp) was the only architecture to consistently fire and maintain a significant latency gap ($t=8$ vs. $t=17$) under the Critical regime. By successfully leveraging temporal momentum to decode the spike order, Model C was utilized as the standardized spiking architecture for the network-scale evaluations in the subsequent phases. ]


#v(2em)
== Phase II: Zero-Shot Weight Transfer <s.res_phase2>

#serif-text()[ Phase II evaluated the accuracy decay incurred when translating a conventionally trained Artificial Neural Network (ANN) to the selected Spiking Neural Network (Model C) without any intermediate retraining.

The Phase 0 baseline ANN ($784 arrow 128 arrow 10$ MLP) was trained for 20 epochs using the Adam optimizer. To satisfy the strict mapping requirements of the SNN, the network was trained with all bias terms disabled ($b=0$). The baseline model achieved a maximum test accuracy of 98.40% on the standard MNIST test set, establishing a strong, expected baseline for a fully connected topology.

To rigorously isolate the sources of information loss during the zero-shot transfer, the SNN was loaded with the exact FP32 weights from the ANN. This isolates the *Temporal Penalty*—the loss incurred strictly by moving from a static dot-product activation to a discrete, time-to-first-spike (TTFS) integration. The SNN utilized the Model C (Linear Ramp) architecture over a $T_"max" = 64$ saccade window. The results are summarized in @tbl:phase2_accuracy. ]

#figure(
  table(
    columns: (1.2fr, 1fr),
    inset: 10pt,
    align: center,
    [*Model Configuration*], [*Accuracy (%)*],
    [ANN (Static FP32, Bias=False)], [98.40%],
    [SNN (Spiking Model C, FP32)], [94.50%],
  ),
  caption: [Performance degradation breakdown. The Temporal Penalty isolates the loss of TTFS encoding without any weight alteration.],
  kind: "table",
  supplement: [Table]
) <tbl:phase2_accuracy>

#serif-text()[ The zero-shot translation achieved a remarkable 94.50% accuracy, demonstrating that the momentum-based decoding of Model C successfully preserves the spatial hierarchies learned by the ANN. @fig:snn_confusion visualizes the classification distribution of the SNN. The network maintained strong structural clustering, effectively utilizing the algorithmic Winner-Takes-All (WTA) mechanism to finalize classifications. ]

#figure(
  image("figures/phase2_cumulative_accuracy.png", width: 100%),
  caption: [Cumulative classification accuracy over the 64-tick temporal window. Accuracy follows a sigmoid-like "S-curve," showing that the majority of correct classifications are finalized early in the saccade.],
) <fig:res_cumulative_acc>

#serif-text()[
  As visualized in @fig:res_cumulative_acc, the network exhibits a rapid "S-curve" profile in its decision-making process. Accuracy remains at 0% for the first three ticks as the initial wavefront of spikes propagates through the hidden layer. Between ticks 5 and 15, there is a violent surge in accuracy, capturing over 80% of the test set.

  This profile is a direct result of the @ttfs encoding and Model C dynamics: the most salient (high-intensity) pixels arrive first, providing enough evidence for the momentum-based integration to cross the threshold for clear, high-contrast digits. The plateau observed after tick 20 indicates that further integration of lower-intensity "background" pixels provides diminishing returns for accuracy, as these spikes represent morphological details that are either redundant or noisy. This validates the "early-exit" hypothesis, proving that the network achieves near-peak performance while the majority of the temporal window remains technically unprocessed.
]

#figure(
  image("figures/phase2_confusion_matrix.png", width: 100%),
  caption: [Confusion matrix for the zero-shot SNN on the MNIST test set. The strong diagonal indicates the temporal integration preserved the original ANN decision boundaries.],
) <fig:snn_confusion>

#serif-text()[ To evaluate the computational efficiency of the architecture, @tbl:phase2_efficiency compares the dense MAC operations of the baseline ANN against the sparse SynOps of the TTFS implementation. ]

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

#serif-text()[ The zero-shot SNN achieved a Mean Time-to-Decision of 8.4 ticks out of the maximum 64. By halting integration upon classification, the network processed an average of only 15,021 SynOps per image, yielding an 85.2% reduction in computational workload compared to the static ANN baseline. ]


#v(2em)
== Phase III: Unsupervised Native Learning

#serif-text()[ Phase III evaluated the capacity of the network to self-organize without gradient descent, relying purely on local Spike-Timing-Dependent Plasticity (STDP), hard WTA competition, and homeostatic threshold adaptation. After processing the unsupervised training set, the network achieved an overall accuracy of 44.3% (@tbl:phase3_accuracy). ]

#figure(
  table(
    columns: (1.2fr, 1fr),
    inset: 10pt,
    align: center,
    [*Model Configuration*], [*Accuracy (%)*],
    [SNN (Spiking Model C, STDP, WTA)], [44.3%],
  ),
  caption: [Top-1 classification accuracy after fully unsupervised STDP training.],
  kind: "table",
  supplement: [Table]
) <tbl:phase3_accuracy>

#figure(
  table(
    columns: (1fr, 1.2fr, 1fr, 1.2fr, 1.2fr),
    inset: 10pt,
    align: center,
    [*Architecture*], [*Metric Target*], [*Avg. Latency*], [*Avg. Operations / Image*],[*Compute Reduction*],
    [SNN (Spiking STDP)], [Sparse SynOps], [8.3 Ticks], [12,954 SynOps],[87.2%],
  ),
  caption: [Computational cost comparison for the unsupervised SNN. The self-organized network maintained the early-exit sparsity observed in Phase II.],
  kind: "table",
  supplement: [Table]
) <tbl:phase3_efficiency>

#serif-text()[ While a 44.3% accuracy represents a significant drop from the 98.40% baseline
achieved via supervised backpropagation, it definitively proves that the local, event-driven
@stdp rule successfully clustered the geometric features of the dataset without any global
error signal. Notably, the self-organized network maintained a computational profile
comparable to the weight-transferred architecture: an average decision latency of 8.3 ticks
and 12,954 @syops per image, yielding an 87.2% reduction in operations relative to the ANN
baseline (@tbl:phase3_efficiency). The marginal improvement in sparsity over Phase II (87.2%
vs. 85.2%) is consistent with the hypothesis that @stdp naturally depresses irrelevant
synapses, producing a slightly sparser weight distribution than the directly transferred
FP32 weights. This confirms that the early-exit temporal behavior is a structural property
of the @ttfs encoding itself, preserved independently of whether the weights were learned
via gradient descent or local plasticity. ]

#figure(
  image("figures/phase3_02_receptive_after.png", width: 100%),
  caption: [Hidden layer receptive fields ($W_1$) after unsupervised STDP training. The hard WTA competition forces Vector Quantization, resulting in recognizable, holistic templates of entire digits.],
) <fig:receptive_stdp>

#figure(
  image("figures/phase3_03_baseline_fcn_weights.png", width: 100%),
  caption: [Hidden layer receptive fields ($W_1$) extracted from the baseline ANN. Gradient descent optimizes for a Distributed Representation, resulting in abstract, non-visual basis functions.],
) <fig:receptive_ann>

#serif-text()[ While template matching provides a robust mechanism for clustering (as evidenced by the strong diagonal in the confusion matrix in @fig:phase3_confusion), it fundamentally struggles with morphological overlaps. For instance, an input '9' will violently trigger a template for a '4' due to overlapping pixel energy. Because the SNN lacks negative weights to penalize specific non-matching regions, it surrenders to the template with the highest overall pixel correlation, capping the theoretical accuracy ceiling for this single-layer, unsupervised topology. ]

#figure(
  image("figures/phase3_04_confusion_matrix.png", width: 100%),
  caption: [Confusion matrix for the unsupervised STDP network. While primary classes are successfully clustered, the network struggles with overlapping topologies (e.g., classifying '4's as '9's) due to the holistic nature of the learned templates.],
) <fig:phase3_confusion>

#serif-text()[ Inspection of the confusion matrix in @fig:phase3_confusion reveals that
performance is highly non-uniform across digit classes. Visually simple, topologically
distinct digits---particularly '0' and '1'---are classified with high confidence, as their
silhouette templates are geometrically orthogonal to all other digit classes and present
minimal risk of pixel-correlation overlap. Classification failures are concentrated among
morphologically similar digit pairs. The '4'/'9' confusion is the most pronounced, as both
digits share a significant region of overlapping pixel energy in the upper portion of the
image frame. Similarly, '3', '5', and '8' exhibit mutual confusion due to their shared
curved stroke structure. Because the @snn relies purely on positive pixel correlation and
lacks the inhibitory negative weights that allow the ANN to penalize non-matching regions,
it cannot suppress the partial template matches that drive these errors. This is the
defining limitation of single-layer @wta with additive @stdp: the network's 128 neurons
are sufficient to cover the ten primary digit archetypes, but insufficient to simultaneously
represent the intra-class morphological variance required to resolve ambiguous cases. ]

#pagebreak()

= Discussion <discussion>

#serif-text()[ While the experimental results validate the core principles of @ttfs encoding and spiking integration, extrapolating these methods from benchmarks to real-world deployment reveals significant engineering bottlenecks. This chapter critically analyzes the limitations observed during the implementation phase, specifically regarding data encoding, spatial representation, architectural translation, and hardware constraints. ]


#v(2em)
== Evaluation Of Neuron Models <s.disc_phase1>

#serif-text()[ The empirical observations from the Phase I threshold sweeps reveal a fundamental tension between traditional biological neuron models and the specific requirements of Time-To-First-Spike (TTFS) decoding. By systematically shifting the threshold bounds, the isolated unit tests exposed severe vulnerabilities in standard integration strategies, while validating the proposed momentum-based and state-discounting architectures.

Under Saturation (Low Threshold) conditions, all models successfully fired earlier for the concordant pattern. However, as hypothesized, this is largely an illusion of magnitude rather than true temporal decoding. The Simple IF (Model A) and LIF (Model B) models simply accumulated the heavily weighted early spikes and crossed the lowered threshold prematurely.

This vulnerability becomes glaringly apparent in the Critical (Balanced) regime. When forced to accumulate the entire sequence, Model A completely failed to differentiate order, firing simultaneously for both patterns. More critically, Model B (LIF) demonstrated a catastrophic misalignment with TTFS principles. At a balanced threshold, Model B registered a "Did Not Fire" (DNF) for the concordant pattern, but successfully fired for the discordant pattern.

This occurs due to the interaction between temporal density and the exponential leak. In the concordant pattern, the strongest spikes arrive early, but their accumulated potential decays significantly while waiting for the weaker, later spikes. Conversely, in the discordant pattern, early weak spikes establish a baseline potential, and the massive late-arriving spikes push the neuron over the threshold immediately before they have a chance to decay. Consequently, standard LIF inherently favors "strongest-last" sequences, actively fighting the "strongest-first" priority of TTFS encoding. ]

#v(1em)
=== Selective Filtering vs. Unbounded Momentum

#serif-text()[ In contrast to the standard models, the custom architectures (Models C and D) exhibited highly desirable temporal dynamics for rank-order decoding.

At the Critical (Balanced) threshold, the State-Dependent Discount (Model D) achieved perfect selective filtering. It successfully fired for the concordant pattern while completely suppressing the discordant pattern. By discounting late-arriving spikes based on internal state rather than elapsed time, it ensures that a neuron only fires if the most critical information arrives exactly when the neuron is empty and highly receptive.

The Linear Ramp (Model C) demonstrated extreme robustness, being the only architecture capable of crossing the High (Deficit) threshold. By converting early spikes into compounding integration momentum, Model C can force a spike even when the raw spatial weight sum is theoretically insufficient. While this unbounded momentum guarantees high firing rates and robust temporal differentiation, it theoretically introduces a risk of network over-excitation. However, in the context of this architecture, this risk is safely mitigated by the strict enforcement of the saccade deadline ($T_"max" = 64$); the global reset prevents the linear ramp from diverging toward hardware saturation across multiple inferences. ]

#v(2em)
== Viability Of Weight Transfer

#serif-text()[ Phase II demonstrated that momentum-based @ttfs integration can recover
94.50% of the ANN's classification accuracy via zero-shot weight transfer, incurring a
temporal penalty of only 3.9 percentage points. This section examines the two primary
sources of that penalty and the structural constraints that would govern any attempt to
scale this methodology beyond a fully connected topology. ]

#v(1em)
=== Architectural Translation (ANN to SNN)

#serif-text()[ The zero-shot inference phase highlighted the frictions of translating classical architectures to spiking substrates. While the Fully Connected topology provided a transparent baseline, mapping state-of-the-art Convolutional Neural Networks (@cnn:pl) is decidedly not a one-to-one process.

Classical continuous networks heavily utilize negative weights, static biases, and mathematical operations like Max Pooling. Neuromorphic systems handle these concepts fundamentally differently. An @snn replaces mathematical pooling with dynamic lateral inhibition, and negative continuous weights must be modeled via discrete inhibitory spike trains. These architectural mismatches dictate that @snn:pl cannot simply act as "drop-in" replacements for deep learning models; they require network topologies natively designed for event-driven dynamics. ]


#v(1em)
=== The Sparsity Paradox and GPU Simulation Overhead

#serif-text()[ A core theoretical advantage of the proposed @ttfs network is its extreme spatial and temporal sparsity. By design, only a small fraction of neurons emit spikes, mathematically reducing the dense @mac operations of a standard @ann to a minimal set of @syops.

However, evaluating this sparse algorithm on conventional GPUs introduces a significant "Sparsity Paradox." Standard deep learning frameworks (such as PyTorch) and standard GPU architectures are heavily optimized for dense, contiguous matrix-matrix multiplications via SIMD (Single Instruction, Multiple Data) execution. In the @snn simulation loop, sparsity is enforced via boolean masking (multiplying inactive neuron outputs by zero). While this successfully zeroes out the potential, the GPU @alu:pl typically still execute the underlying floating-point multiplication cycle.

Consequently, the hardware does not naturally skip the computation for quiescent neurons unless specialized, unstructured sparse tensor kernels are deployed. When combined with the overhead of maintaining the temporal loop (the "saccade" ticks), the @snn simulator ironically consumes more absolute clock cycles and physical power on a GPU than the continuous @ann baseline.

This paradox highlights that true energy efficiency cannot be realized via matrix-masking on von Neumann hardware. Realizing the calculated SyOP savings requires deployment on native event-driven Neuromorphic ASICs (Application-Specific Integrated Circuits). These chips discard the matrix-multiplication paradigm entirely, utilizing @aer to asynchronously route data packets only when a spike physically occurs, reducing idle power draw to near zero. ]

#v(2em)
== Addressing Native Unsupervised Learning With STDP

#serif-text()[ Phase III demonstrated that a 128-neuron @ttfs network, trained exclusively
via local @stdp, successfully self-organizes into a functional classifier achieving 44.3%
top-1 accuracy on MNIST. While this result validates the core biological plausibility of
the learning rule, the gap relative to both the supervised ANN (98.40%) and the Diehl and
Cook rate-coded baseline (95% with 6,400 neurons) can be attributed to three interacting
architectural constraints, each of which represents a distinct engineering challenge for
scaling native neuromorphic learning. ]

#v(1em)
=== Plasticity Dynamics and Forgetting

#serif-text()[ A fundamental tension emerged between the network's learning velocity and
its stability during Phase III training. Unsupervised, local learning rules like @stdp
require aggressive hyperparameter tuning. If the plasticity rate is too high, the network
adapts quickly but suffers from rapid catastrophic forgetting---overwriting previously
learned geometric features when presented with novel stimuli. Conversely, if the plasticity
rate is too low, the network fails to converge on meaningful representations within an
acceptable time frame.

This instability directly suppresses classification accuracy: neurons that have partially
specialized toward one digit class can be corrupted by subsequent exposures to morphologically
similar digits, eroding the clean template boundaries visible in @fig:receptive_stdp.
Homeostatic threshold adaptation partially mitigates this by normalizing competitive firing
rates across the hidden layer, but it does not eliminate the fundamental tension between
rapid adaptation and long-term retention. Resolving this trade-off---for instance via
synaptic consolidation mechanisms or spike-timing-dependent metaplasticity---remains a
core open problem for native neuromorphic learning. ]

#v(1em)
=== Representing Space and Dimensionality

#serif-text()[ The second architectural constraint concerns spatial representation. In
the current implementation, absolute pixel intensity is mapped directly to spike latency,
which provides adequate discriminability for the centered, high-contrast digits of MNIST.
However, encoding precise spatial coordinates within a pure @ttfs scheme proves inherently
difficult. In biological visual and motor cortices, spatial locations are represented via
orthogonal population codes, where specific neuron populations activate to signal direction
or position. Crucially, these biological populations largely utilize rate coding; the
certainty of a spatial position translates naturally into a higher firing frequency, a
representation that has no direct equivalent in single-spike @ttfs.

This limitation manifests directly in the template-matching behavior observed in
@fig:receptive_stdp. Because the network cannot encode the relative spatial position of
a stroke independently of its intensity, neurons cannot learn localized, translation-invariant
features. Instead, they converge on holistic digit silhouettes that are inherently brittle
to spatial shifts. A convolutional spiking architecture with local receptive fields would
partially address this by binding spatial position into the kernel structure rather than
requiring the neuron to encode it temporally. ]

#v(1em)
=== Global WTA vs. Local Lateral Inhibition

#serif-text()[ The third constraint is the competitive mechanism itself. In the current
@fcn implementation, competitive specialization is enforced via a Global @wta mechanism:
the first output neuron to fire retroactively vetoes all other neurons in the hidden layer.
While computationally efficient for isolated, centered digits like MNIST, this global
suppression is the primary driver of the accuracy ceiling observed in Phase III.

If an input image contains multiple distinct spatial features---a circle in the upper
region and a vertical stroke in the lower region---a global @wta allows the network to
respond to only the single most salient feature, assigning the entire image to one
neuron's template and discarding the complementary spatial evidence. This is directly
responsible for the inter-class confusions observed in @fig:phase3_confusion: a '9' is
suppressed in favor of a '4' template because the upper-loop feature fires first and
claims the @wta before the descending stroke can contribute disambiguating evidence.

Biological nervous systems address this via Local Lateral Inhibition, where an active
neuron suppresses only its immediate spatial neighbors rather than the entire population.
Transitioning to a spiking convolutional architecture would necessitate replacing the
global post-hoc mask with continuous, spatially-bounded inhibition routed across feature
channels at identical spatial coordinates. This would allow neurons at distant spatial
locations to fire independently, enabling the network to simultaneously represent multiple
features of a single image and substantially increase the representational ceiling of
unsupervised @ttfs learning. ]



#v(2em)
== Future Work <s.futurework>

#serif-text()[ The findings and limitations discussed in this thesis present several promising avenues for future research in neuromorphic engineering, ranging from algorithmic enhancements to physical hardware deployment. ]

#v(1em)
=== Event-Based Datasets and Surrogate Gradients

#serif-text()[ Transitioning the experimental framework from static images (MNIST) to native temporal datasets, such as Neuromorphic-MNIST (N-MNIST) or DVS Gesture datasets, is essential to fully exploit the asynchronous dynamics of the evaluated neuron models. Furthermore, while this work focused on direct weight transfer and native @stdp, future implementations should evaluate the efficacy of Surrogate Gradient Descent. This approach promises to combine the optimization power of backpropagation with the inference efficiency of spiking dynamics. ]

#v(1em)
=== Convolutional Spiking Architectures

#serif-text()[ Replacing the fully connected topology with a spiking Convolutional Neural Network (CNN) utilizing local receptive fields and spatially bounded lateral inhibition represents another critical next step. This architectural shift would directly address the template-matching limitations identified in Phase III by introducing translation invariance and enabling multi-feature representation within a single saccade window. ]

#v(1em)
=== Mitigation via Synaptic Pruning

#serif-text()[ While dynamic activation sparsity struggles on traditional GPUs, structural sparsity offers a highly viable software-level mitigation. Future iterations of this work could introduce aggressive synaptic pruning following the unsupervised @stdp learning phase. Because @stdp naturally depresses irrelevant synapses toward zero, a thresholding function could permanently sever these connections, converting dense weight matrices ($W^((1))$, $W^((2))$) into highly sparse structures. By utilizing block-sparse tensor formats (e.g., Compressed Sparse Row), both software simulators and specialized sparse-accelerator chips can mathematically bypass zero-weights. This yields physical reductions in memory bandwidth and computation even prior to transitioning to pure neuromorphic silicon. ]

#v(1em)
=== Computing Contrast in Images

#serif-text()[ In the engineered test environment, absolute pixel luminance was mapped directly to spike latency. While this approach yields adequate performance for highly controlled, isolated datasets like MNIST, absolute luminance is a notoriously brittle feature for real-world computer vision. Robust biological and artificial vision systems rely instead on local contrast---the relative intensity difference between adjacent pixels---which remains invariant under shifts in global illumination.

Attempting to compute true, normalized contrast natively within a @ttfs spiking network presents a severe temporal bottleneck. To accurately assess relative darkness in a purely temporal code, downstream neurons must wait for the slowest (darkest) signals to arrive before a contrast ratio can be assessed. This effectively nullifies the high-speed, priority-queue advantages of @ttfs encoding. The optimal mitigation is to offload this computation to the sensory periphery. Dedicated neuromorphic sensors, such as Dynamic Vision Sensors (DVS), natively output logarithmic intensity differences. Because a difference in log-space corresponds mathematically to a true contrast ratio independent of absolute luminance, passing this pre-encoded contrast data directly into the @ttfs network avoids the temporal delay problem entirely, substantially improving robustness on naturalistic image datasets. ]

#v(1em)
=== The Physical Hardware Gap

#serif-text()[ Ultimately, the theoretical energy efficiency of neuromorphic algorithms is bounded by physical hardware constraints. Porting the verified @ttfs algorithms and current-accumulating neuron models from GPU simulation onto dedicated neuromorphic silicon (e.g., Intel Loihi) is necessary to empirically measure true joule-per-inference energy consumption against classical von Neumann baselines.

Currently, the connection density of the biological mammalian cortex vastly exceeds the routing capabilities of modern CMOS (Complementary Metal-Oxide-Semiconductor) fabrication. Achieving true biological efficiency will require novel materials. Until non-volatile memory technologies---such as memristors and spintronics, which offer the ability to physically colocate extreme-density analog weights with logic gates---become commercially viable, massively parallel, asynchronous digital ASICs remain the most tractable near-term substrate. These standard CMOS designs serve as a vital transitional bridge to realizing the energy efficiency demonstrated algorithmically in this thesis. ]

#v(2em)
== Closing Remarks

#serif-text()[ Taken together, the three experimental phases of this thesis trace a coherent arc from isolated neuron dynamics to self-organizing network behavior. Phase I established that standard integrate-and-fire models are fundamentally misaligned with @ttfs decoding, and that momentum-based integration provides a principled and necessary correction. Phase II demonstrated that this correction is sufficient to recover 94.50% of Artificial Neural Network (ANN) accuracy via zero-shot weight transfer, yielding an 85.2% reduction in computational operations. Finally, Phase III showed that the same architecture, trained entirely without supervision, achieves 44.3% accuracy through purely local plasticity---a result whose limitations are architectural rather than algorithmic, largely attributable to global @wta dynamics, the absence of spatial invariance, and the representational constraints of single-spike encoding.

The central finding that unifies all three phases is that the theoretical efficiency advantages of neuromorphic algorithms are real and measurable at the algorithmic level. However, they are presently obscured by the mismatch between event-driven computation and the von Neumann hardware on which they are simulated. This gap between algorithmic efficiency and physical realizability is not a flaw in the neuromorphic approach; rather, it is the defining engineering challenge of the field, and one that is actively being closed by the ongoing development of native neuromorphic substrates. ]

#pagebreak()

= Conclusion <s.conclusion>

#serif-text()[
The central premise of this thesis is that the computational inefficiency of modern
Deep Learning is a paradigmatic issue rather than an engineering one. By simulating
neuromorphic algorithms on conventional CPU and GPU hardware, this work has
demonstrated that sparse, event-driven computation and biologically plausible local
learning are implementable, measurable, and capable of producing meaningful results
on real classification tasks---even in the absence of the native hardware they are
ultimately designed for.

#v(1em)
== Addressing the Objectives

#v(0.5em)
=== Sparse Efficient Computing

The @ttfs encoding scheme successfully compressed MNIST images into a sparse,
priority-driven temporal queue, achieving an 85.2% reduction in synaptic operations
relative to the dense ANN baseline. The network reached correct classifications at
an average latency of 8.4 ticks within a 64-tick saccade window, processing less
than 14% of the available temporal budget before halting. These results confirm that
sparse event-driven algorithms measurably reduce theoretical computational workload
on standard hardware.

However, a critical finding is that this algorithmic sparsity does not translate to
physical energy savings when simulated on von Neumann processors. GPU architectures
continue to execute floating-point operations for quiescent neurons masked to zero---a
"Sparsity Paradox" that underscores the dependence of neuromorphic efficiency on
native event-driven substrates. The theoretical savings are real; their physical
realization awaits deployment on dedicated neuromorphic silicon.

#v(0.5em)
=== Neuron Model Evaluation

Systematic threshold sweeps across four neuron models established that standard
Integrate-and-Fire and Leaky Integrate-and-Fire architectures are fundamentally
misaligned with @ttfs decoding. The exponential leak of the LIF model actively
penalizes early, high-salience spikes, causing it to implicitly favor discordant
sequences---the inverse of @ttfs priority ordering. The proposed Current-Accumulating
Linear Ramp (Model C) was the only architecture to maintain robust temporal
separability under critical threshold constraints, and was selected as the basis
for all subsequent network-scale evaluations.

#v(0.5em)
=== Inference Via Weight Transfer

Zero-shot transfer of FP32 weights from a conventionally trained ANN achieved
94.50% top-1 accuracy on MNIST, incurring a temporal penalty of 3.9 percentage
points relative to the 98.40% ANN baseline. This isolates the cost of transitioning
from static continuous activations to momentum-based spike integration, and
demonstrates that the spatial hierarchies learned by gradient descent are recoverable
from a purely temporal spike ordering without any intermediate retraining or
fine-tuning.

#v(0.5em)
=== Native Unsupervised Learning

Trained exclusively via local @stdp, the network self-organized into a functional
classifier achieving 44.3% top-1 accuracy without labeled data or global error
signals. The learned representations converged on holistic digit templates,
confirming that local plasticity rules can extract statistically meaningful geometric
features from raw visual input. The performance gap relative to the supervised
baseline is attributable to identifiable architectural constraints---global @wta
competition, the absence of translation invariance, and the representational limits
of single-spike encoding in a 128-neuron population---rather than a fundamental
failure of the @stdp rule itself.

#v(1em)
== Broader Significance

These results make two contributions to the broader case for neuromorphic computing.
First, they demonstrate that temporal coding and local learning rules are viable
algorithmic primitives on standard hardware. The fact that meaningful classification
emerges from purely local, event-driven computation---even under the severe constraints
of a small network on a controlled benchmark---confirms that the paradigmatic shift
away from dense synchronous processing is achievable in principle.

Second, this work precisely characterizes where the limits of that paradigm currently
lie. The Sparsity Paradox, the template-matching ceiling of global @wta, and the
instability of unsupervised @stdp under small population sizes are not incidental
implementation details---they are the concrete engineering problems the neuromorphic
field must resolve to move from algorithmic demonstration to deployed efficiency.
Identifying and quantifying these boundaries on a controlled benchmark is a necessary
precondition for the more ambitious architectures that follow.

The trajectory is clear. As native neuromorphic substrates mature---from digital ASICs
toward analog memristive crossbars---the algorithmic foundations validated here become
the blueprints for systems that physically co-locate memory and computation, route data
only when spikes occur, and learn continuously from unlabeled sensory streams. The
human brain operates on 20 watts. Closing the gap between that biological fact and
the current engineering reality is not a question of whether the principles are sound,
but of whether the hardware can yet embody them. ]

#pagebreak()

#set text(weight: "medium", size: 10pt)
#bibliography("references.bib")
