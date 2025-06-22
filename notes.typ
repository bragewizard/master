/* Configurations */

#import "@preview/lovelace:0.3.0": *
#set text(font: "Source Serif 4 18pt")
#show math.equation : set text(font:"TeX Gyre Schola Math")
#show heading: set text(font:"Source Serif 4",weight: "black", style: "italic")
#set page(numbering: "1")
#set par(justify: true)
#set list(marker: sym.hexa.filled)

#text("Sparse Spiking Neural Networks", size: 30pt, weight: "black")

#datetime.today().display()

#v(1em)

= Encoding

- Use temporal coding

- Use intensity to delay encoding

- Allow only the first n of m spikes to pass through ($N$ of $M$ encoding)

- Alternativly use Rank Order Coding or N of M Rank Order

- Alternativly use a dynamic $N$. This could be a thresold per region of an image
  use relative threshold so that dark spots still get their information trough

- For images, divide the input into spatial chunks and apply n-of-m coding within each chunk to
  preserve spatial information.

- For sequences like text, audio, or video, apply n-of-m coding
  to time frames. For video, combine n-of-m coding across both spatial chunks
  and temporal frames. Can be linked to brain waves.

= Theoretical Foundations

- Prove that the network satisties the universal approximation theorem

- learn how visiontransformers work and how they differ from CNNs


- ressonator neurons and integrator neurons ???

- Design a critical system lookup _Critical Brain Theory_
  https://www.youtube.com/watch?v=vwLb3XlPCB4
  This will make shure signal does not grow out of proportions or shrink to a halt

= Network

- use inhibitory connections---but more importantly figure out how

- Adress event representaion (AER)

- Binary weights, only check if there is and outgoing/incomming spike or not

= Neuron Models
- Integrate-and-Fire Neurons (IF):  
  Output neurons accumulate spikes from their connected synapses within a short time window. If the
  accumulated input exceeds a threshold (e.g., 4/4 synapses fire), the neuron fires. This process
  ensures that only significant patterns propagate further. They need to reset after, either leaky
  or instant, also dependng on wether they fired or not

= Learning

- Use lateral inhibition

- Use Homeostatic Plasticity

- Use Synaptic Competition

- Grow Synapses: If a neuron is close to firing (e.g., 3/4 synapses activate), connect its
  final synapse to the most recent active input neuron. This mimics biological synapse growth

- Move Synapses: Adjust existing synapses toward frequently active input neurons to refine connections.  

- Prune Synapses: Remove inactive synapses over time to maintain efficiency and sparsity.


// TODO
// = Algorithm
// #align(center,
// pseudocode-list(hooks: .5em, line-numbering: none)[
//   + do something
//   + do something else
//   + *while* still something to do
//     + do even more
//     + *if* not done yet *then*
//       + wait a bit
//       + resume working
//     + *else*
//       + go home
// ]
// )

= Experiment 1

#bibliography("citations.bib")
