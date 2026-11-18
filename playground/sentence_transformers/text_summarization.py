"""
This example uses LexRank (https://jair.org/index.php/jair/article/view/10396/24901)
to create an extractive summarization of a long document.

The document is split into sentences using NLTK, then the sentence embeddings are computed. 
We then compute the cosine-similarity across all possible sentence pairs.

We then use LexRank to find the most central sentences in the document, which form our summary.

Input document: First section from the English Wikipedia Section
Output summary:
Located at the southern tip of the U.S. state of New York, the city is the center of the New York metropolitan area, the largest metropolitan area in the world by urban landmass.
New York City (NYC), often called simply New York, is the most populous city in the United States.
Anchored by Wall Street in the Financial District of Lower Manhattan, New York City has been called both the world's leading financial center and the most financially powerful city in the world, and is home to the world's two largest stock exchanges by total market capitalization, the New York Stock Exchange and NASDAQ.
New York City has been described as the cultural, financial, and media capital of the world, significantly influencing commerce, entertainment, research, technology, education, politics, tourism, art, fashion, and sports.
If the New York metropolitan area were a sovereign state, it would have the eighth-largest economy in the world.

Note: Requires NLTK: `pip install nltk`
"""

# NB
# The pip implementation of LexRank is broken.
# A working implementation can be found at https://github.com/huggingface/sentence-transformers/blob/main/examples/sentence_transformer/applications/text-summarization/lex_rank.py.
import random
import nltk
import numpy as np
from lex_rank import degree_centrality_scores

from sentence_transformers import SentenceTransformer

nltk.download('punkt_tab')
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Our input document we want to summarize
# As example, we take the first section from Wikipedia
document = """
From [https://www.nist.gov/quantum-information-science/quantum-computing-explained]
An official website of the United States government
Official websites use .govA.govwebsite belongs to an official government organization in the United States.
Secure .gov websites use HTTPSAlock(LockA locked padlock) orhttps://means you’ve safely connected to the .gov website. Share sensitive information only on official, secure websites.
https://www.nist.gov/quantum-information-science/quantum-computing-explained
A computer that could break the encryption that safeguards your private information on the internet. A machine that can design powerful new drugs by precisely simulating the behavior of individual molecules. A device that optimizes complex supply chains to help companies get the parts they need and assemble them in the most efficient way possible.
These are all examples of how an emerging technology — the quantum computer — could change our world.
These computers work by harnessing quantum physics — the strange, often counterintuitive laws that govern the universe at its smallest scales and coldest temperatures. Today’s quantum computers are rudimentary and error-prone. But if more advanced and robust versions can be made, they have the potential to rapidly crunch through certain problems that would take current computers years. That’s why governments, companies and research labs around the world are working feverishly toward this goal.
Quantum computers will not replace our familiar “classical” computers. Rather, the two types of machines could work together to solve problems that stymie classical computers, potentially supercharging scientific research in fields such as materials and drug discovery, giving a boost to industry and upending cybersecurity as we know it.
Let’s explore how quantum computers work.
Quantum physics describes the universe at its smallest and most fundamental scales — think atoms and molecules; light and energy. Things at these scales behave very differently from everyday objects we’re familiar with.
One of the most important differences involves a concept calledsuperposition. Let’s start by considering an everyday, human-scale object such as a person on a ladder. Depending on which rung the person stands on, they have a certain amount of potential energy. (This potential energy determines how fast the person would be moving when they hit the ground if they were to jump off the ladder.) A person on the ground has the smallest possible amount of energy in this system. Someone on the first rung has slightly more energy, and so on up to the highest rung.
By contrast, tiny objects such as atoms can act as though they have two or more distinct amounts of energy at once. In our ladder example, this would be akin to simultaneously standing on the ladder’s lowest and highest rungs — something that makes no sense for a person.
Once placed into this kind of mixed energy state, known as a “quantum superposition,” an atom will remain there until it is measured or disturbed by the outside world. Then the atom “collapses” to a single energy state — following our analogy, to either the low or high rung of the ladder.
To cast this idea into familiar terms, the famous physicist Erwin Schrödinger came up with a memorable, though absurd, thought experiment: Imagine a perfectly sealed box containing a cat and a poison trap that can be triggered by the decay of a radioactive atom. Because the decay of the atom is uncertain, at any given time, the cat is in a superposition of dead and alive. Only when someone opens the box and measures the cat does its state “collapse” to being either definitively alive or definitively dead. Real cats can’t be both alive and dead, of course, but Schrödinger’s imaginary cat has become an enduring metaphor to help people grapple with the strangeness of superposition.
Building on the superposition concept, multiple atoms or other quantum objects can beentangledwith each other to share a single quantum state. Now imagine several cats in Schrödinger’s box, potential victims of the same trap. These cats are “entangled” in a superposition of all being alive or all being dead. When someone opens the box, not just the state of one cat but those of all the cats immediately collapse, and each cat is found to be fully alive or fully dead.
“Entanglement means you’ve got at least two things that are always connected; they have no independent existence,” explains NIST physicist Andrew Wilson.
These scenarios strike us as absurd when applied to familiar objects such as cats. But at the atomic level, this is how the world works. Tiny objects such as atoms can exist in multiple states simultaneously. And these states can be entangled with those of other objects even when the objects are far apart. “Let’s say you have an entangled pair of particles and you put one on the Moon and the other on the surface of the Earth. If you then do something to the one on the Earth, you simultaneously affect the other,” Wilson says. “It’s kind of romantic!”
We’ll soon explore how physicists use these ideas to build quantum computers. But first we need to understand ...
These days, we use computers for just about everything: gaming and streaming, banking and shopping, following our favorite sports teams and chatting with friends and family members. But we rarely think about what a computer is or how it works.
At its most fundamental level, a computer is any device that takes in data, processes it, stores it and spits it out. The phones in our pockets, the servers in data centers, the microprocessors in our cars and the room-sized supercomputers at national labs: All of these digital computers encode and process information using “bits.” Bits are “binary digits” that encode information — text, graphics and so on — as 1s and 0s. For example, computers typically represent the letter “A” using the bit string “01000001.”
But bits, like the computers they are part of, are not just mathematical concepts. They need to be realized in physical objects such as tiny bar magnets or electric switches that can be placed into one of two distinct states, say pointed up or pointed down.
Bits are very good at what they do. Put a bit into a “0” or “1” state and it will usually stay there for a long time, meaning the information it encodes is stable and long-lasting. But bits are also limited.
Quantum computers also have input, output, information processing and memory. But instead of regular classical bits, quantum computers use quantum bits, orqubits. Like Schrödinger’s unfortunate cat, qubits can be put into superpositions of multiple states. In other words, a qubit can be in state 0, state 1, or a mix of the two. And the quantum states of individual qubits can be entangled with each other.
These capabilities give quantum computers their superpower. Whereas two classical bits can be placed in only one combination of states at a time (such as 1 and 0), two qubits can contain a superposition of four combinations of 0s and 1s simultaneously. Three qubits can contain eight combinations; four qubits, 16 combinations and so on. Each additional qubit doubles the number of combinations: an exponential increase.
Someone using a quantum computer must first entangle qubits to harness their exponential computing power. The operator then carries out operations on the qubits, such as addition, multiplication or more complicated computations. Depending on the type of quantum computer, electromagnetic signals or lasers create the entanglement and operations.
Though they are capable of exponential computation, quantum computers are limited in the amount of data they can extract from these computations — a fact that’s often lost in popular descriptions giving the impression that quantum computers try every solution to a problem at once.
“Different computations can indeed be done in superposition, achieving a kind of parallel computing,” says Stephen Jordan, a Google quantum computing researcher who was a longtime NIST staff member andJoint Center for Quantum Information and Computer Science fellow. “But contrary to popular belief, this doesn’t allow quantum computers to do an efficient ‘brute force’ search over all the potential solutions.
“The measurement at the end of the computation can only extract a small amount of information about the results of all of these computations,” Jordan explains. “The key is to design the measurement so that it extracts useful information about the whole set of results done in superposition.”
(Note: This article describes quantum computers that do computations using logic gates, similar to classical computers. Some scientists and companies are pursuing another technology known as “quantum annealing” that could be used to solve certain physics and optimization problems faster than classical computers can.)
At a 1981 gathering of physicists outside Boston, the famous physicist Richard Feynman spoke about the possibility of “simulating physics with computers.” Though other scientists had independently developed similar ideas around the same time, Feynman’s talk is often credited with launching the field of quantum computing.
Since then, scientists have explored how quantum computers could, in theory, simulate the fundamental quantum rules that govern molecules, chemicals and materials — something today’s computers can only approximate with great effort. If quantum computers eventually become large and powerful enough, scientists hope that such quantum simulations could bring about major advances in materials science, drug development and other areas. Potentially transformative “killer apps” for quantum simulation could involve discovering a new blockbuster drug or chemical catalysts that make the production of fertilizer or the capture of greenhouse gases from the air more efficient.
In 1994, a mathematician named Peter Shorpublished a paperabout a very different application that instantly made quantum computing a national security issue. The algorithms that encrypt much of our data work by multiplying very large prime numbers together to create a secret key — something that’s very hard for classical computers to undo. Shor’s paper described a quantum algorithm that could quickly factor the immense numbers that are products of these huge prime numbers, potentially putting much of the world’s encrypted information at risk.
Scientists also believe quantum computers could outpace classical computers at solving complicated optimization problems such as helping companies organize complex processes such as airplane assembly in more efficient ways.
Most of these applications are years — perhaps even decades — in the future. But scientists have started to publish papers claiming that quantum computers have demonstrated a “quantum advantage,” meaning they can outdo classical computers for certain tasks. Quantum computers have been used to calculate the energies of small molecules, for example, and simulate the magnetic properties of collections of interacting atoms.
So far, none of these early demonstrations have proved truly useful, says Scott Glancy, a physicist at NIST. And in some cases, scientists later showed that traditional computers could equal or exceed the performance of quantum processors for some tasks. The demonstrations do, however, prove that quantum computers work and can be scaled up.
“It seems to me we’re just on the threshold of quantum systems doing genuinely new simulations that we can’t do classically,” says Glancy.
Beyond these practical applications, quantum computers could offer a new way to probe the fundamental nature of reality. A full-scale quantum computer, if successfully built, would contain some of the most complex quantum states ever created (assuming aliens have not already built such devices). Those states would provide an important, albeit not surprising, confirmation of quantum theory.
If, on the other hand, scientists find that a large-scale quantum computer cannot be built, that would be “shocking,” says Glancy. “It might inspire a revolution in physics. In my opinion that is a good reason to build a quantum computer.”
Qubits are exquisite but fragile. A stray electric or magnetic field, temperature fluctuations or even a cosmic ray can ruin a superposition or entanglement. This forces qubits into a 0 or 1 state in which they act like ordinary bits. Anyone building a quantum computer must find ways to manipulate the qubits carefully while protecting them from outside disturbances.
Moreover, a single qubit by itself is worth little. For a quantum device to do something useful, many qubits must be entangled with each other while sustaining superpositions. The best quantum computers today contain hundreds of interconnected qubits and make an error roughly once in every thousand operations. An error changes the state of a qubit, destroying or corrupting the information it carries.
(By contrast, a classical computer makes around one error, such as a bit randomly flipping from 0 to 1, for every quintillion — 1 followed up 18 zeroes — calculations. And correcting errors in a classical computer is much easier.)
Industry, university and government researchers around the world are racing to make more reliable qubits and build electronics and laser systems that create entanglement more efficiently and robustly. And they are experimenting with many kinds of qubits. In theory, any particle or system that obeys the rules of quantum physics, from atoms to tiny circuits to semiconductors, can act as a qubit.
Each qubit has advantages and disadvantages. For example, one of the most popular qubit types useselectrically charged atoms known as ions. The quantum energy states of electrons inside these ions represent the 0s and 1s (and combinations thereof) for quantum computation. Ion qubits can sustain quantum superpositions for a long time, but they are relatively sluggish at performing computations.
Another popular qubit usestiny circuits made from superconductors— materials that conduct electricity without resistance at very cold temperatures. The behavior of the electrons in the circuits creates quantized energy states that can be used to encode 0s and 1s. These qubits allow for fast computations and can be made using existing chip manufacturing techniques. But their quantum states are more fragile and shorter-lived than those of ion qubits.
Scientists are also experimenting with qubits based onarrays of neutral (non-electrically charged) atoms,atoms embedded in diamonds,particles of light known as photonsandsmall bits of silicon.
Some researchers are also trying to develop a radically different type of qubit, known as a “topological” qubit, that would have some built-in immunity to errors. In theory, topological qubits could encode quantum information into the braiding pattern of “quasiparticles” that emerge from the collective behavior of individual particles such as electrons. These braiding patterns, and thus the quantum states, would be protected from some of the outside disturbances that can disrupt other qubits.
Topological qubits require temperatures near absolute zero and complicated structures often involving superconducting and semiconducting materials. They have proved challenging to build, and researchers are still seeking definitive evidence that they have managed to make one.
Ultimately, quantum computers may marry multiple kinds of qubits so that each can play to its strengths. Superconducting or photonic qubits could crunch through operations quickly, for example, then transfer their information to ion or diamond qubits for storage.
An overview of some of the most-studied and best-funded qubit types.Image credits: N. Hanacek/NIST
In the near term, quantum computer designers hope to build machines that are sophisticated and stable enough to do useful tasks that traditional computers can’t do. Some researchers expect these “noisy intermediate-scale” quantum computers to excel at simulation; others are skeptical. Current quantum computers are being used mainly to explore certain physics, chemistry and mathematical problems and as test beds to understand how to make more powerful quantum computers.
For other tasks, such as running Shor’s code-breaking algorithm, a quantum computer may need millions of qubits that can run error-free indefinitely, like our classical computers do. Such a quantum computer is probably still much further away.
(Even so,NIST has developed new algorithmsthat are thought to be quantum computer-proof and is encouraging businesses and government agencies to adopt them.Learn more about post-quantum cryptography.)
Many experts believe that because they are so complex and delicate, quantum computers will probably never sit on or desks or in our pockets. Rather, they may live inside commercial computing centers, national labs and universities, where they will crunch quantum information and deliver solutions that make our world smarter, safer and more efficient.
--Written by Gabriel Popkin
Webmaster|Contact Us|Our Other Offices
"""

# Split the document into sentences
sentences = nltk.sent_tokenize(document)
print("Num sentences:", len(sentences))

threshold = random.uniform(0.5, 1.0)
print("Threshold:", threshold)

relevant_sentences = int(len(sentences)/10)
print("Relevant sentences:", relevant_sentences)

# Compute the sentence embeddings
embeddings = model.encode(sentences)

# Compute the similarity scores
similarity_scores = model.similarity(embeddings, embeddings).numpy()

# Compute the centrality for each sentence
centrality_scores = degree_centrality_scores(similarity_scores, threshold)

# We argsort so that the first element is the sentence with the highest score
most_central_sentence_indices = np.argsort(-centrality_scores)


# Print the 5 sentences with the highest scores
print("\n\nSummary:")
for idx in most_central_sentence_indices[0:relevant_sentences]:
    print(sentences[idx].strip())
