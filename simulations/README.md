## Project Structure

The experiments are split into three main phases. The codebase is organized into executable scripts and internal modules.

### Executable Scripts
These files serve as the entry points for the experiments. They are meant to be run directly:
* `phase.py`: Executes Phase 1.
* `phase2a.py`: Executes Phase 2 (Part A).
* `phase2b.py`: Executes Phase 2 (Part B).
* `phase3.py`: Executes Phase 3.

### Internal Modules
These files contain the core implementations, algorithms, and data handling. They are designed to be imported by the executable scripts and should not be run directly:
* `_ssn.py`: Snn implementation
* `_fcn.py`: Simple Pytorch fully connected feed forward network implementation
* `_sim.py`: Simulation logic and helper functions.
* `_data.py`: Data loading and processing utilities.
