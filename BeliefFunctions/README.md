# Belief Function Theory (Dempster-Shafer) Implementation

## Overview
This project implements belief function theory (also known as Dempster-Shafer theory or evidence theory) to model the stratified knowledge base from previous exercises.

## Installation

### Step 1: Install required packages
```bash
pip install -r requirements.txt
```

### Alternative: Manual installation
```bash
pip install pyds numpy pandas
```

## Toolboxes Used

We use the **pyds** library, which is available at:
- PyPI: https://pypi.org/project/pyds/
- Documentation: https://pyds.readthedocs.io/

Other notable belief function toolboxes (for reference):
1. **Dempster-Shafer Engine** (C++): https://www.softpedia.com/get/Science-CAD/Dempster-Shafer-Engine.shtml
2. **Arnaud Martin's Toolboxes** (MATLAB): http://people.irisa.fr/Arnaud.Martin/toolboxes/
3. **BFA Society Software**: https://bfasociety.org/#software
4. **ipptoolbox** (R): https://cran.r-project.org/web/packages/ipptoolbox/

## Usage

Run the belief function model:
```bash
cd BeliefFunctions
python BeliefFunctionModel.py
```

This will:
1. Load the stratified knowledge base
2. Create mass functions for each variable
3. Calculate belief and plausibility measures
4. Demonstrate Dempster's rule of combination
5. Generate a CSV report with results

## Key Concepts

### Mass Function
A mass function m assigns values in [0,1] to subsets of the frame of discernment:
- `m(A)` represents the belief exactly assigned to hypothesis A
- `Σ m(A) = 1` for all focal elements A

### Belief Function (Bel)
The belief function measures the total support for a hypothesis:
- `Bel(A) = Σ m(B)` for all B ⊆ A
- Represents the minimum belief in A

### Plausibility Function (Pl)
The plausibility function measures the maximum possible belief:
- `Pl(A) = 1 - Bel(¬A)`
- `Pl(A) = Σ m(B)` for all B ∩ A ≠ ∅
- Represents the maximum belief in A

### Uncertainty Interval
The interval `[Bel(A), Pl(A)]` represents the uncertainty about hypothesis A:
- Lower bound: confirmed evidence
- Upper bound: possible evidence
- Width: degree of ignorance

### Dempster's Rule of Combination
Combines evidence from multiple independent sources:
```
m₁₂(A) = [Σ m₁(B)m₂(C)] / (1 - K)
         B∩C=A

where K = Σ m₁(B)m₂(C) is the conflict
      B∩C=∅
```

## Model Description

### From Possibility Theory to Belief Functions

Our previous work (in `../PossibilityTheory/`) calculated **interest values** for variables based on a stratified knowledge base. These values can be interpreted as belief measures:

1. **Interest Value → Belief Mass**
   - Interest value of 0.14 for variable `d` → `m({d=True}) = 0.14`
   - Remaining mass → `m({d=True, d=False}) = 0.86` (uncertainty)

2. **Stratified Weights → Evidence Strength**
   - Higher stratum weights (e.g., 0.9) indicate stronger evidence
   - Lower weights (e.g., 0.14) indicate weaker evidence

3. **Combining Multiple Strata**
   - Each stratum provides independent evidence
   - Use Dempster's rule to combine evidence from multiple strata

### Results Interpretation

For each variable, the model provides:
- **Bel(var=True)**: Minimum guaranteed belief that var is true
- **Pl(var=True)**: Maximum possible belief that var is true
- **Uncertainty**: The width of the interval [Bel, Pl]

Example for variable `d` with interest value 0.14:
- `Bel(d=True) = 0.14` (at least 14% belief)
- `Pl(d=True) = 1.00` (at most 100% belief)
- `Uncertainty = 0.86` (86% ignorance)

## Files Generated

1. **belief_function_results.csv**: Complete results for all variables
   - Columns: Variable, Interest_Value, Bel_True, Pl_True, Bel_False, Pl_False, Uncertainty

## Comparison: Possibility Theory vs. Belief Functions

| Aspect | Possibility Theory | Belief Functions |
|--------|-------------------|------------------|
| **Measure** | Possibility (Π), Necessity (N) | Belief (Bel), Plausibility (Pl) |
| **Interpretation** | What is possible/necessary | What is believed/plausible |
| **Duality** | Π(A) = 1 - N(¬A) | Pl(A) = 1 - Bel(¬A) |
| **Combination** | Max-min operations | Dempster's rule |
| **Ignorance** | Π(A) = Π(¬A) = 1 | Bel(A) = 0, Pl(A) = 1 |

## Example Output

```
Variable: d
===================
Mass Function:
   m(d=T) = 0.1400
   m(d=?) = 0.8600

Belief and Plausibility:
   Bel(d=True)  = 0.1400
   Pl(d=True)   = 1.0000
   Bel(d=False) = 0.0000
   Pl(d=False)  = 0.8600
   Uncertainty Interval: [0.1400, 1.0000]
```

## Stratified Knowledge Base

The knowledge base contains 9 strata with weights from 0.9 (most certain) to 0.14 (least certain):

| Stratum | Weight | Formulas |
|---------|--------|----------|
| 1 | 0.90 | (a∨¬b∨d∨e), (¬b∨c) |
| 2 | 0.78 | (b∨¬c) |
| 3 | 0.65 | (a∨¬b∨d∨e), (¬b∨c), (¬a∨b∨d) |
| 4 | 0.60 | (a∨¬b∨d∨e), (b∨c) |
| 5 | 0.58 | (¬a∨¬b∨d) |
| 6 | 0.43 | (a∨¬b∨d∨e), (¬b∨c), (a∨b∨d) |
| 7 | 0.36 | (a∨e), (¬b∨c∨f) |
| 8 | 0.26 | (¬a∨d) |
| 9 | 0.14 | (d) |

## References

1. Shafer, G. (1976). *A Mathematical Theory of Evidence*. Princeton University Press.
2. Smets, P. (1988). "Belief functions". In *Non-Standard Logics for Automated Reasoning*.
3. Dempster, A. P. (1967). "Upper and lower probabilities induced by a multivalued mapping". *Annals of Mathematical Statistics*.
4. Dubois, D., & Prade, H. (1988). "Representation and combination of uncertainty with belief functions and possibility measures". *Computational Intelligence*.

## Project Structure

```
BeliefFunctions/
├── BeliefFunctionModel.py    # Main implementation
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── belief_function_results.csv # Generated results (after running)
```

## Author
Created for RCR2 - Reasoning under Uncertainty course

## License
Educational use - M2 Project
