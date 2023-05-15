# Introduction
A biomarker, a molecular marker or signature molecule, refers to a biological substance or characteristic found in body fluids, tissues, or blood that indicates the presence of a condition, disease, or abnormal process. Biomarkers can be measured to assess how well the body responds to treatment for a particular disease or condition. Biomarkers play a crucial role in drug discovery and development by providing essential information on the safety and effectiveness of drugs. These measurable indicators can be categorized into diagnostic, prognostic, or predictive biomarkers, and they are utilized to choose patients for clinical trials or track patient response and treatment efficacy. Despite its valuable role, biomarker discovery is a challenging task for classical-computational platforms due to the massive search space.

Quantum computing is an emerging technology that utilizes the principles of quantum mechanics to solve problems beyond classical computers' capabilities. Quantum Machine Learning and Quantum Neural Networks are an advanced class of machine intelligence on quantum hardware, which promises more powerful models for myriad learning tasks.

This work uses a class of Quantum Artificial Intelligence (AI) models to discover biomarkers in biomedical research. We adopt the neural architecture proposed in a contemporary work that addresses the body dynamics modeling problem. Here, we make a non-trivial adaptation of the proposed game-theoric to tackle a different class of problems. The main contribution of this study is summarized as follows:
- The proposed quantum AI model is a general, cost-efficient, and cost-effective algorithm for biomarker discovery, despite the extensive problem complexity.
- The model outcomes suggest novel biomarkers for the mutational activation of the notable target in immuno-therapy - CLTA4, including $20$ genes: _CLIC4_, _CPE_, _ETS2_, _FAM107A_, _GPR116_, _HYOU1_, _LCN2_, _MACF1_, _MT1G_, _NAPA_, _NDUFS5_, _PAK1_, _PFN1_, _PGAP3_, _PPM1G_, _PSMD8_, _RNF213_, _SLC25A3_, _UBA1_ and _WLS_.

# Sampling Codes
- Select the target pathway in python script:
```python
evaluator.py
```
- Sampling parallel with multiple CPU workers:
```python
main.py
```

# Implementation Details
- Model:
```
model.py
```
- Loss Module:
```
loss.py
```
- Utilities:
```
utils.py
```
