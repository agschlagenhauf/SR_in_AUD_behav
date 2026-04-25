# Simulating agents' behavior in sequential decision-making task

Code in this folder simulates the behavior of the various RL models in the multi-stage decision-making task, either based on selected fixed parameter values or on individually fitted parameter values (parametric bootstrapping simulations).

## Usage
Setup python virtualenv and install dependencies:
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Run the simulations with:
```
python main.py
```
This script will run `simulate.py` for the models and conditions specified in `main.py`. Each model is saved as a separate script containing `learning` and `relearning` functions which run the respective `run_trial` function.

## Agent Types
1. **sr**: Standard successor representation
2. **mb**: Model-based agent
3. **mb_learnt**: Model-based agent with separate learning rates for reward and transition structure
4. **redsr_2**: Reduced successor representation (2 goals)
5. **redsr_3**: Reduced successor representation (3 goals)
6. **redsr_4**: Reduced successor representation (4 goals)
7. **randsr_noupdate**: Random-policy SR (MB learning (trials 0-4) → random-policy SR → frozen matrix, no weight updates)
8. **randsr_wupdate**: Random-policy SR (MB learning (trials 0-4) → random-policy SR → frozen matrix, weight updates only)
9. **redsr_2_randsr_wupdate**: Reduced random-policy (2 goals, MB → random-policy SR → RedSR_2 reduction → weight updates only)
10. **redsr_3_randsr_wupdate**: Reduced random-policy (3 goals, MB → random-policy SR → RedSR_3 reduction → weight updates only)
11. **redsr_4_randsr_wupdate**: Reduced random-policy (4 goals, MB → random-policy SR → RedSR_4 reduction → weight updates only)
12. **mf**: Model-free agent
13. **hybrid_mf_redsr_4_randsr_wupdate**: Mixture of MF and RedSR-4 random-policy SR agent
14. **hybrid_mf_randsr_noupdate**: Mixture of MF and random-policy noupdate SR agent
15. **hybrid_mf_randsr_wupdate**: Mixture of MF and full random-policy SR with SR weight updates during free-choice learning (no RedSR reduction; same three-parameter fit as items 13–14)
16. **hybrid_mf_mb_learnt**: Mixture of MF and MB agent with separate learning rates for reward and transition structure

## Output
Output is stored either in the `results_fixed/` directory (for simulations based on selected fixed parameter values) or in the `results_ppc/` directory (for parametric bootstrapping simulations). Output files per model contain an entire log of choices across all task phases.
