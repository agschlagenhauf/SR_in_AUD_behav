# Model fitting

This directory contains scripts for fitting the different RL agents to participants' choice data, for model comparison, and parameter recovery.

## How to use

Clone the repo to a local folder and cd into it. 

Then install dependencies:
```bash
# Install dependencies
pip install -r requirements.txt
```

To run model fitting, run the `run_estimation.py` script in your console. The script takes several input arguments:

```bash
# Run all agents
python run_estimation.py --all-agents --data-dir /path/to/data

# Run only a selected subset of agents
python run_estimation.py --agents sr,mb,redsr_3 --data-dir /path/to/data
```

## Agent Types
You can either run **all** agents, or a **selected subset** (comma-separated)
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

## Input data
The estimation script reads a **single combined CSV file** for all subjects from the directory given by `--data-dir`. 
The filename is set at the top of `run_estimation.py` via `BEHAV_FILE_NAME`:
- For empirical participant data, set `BEHAV_FILE_NAME = "modeling_data.csv"` in `run_estimation.py` (exported from `03_data_analysis/05_exclusion_and_balancing_checks.qmd`)
- For simulated data, set `BEHAV_FILE_NAME = model_fitting_data_....csv` (produced by `prepare_simulated_data_for_fitting.qmd` for parameter-recovery runs)

Required columns (column names must match exactly):
- `ID`: subject identifier
- `component`: e.g. `reward-learning`, `reward-relearning`, `reward-test`, `transition-learning`, `goal-state-learning`, `policy-learning`, etc.
- `trial`: trial number within each phase
- `state`: state description (e.g., `"1LeftTo2Right"`)
- `choice`: action chosen (`"left"`, `"right"`, or NaN for forced transitions)
- `reward`: reward received (numeric)
- `correct_first_state_action`: `"left"` or `"right"`; used to construct transition and reward structures

## RL Parameters
1. **α_RWQ** (`alpha_rwq`): learning rate for the reward structure 
2. **α_MT** (`alpha_mt`): set to equal alpha_rwq (except for `mb_learnt` agent)  
3. **β** (`beta`): Inverse temperature (choice randomness; fixed to 1.0)
4. **γ** (`gamma`): Time discount factor
5. **w** (`w`): Mixture weight for hybrid agents only—MF vs. non-MF branch. The three-parameter hybrids (`hybrid_mf_redsr_4_randsr_wupdate`, `hybrid_mf_randsr_noupdate`, `hybrid_mf_randsr_wupdate`) estimate `alpha_rwq`, `gamma`, and `w`; `hybrid_mf_mb_learnt` additionally estimates `alpha_mt` (four parameters total).

## Output Files
Parameter estimates are written to the folder set by `RESULTS_PATH` at the top of `run_estimation.py` (default: `results_param_recovery/`). For empirical participant fits, set `RESULTS_PATH = "results_real_data/"`.

For each agent type and method, `run_estimation.py` creates **one CSV per model** with one row per subject:
- `{RESULTS_PATH}/best_parameters_{method}_{agent_type}.csv`

Columns include:
- `subject`, `agent_type`, `method`
- `alpha_rwq`, `alpha_mt`, `beta`, `gamma`, `w` (mixture weight for hybrid agents only; otherwise present but may be empty/NaN depending on export)
- `best_likelihood`, `estimation_time`, `evaluations`

## Model Comparison (AIC / BIC)
`compare_models.py` aggregates parameter files across models and computes model comparison metrics and parameter summaries.

Run, for example:
```bash
python compare_models.py --results-dir results_real_data
```

This produces (in `--results-dir`):

- `model_fit_per_participant_{method}.csv`  
  Per-subject, per-model table with:
  - `subject`, `agent_type`, `method`
  - `alpha_rwq`, `alpha_mt`, `beta`, `gamma`
  - `best_likelihood`, `k_params`, `n_choices`, `AIC`, `BIC`

- `model_comparison_summary_{method}.csv`  
  One row per model with:
  - `agent_type`, `n_subjects_total`, `n_subjects_excluded_inf`, `n_subjects_used`
  - `k_params`, `n_choices`
  - `loglik_mean`, `loglik_sd`
  - `AIC_mean`, `AIC_sd`
  - `BIC_mean`, `BIC_sd`

- `parameter_summary_{method}.csv`  
  One row per model with:
  - `agent_type`
  - Mean and SD per parameter, e.g. `alpha_rwq_mean`, `alpha_rwq_sd`, `alpha_mt_mean`, `alpha_mt_sd`, `beta_mean`, `beta_sd`, `gamma_mean`, `gamma_sd`

BIC uses `n_choices = 250` (5 conditions × 25 likelihood trials × 2 choices per trial).