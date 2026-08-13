#!/usr/bin/env python3
"""
Script to run parameter estimation using the likelihood computation functions.
This script handles the execution flow while keeping all functions in likelihood_computation.py
"""

import sys
import os
import numpy as np
import glob
import time
import pandas as pd
from likelihood_computation import run_parameter_estimation_lbfgsb

############# Define paths #############
RESULTS_PATH = "results_param_recovery/"
BEHAV_FILE_NAME = "model_fitting_data_hybrid_mf_mb_learnt.csv"
#########################################

def process_single_subject(subject_data, agent_type='sr'):
    """
    Process a single subject's behavioral data with parameter estimation.
    
    Arguments:
        - subject_data: DataFrame with all rows for a single participant
        - agent_type: one of
            'sr', 'mb', 'mf', 'mb_learnt',
            'redsr_2', 'redsr_3', 'redsr_4',
            'randsr_noupdate', 'randsr_wupdate',
            'redsr_2_randsr_wupdate', 'redsr_3_randsr_wupdate', 'redsr_4_randsr_wupdate',
            'hybrid_mf_redsr_4_randsr_wupdate', 'hybrid_mf_randsr_noupdate', 'hybrid_mf_randsr_wupdate', 'hybrid_mf_mb_learnt'
        
    Returns:
        - success: boolean indicating if processing was successful
        - results: dictionary with results or error message
    """
    
    try:
        
        # Define task structure
        num_pairs = 13
        num_states = 10
        
        subject_id = subject_data["ID"].unique()[0]
        
        print("\n🚀 Running L-BFGS-B optimization...")
        
        # Fit model to data
        start_time = time.time()
        best_params, best_likelihood, kept_indices, evaluations = run_parameter_estimation_lbfgsb(
            subject_data, num_pairs, num_states, agent_type
        )
        estimation_time = time.time() - start_time
        
        # Package results for aggregation at the agent level
        alpha_rwq = best_params[0] if len(best_params) >= 1 else None
        alpha_mt = best_params[1] if len(best_params) >= 2 else None
        # For MB agents, alpha_mt is conceptually None
        if agent_type == 'mb':
            alpha_mt = None
        beta = best_params[2] if len(best_params) >= 3 else None
        gamma = best_params[3] if len(best_params) >= 4 else None
        w = best_params[4] if len(best_params) >= 5 else None

        # Print best parameters for this subject
        print(
            f"      alpha_rwq={alpha_rwq}, alpha_mt={alpha_mt}, beta={beta}, "
            f"gamma={gamma}, w={w}, logL={best_likelihood}"
        )

        return True, {
            'subject': subject_id,
            'agent_type': agent_type,
            'method': 'L-BFGS-B',
            'alpha_rwq': alpha_rwq,
            'alpha_mt': alpha_mt,
            'beta': beta,
            'gamma': gamma,
            'w': w,
            'best_likelihood': best_likelihood,
            'estimation_time': estimation_time,
            'evaluations': evaluations,
        }

    except Exception:
        error_msg = f"Error processing subject {subject_id}"
        print(f"❌ {error_msg}")
        import traceback
        traceback.print_exc()
        return False, {'subject': subject_id, 'error': error_msg}

def main():
    """Main execution function"""
    
    # Define all available agent types
    ALL_AGENT_TYPES = [
        'sr', 'mb', 'mb_learnt', 'mf',
        'redsr_2', 'redsr_3', 'redsr_4', 
        'randsr_noupdate', 'randsr_wupdate', 
        'redsr_2_randsr_wupdate', 'redsr_3_randsr_wupdate', 'redsr_4_randsr_wupdate',
        'hybrid_mf_redsr_4_randsr_wupdate', 'hybrid_mf_randsr_noupdate', 'hybrid_mf_randsr_wupdate', 'hybrid_mf_mb_learnt'
    ]
    
    # Only one optimization method is supported.
    method = "lbfgsb"
    data_directory = "."  # default data directory (current directory)
    agent_type = 'sr'  # default agent type
    run_all_agents = False  # flag for running all agents
    agents_list = None  # optional explicit list of agents to run (comma-separated)
    
    # Parse arguments
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        
        # Handle flags
        if arg == '--data-dir':
            if i + 1 < len(sys.argv):
                data_directory = sys.argv[i + 1]
                i += 1  # Skip the directory argument
            else:
                print("Error: --data-dir requires a directory path")
                return
        elif arg == '--all-agents':
            run_all_agents = True
        elif arg == '--agents':
            # Comma-separated list of agent types, e.g. --agents sr,mb,redsr_3
            if i + 1 < len(sys.argv):
                raw_list = sys.argv[i + 1]
                agents_list = [a.strip() for a in raw_list.split(',') if a.strip()]
                unknown = [a for a in agents_list if a not in ALL_AGENT_TYPES]
                if unknown:
                    print(f"Error: Unknown agent types in --agents: {', '.join(unknown)}")
                    print(f"Available agent types: {', '.join(ALL_AGENT_TYPES)}")
                    return
                i += 1  # Skip the list argument
            else:
                print("Error: --agents requires a comma-separated list of agent types, e.g. --agents sr,mb,redsr_3")
                return
        else:
            print(f"Error: Unknown argument '{arg}'.")
            print("Usage: python run_estimation.py "
                  "[--data-dir DIRECTORY] "
                  "[--agents TYPE1,TYPE2,...] "
                  "[--all-agents]")
            return
        
        i += 1
    
    # Determine which agent types to run
    if run_all_agents:
        agent_types_to_run = ALL_AGENT_TYPES
        print("🚀 Running L-BFGS-B optimization for ALL AGENTS")
        print(f"📁 Data directory: {os.path.abspath(data_directory)}")
        print(f"🤖 Agent types: {', '.join(agent_types_to_run)}")
    elif agents_list is not None:
        # Run only the explicitly requested subset
        agent_types_to_run = agents_list
        print("🚀 Running L-BFGS-B optimization for SELECTED AGENTS")
        print(f"📁 Data directory: {os.path.abspath(data_directory)}")
        print(f"🤖 Agent types: {', '.join(agent_types_to_run)}")
    else:
        agent_types_to_run = [agent_type]
        print("🚀 Running L-BFGS-B optimization")
        print(f"📁 Data directory: {os.path.abspath(data_directory)}")
        print(f"🤖 Agent type: {agent_type.upper()}")
    
    ###### Read behavioral data file ######
    behavioral_file = os.path.join(data_directory, BEHAV_FILE_NAME)
    
    if not behavioral_file:
        print("❌ No behavioral data file found!")
        print(f"   Search directory: {os.path.abspath(data_directory)}")
        return
    
    print(f"📁 Found behavioral data file: {behavioral_file}")
    
    df = pd.read_csv(behavioral_file)
    
    # Get subjects
    subjects = list(df['ID'].unique())
    
    ###### Process each agent type ######
    
    # Track overall results
    overall_successful_subjects = []
    overall_failed_subjects = []
    overall_all_results = []
    
    for agent_idx, current_agent_type in enumerate(agent_types_to_run, 1):
        print(f"\n{'='*120}")
        print(f"🤖 PROCESSING AGENT {agent_idx}/{len(agent_types_to_run)}: {current_agent_type.upper()}")
        print(f"{'='*120}")
        
        # Track results for this agent
        successful_subjects = []
        failed_subjects = []
        all_results = []
        
        # Process each file for this agent
        for i, subject in enumerate(subjects, 1):
            print(f"\n{'='*80}")
            print(f"📊 Processing subject {i}/{len(subjects)}: {subject}")
            print(f"{'='*80}")
            
            subject_data = df[df['ID']==subject]
            
            # Process the file
            success, results = process_single_subject(subject_data, current_agent_type)
            
            if success:
                successful_subjects.append(subject)
                all_results.append(results)
                print(f"✅ Successfully processed: {subject}")
            else:
                failed_subjects.append(results)
                print(f"❌ Failed to process: {subject}")
        
        # Print summary for this agent
        print(f"\n{'='*80}")
        print(f"📊 SUMMARY FOR AGENT: {current_agent_type.upper()}")
        print(f"{'='*80}")
        print(f"Total subject processed: {len(subjects)}")
        print(f"Successful: {len(successful_subjects)}")
        print(f"Failed: {len(failed_subjects)}")
        
        if successful_subjects:
            print("\n✅ Successfully processed subjects:")
            for subject in successful_subjects:
                print(f"   - {subject}")
        
        if failed_subjects:
            print("\n❌ Failed subjects:")
            for result in failed_subjects:
                print(f"   - {result['subject']}: {result['error']}")
        
        # Add to overall results
        overall_successful_subjects.extend(successful_subjects)
        overall_failed_subjects.extend(failed_subjects)
        overall_all_results.extend(all_results)

        # Save aggregated parameter estimates for this agent (one row per subject)
        if all_results:
            results_df = pd.DataFrame(all_results)
            os.makedirs(RESULTS_PATH, exist_ok=True)
            aggregated_filename = os.path.join(
                RESULTS_PATH,
                f"best_parameters_{method}_{current_agent_type}.csv"
            )
            results_df.to_csv(aggregated_filename, index=False)
            print(f"\n📁 Aggregated parameters for {current_agent_type.upper()} saved to: {aggregated_filename}")
    
    ###### Print overall summary ######
    print(f"\n{'='*120}")
    print("📊 OVERALL PROCESSING SUMMARY")
    print(f"{'='*120}")
    print(f"Agent types processed: {len(agent_types_to_run)}")
    print(f"Total subject-agent combinations: {len(subjects) * len(agent_types_to_run)}")
    print(f"Successful: {len(overall_successful_subjects)}")
    print(f"Failed: {len(overall_failed_subjects)}")
    
    print("\n🎉 All processing complete!")

if __name__ == "__main__":
    main()
