from agents import challenge_agent, metadata_extraction, code_agent 
from agents import research_agent, planner_agent, data_analysis_agent
from utils import prompt_input, token_counter, artifacts
from pathlib import Path
import time
import json

def main():
    run_dir = artifacts.create_run_dir()
    print(f"\n Artifacts will be saved to: {run_dir}")

    challenge_rules = prompt_input.ask_for_contest_rules()
    if isinstance(challenge_rules, dict):
        if challenge_rules.get("type") == "url":
            challenge_rules = challenge_rules.get("url", "")
        elif challenge_rules.get("type") == "file":
            challenge_rules = str(challenge_rules.get("path", ""))
        else:
            challenge_rules = ""
    if not challenge_rules:
        raise ValueError("Contest rules are required (URL or PDF path).")

    if challenge_agent.is_url(challenge_rules):
        challenge_debrief, ch_tokens = challenge_agent._extract_from_url(challenge_rules)
    else:
        challenge_debrief, ch_tokens = challenge_agent._extract_from_pdf(Path(challenge_rules))

    dataset_path = prompt_input.ask_for_zip_folder()

    # 1) list files in zip
    files_in_zip = metadata_extraction.list_zip_files(dataset_path)

    # 2) extract safely
    extract_dir = metadata_extraction.safe_extract_zip(
        dataset_path,
        extract_dir="extracted_data",
        max_files=5000,  # optional safety cap
    )

    # 3) inspect extracted files and build metadata
    dataset_metadata = metadata_extraction.extract_metadata_file(
        files_in_zip,
        extract_dir=str(extract_dir),
    )

    # optional: serialize for passing to agents
    dataset_metadata_json = json.dumps(dataset_metadata, ensure_ascii=False)

    print("\n Challenge debrief is ready.")
    
    # estimating costs per agent
    ch_costs = token_counter.calculate_cost(ch_tokens[0], ch_tokens[1], 0.30, 2.50)
    print(f"\n Estimated costs from Challenge Debrief Agent: {ch_costs}")
    time.sleep(5) 
    
    # running research agent
    research_output, research_tokens = research_agent.run_research(
        dataset_metadata_json,
        challenge_debrief,
        artifacts_run_dir=str(run_dir),
    )
    print("\n Research is ready.")
    
    # estimating costs per agent
    research_costs = token_counter.calculate_cost(research_tokens[0], research_tokens[1], 1.50, 9.00)
    print(f"\n Estimated costs from Research Agent: {research_costs}")
    time.sleep(5) 
    
    # next step
    print("\n Now, EDA of the dataset")
    dataset_profile_for_planner = dataset_metadata_json

    # get the EDA of the dataset
    while True: 
        eda_question = input("\n Do you have data analysis in the form of json file of the dataset? (yes/no):\n> ").strip().lower()
        if eda_question == "yes": 
            eda_file = input("\n Please provide file name if it is locally installed:\n> ").strip()
            eda_filepath = Path(eda_file)
            data_analysis_text, data_analysis_tokens, data_analysis_debug = (
                data_analysis_agent.get_data_analysis_for_planner(
                    dataset_access=str(extract_dir),
                    metadata=dataset_metadata_json,
                    challenge_rules=challenge_debrief,
                    eda_json_path=eda_filepath,
                )
            )
            dataset_profile_for_planner = (
                f"{dataset_metadata_json}\n\n--- DATA ANALYSIS ---\n{data_analysis_text}"
            )
            break
        elif eda_question == "no":
            print("\n Please note that running Data Analysis Agent can incur greater costs")
            confirm = input("\n Would you like to run Data Analysis Agent? (yes/no):\n> ").strip().lower()
            if confirm == "yes": 
                data_analysis_text, data_analysis_tokens, data_analysis_debug = (
                    data_analysis_agent.get_data_analysis_for_planner(
                        dataset_access=str(extract_dir),
                        metadata=dataset_metadata_json,
                        challenge_rules=challenge_debrief,
                        max_files=3,
                        max_total_mb=200,
                    )
                )
                dataset_profile_for_planner = (
                    f"{dataset_metadata_json}\n\n--- DATA ANALYSIS ---\n{data_analysis_text}"
                )
                # estimating costs per agent
                data_al_costs = token_counter.calculate_cost(data_analysis_tokens[0], data_analysis_tokens[1],0.30, 2.50)
                print(f"\n Estimated costs from Data Analysis Agent: {data_al_costs}")
                time.sleep(5) 
                break
            if confirm == "no": 
                break
            else:
                print("Please answer yes or no.")
        else:
            print("Please answer yes or no.")
    print("\n Next, compile actionable plan for research implementation")
    
    # creating plan for code agent 
    plan_output, plan_tokens= planner_agent.create_plan(
        challenge_debrief,
        dataset_profile_for_planner,
        research_output,
        artifacts_run_dir=str(run_dir),
    )
    print("\n Actionable plan was created by the agent")
    print("\n For now we don't estimate costs for Planner Agent")
    #compiling code, and extracting final results of ML lab
    code_output, code_tokens = code_agent.execute_code(
        str(extract_dir),
        plan_output,
        artifacts_run_dir=str(run_dir),
    )
    code_costs = token_counter.calculate_cost(code_tokens[0], code_tokens[1], 1.50, 9.00)
    print(f"\n Estimated costs from Code Agent: {code_costs}")
    time.sleep(5)
    return code_output 


    
