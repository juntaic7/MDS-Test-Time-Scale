import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any
import agents.unified_batch_agent as batch_agent


def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """Load configuration from JSON file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file {config_path} does not exist.")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def setup_argument_parser() -> argparse.ArgumentParser:
    """Setup minimal command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Run batch experiments with commercial LLM agents using config file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Example usage: \n 
        python batch_exp.py --config experiments/consistency_gpt.json\n
        python batch_exp.py --config config.json"""
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to configuration file containing all experiment parameters (default: config.json)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually sending the batch request"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser


def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration structure and required fields."""
    required_experiment_fields = ["agent", "requests_file"]
    
    if "experiment" not in config:
        raise ValueError("Config must contain 'experiment' section")
    
    experiment = config["experiment"]
    missing_fields = [field for field in required_experiment_fields if field not in experiment]
    if missing_fields:
        raise ValueError(f"Missing required experiment fields: {missing_fields}")
    
    if not os.path.exists(experiment["requests_file"]):
        raise FileNotFoundError(f"Requests file {experiment['requests_file']} does not exist.")
    
    # Validate agent is supported
    if "models" in config and experiment["agent"] not in config["models"]:
        available_agents = list(config["models"].keys())
        raise ValueError(f"Agent '{experiment['agent']}' not supported. Available: {available_agents}")

def load_prompts_from_requests(requests_file: str) -> Dict[int, str]:
    """Load prompts from JSONL file in requests folder."""
    prompts = {}
    with open(requests_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    request_data = json.loads(line)
                    # Extract id and msg from request structure
                    if 'id' in request_data and 'msg' in request_data:
                        prompts[request_data['id']] = request_data['msg']
                except json.JSONDecodeError:
                    continue
    return prompts


def get_model_for_agent(config: Dict[str, Any], experiment_config: Dict[str, Any]) -> str:
    """Get model name for agent from experiment config or use default."""
    agent = experiment_config["agent"]
    model = experiment_config.get("model")
    
    if model:
        return model

    if "models" in config and agent in config["models"]:
        return config["models"][agent]["default"]
    
    # Handle unknown agents
    defaults = {"gpt": "gpt-4o-mini", "claude": "claude-3-5-sonnet-20240620", "qwen": "qwen-turbo"}
    return defaults.get(agent, "gpt-4o-mini")


def generate_output_filename(experiment_config: Dict[str, Any], model: str) -> str:
    """Generate output filename for batch input."""
    agent = experiment_config["agent"]
    requests_name = os.path.splitext(os.path.basename(experiment_config["requests_file"]))[0]
    
    base_name = f"{agent}_{model}_{requests_name}_batch_input.jsonl"
    
    output_dir = experiment_config.get("output_dir")
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        return os.path.join(output_dir, base_name)
    
    return base_name


def print_experiment_summary(config: Dict[str, Any], experiment_config: Dict[str, Any], model: str, prompt_count: int, filename: str, verbose: bool = False) -> None:
    """Print a summary of the experiment configuration."""
    print("\n" + "="*60)
    print("BATCH EXPERIMENT SUMMARY")
    print("="*60)
    print(f"Agent: {experiment_config['agent']} ({model})")
    print(f"Requests file: {experiment_config['requests_file']}")
    print(f"Request prompts: {prompt_count}")
    print(f"Output: {filename}")
    if experiment_config.get('description'):
        print(f"Description: {experiment_config['description']}")
    
    if verbose and "batch_settings" in config:
        print("\nBatch Settings:")
        for key, value in config["batch_settings"].items():
            print(f"   {key}: {value}")
    print("="*60)


def main():
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        experiment_config = config["experiment"]
        batch_settings = config.get("batch_settings", {})
    
        validate_config(config)
        
    
        prompts = load_prompts_from_requests(experiment_config["requests_file"])
        model = get_model_for_agent(config, experiment_config)
        f_in = generate_output_filename(experiment_config, model)
        
        print_experiment_summary(config, experiment_config, model, len(prompts), f_in, args.verbose)
        
        if args.dry_run:
            print("\nDRY RUN - No batch request will be sent.")
            print(f"Would prepare file: {f_in}")
            print(f"Would process {len(prompts)} request prompts")
            return 0
        
        agent = batch_agent.create_batch_agent(experiment_config["agent"], model)
        
        prepare_kwargs = {
            "docs": prompts,
            "prompt": "",
            "filename": f_in,
            "max_tokens": batch_settings.get("max_tokens", 4096),
            "temperature": batch_settings.get("temperature", 0.8)
        }
        
        if batch_settings.get("sys_message"):
            prepare_kwargs["sys_message"] = batch_settings["sys_message"]
            
        agent._prepare_request_file(**prepare_kwargs)
        print(f"\nPrepared batch input file: {f_in}")
        
        # Send batch request
        print("\nSending batch request...")
        description = experiment_config.get("description", "Batch experiment")
        batch_id = agent._send_batch_request(description=description, filename=f_in)
        
        if batch_id:
            print(f"\nBatch request sent successfully!")
            print(f"Batch ID: {batch_id}")
            print(f"Input file: {f_in}")
            print("\nYou can monitor progress using our batch_manager, or through your API dashboard!")
        else:
            print("\nFailed to send batch request.")
            return 1
    
    except Exception as e:
        print(f"\nError: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())