import argparse
from agents.unified_batch_agent import create_batch_agent


def check_status(agent_type: str, batch_id: str) -> None:
    """Check the status of a batch job."""
    try:
        agent = create_batch_agent(agent_type)
        
        if agent_type == "claude":
            batch_obj = agent.client.messages.batches.retrieve(batch_id)
        else:  # gpt or qwen
            batch_obj = agent.client.batches.retrieve(batch_id)
        
        print("="*60)
        print(f"BATCH STATUS: {batch_id}")
        print("="*60)
        print(f"Agent: {agent_type}")
        print(f"Status: {batch_obj.status}")
        
        if hasattr(batch_obj, 'created_at'):
            print(f"Created: {batch_obj.created_at}")
        if hasattr(batch_obj, 'completed_at') and batch_obj.completed_at:
            print(f"Completed: {batch_obj.completed_at}")
        if hasattr(batch_obj, 'request_counts'):
            counts = batch_obj.request_counts
            print(f"Requests - Total: {counts.total}, Completed: {counts.completed}, Failed: {counts.failed}")
        if hasattr(batch_obj, 'errors') and batch_obj.errors:
            print(f"Errors: {batch_obj.errors}")
        
        print("="*60)
        
    except Exception as e:
        print(f"Error checking batch status: {e}")


def retrieve_results(agent_type: str, batch_id: str, output_path: str, raw: bool = False) -> None:
    """Retrieve results from a completed batch job."""
    try:
        agent = create_batch_agent(agent_type)
        
        if raw:
            results = agent._retrieve_batch_results_raw(batch_id, output_path)
            print(f"Raw batch results saved to {output_path}")
        else:
            results = agent._retrieve_batch_results(batch_id, output_path)
            print(f"Processed batch results saved to {output_path}")
        
        print(f"Retrieved {len(results)} results from batch {batch_id}")
        
    except Exception as e:
        print(f"Error retrieving batch results: {e}")


def list_batch_history() -> None:
    """List recent batch jobs from history file."""
    try:
        with open("batch_history.txt", "r") as f:
            lines = f.readlines()
        
        if not lines:
            print("No batch history found.")
            return
            
        print("="*60)
        print("RECENT BATCH JOBS")
        print("="*60)
        print("Timestamp           | Agent  | Batch ID                     | Description")
        print("-" * 60)
        
        for line in lines[-10:]:  # Show last 10 entries
            if line.strip():
                print(line.strip())
                
    except FileNotFoundError:
        print("No batch history file found.")
    except Exception as e:
        print(f"Error reading batch history: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Manage batch jobs: check status, retrieve results, or view history.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
                python batch_manager.py status -m gpt -b batch_123
                python batch_manager.py retrieve -m claude -b batch_456 -p results.jsonl
                python batch_manager.py retrieve -m qwen -b batch_789 -p results.jsonl
                python batch_manager.py history"""
    )
    
    parser.add_argument(
        "action",
        choices=["status", "retrieve", "history"],
        help="Action: 'status' to check batch status, 'retrieve' to download results, 'history' to view recent jobs"
    )
    
    parser.add_argument(
        "-m", "--model-type",
        type=str,
        choices=["gpt", "claude", "qwen"],
        help="Type of LLM agent (required for status/retrieve actions)"
    )
    
    
    parser.add_argument(
        "-b", "--batch-id",
        type=str,
        help="Batch ID (required for status/retrieve actions)"
    )
    
    parser.add_argument(
        "-p", "--path",
        type=str,
        help="Output file path (required for retrieve action)"
    )
    
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Retrieve raw results instead of processed format"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    try:
        if args.action == "history":
            list_batch_history()
            
        elif args.action == "status":
            if not args.model_type or not args.batch_id:
                parser.error("status action requires --model-type and --batch-id")
            check_status(args.model_type, args.batch_id)
            
        elif args.action == "retrieve":
            if not args.model_type or not args.batch_id or not args.path:
                parser.error("retrieve action requires --model-type, --batch-id, and --path")
            retrieve_results(args.model_type, args.batch_id, args.path, args.raw)
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()