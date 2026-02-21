# Multi$$^2$$: Multi-Agent Test-Time Scalable Framework for Multi-Document Processing

## Workflow

1. Construct the batch request file in `json` format, each line of the request file should have format `{id: some_number, msg: some_content}`
2. Edit request metadata in `config.json`
3. Submit request using `python batch_exp.py`, request metadata will be saved in `batch_history.txt` for future reference
4. Monitor status by `python batch_manager.py status -m {agent_name} -b {batch_id}`, or via the official API (supported by GPT and Claude)
5. Retrieve the results by `python batch_manager.py retrieve -m {agent_name} -p {path_to_save} -b {batch_id}`
6. ATTENTION: Retrieved summaries should be saved to `summaries/dataset_name/model_name` folder, LLMCompare results should be saved to `results/dataset_name/model_name` folder
7. Compute scores using `eval.py` and `compute_cc_score.py`
