import os
import json
from typing import Dict, List

def get_final_results(base_path: str) -> Dict[str, List[float]]:
    """
    Get final results from each approach's result files.
    
    Args:
        base_path (str): Base path to the results directory
        
    Returns:
        Dict mapping approach names to lists of accuracy values
    """
    print(f"Getting final results")
    results = {}
    approaches = ['cot', 'fot', 'got', 'io', 'tot']
    
    for approach in approaches:
        approach_path = os.path.join(base_path, approach)
        if not os.path.exists(approach_path):
            print(f"Approach directory not found: {approach_path}")
            continue
            
        results[approach] = []
        
        # Iterate through all json files in the approach directory
        for filename in os.listdir(approach_path):
            print(f"Processing {filename}")
            if filename.endswith('.json'):
                file_path = os.path.join(approach_path, filename)
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        # Extract final accuracy from the result
                        if isinstance(data, dict):
                            # Build graph from nodes and edges
                            nodes = {node["id"]: node for node in data["nodes"]}
                            adj_list = {node_id: [] for node_id in nodes}
                            for edge in data["edges"]:
                                adj_list[edge["source"]].append(edge["target"])
                            
                            # Find nodes with no outgoing edges (final nodes)
                            final_nodes = []
                            for node_id in nodes:
                                if not adj_list[node_id]:
                                    final_nodes.append(nodes[node_id])
                            print(f"Final nodes: {final_nodes}")
                                    
                            # Get accuracy from final node content
                            for node in final_nodes:
                                if "content" in node["data"]:
                                    try:
                                        # Try to extract numeric accuracy value from content
                                        content = node["data"]["content"]
                                        if isinstance(content, (int, float)):
                                            results[approach].append(float(content))
                                    except:
                                        print(f"Error extracting accuracy from {file_path}")
                                        continue
                        else:
                            print(f"Invalid data format in {file_path}")
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    
    return results

def main():
    # Path to results directory
    # Get current file's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, "sorting_test/results/chatgpt4o-mini_io-cot-tot-got-fot_2024-12-02_14-26-23")
    
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return
    else:
        print(f"Results directory found: {results_dir}")
        
    results = get_final_results(results_dir)
    print(f"Results: {results}")
    
    # Print summary for each approach
    for approach, accuracies in results.items():
        if accuracies:
            avg_accuracy = sum(accuracies) / len(accuracies)
            print(f"{approach.upper()} Results:")
            print(f"Number of tests: {len(accuracies)}")
            print(f"Average accuracy: {avg_accuracy:.2%}")
            print(f"Individual accuracies: {accuracies}")
            print()

if __name__ == "__main__":
    main()
