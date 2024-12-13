import os
# Get current file directory
directory = os.path.dirname(os.path.abspath(__file__))

def find_missing_indices(directory=os.path.join(directory, "RACE_high")):
    # Get list of files in directory
    files = [f for f in os.listdir(directory) if f.endswith('.txt')]
    
    # Extract indices from filenames
    indices = set(int(f.split('.')[0]) for f in files)
    
    # Find max index
    max_index = max(indices)
    
    # Find missing indices
    all_indices = set(range(1, max_index + 1))
    missing = sorted(all_indices - indices)
    
    return missing

def find_longest_passages(directory=os.path.join(directory, "RACE_high"), n=100):
    """
    Find the n longest passages in the given directory.
    
    Args:
        directory (str): Directory containing text files
        n (int): Number of longest passages to return (default 100)
        
    Returns:
        list: List of tuples (filename, passage length) sorted by length in descending order
    """
    # Get list of files in directory
    files = [f for f in os.listdir(directory) if f.endswith('.txt')]
    
    # Store (filename, length) tuples
    passage_lengths = []
    
    for filename in files:
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            # Load file content and parse article text
            import json
            content = json.load(f)
            article = content['article']
            passage_lengths.append((filename, len(article)))
    
    # Sort by length in descending order and get top n
    passage_lengths.sort(key=lambda x: x[1], reverse=True)
    return passage_lengths[:n]

def copy_longest_passages(directory=os.path.join(directory, "RACE_high"), n=100):
    """
    Copy the n longest passages to a new directory called RACE_min.
    
    Args:
        directory (str): Source directory containing text files
        n (int): Number of longest passages to copy (default 100)
    """
    # Get longest passages
    longest = find_longest_passages(directory, n)
    
    # Create RACE_min directory if it doesn't exist
    min_dir = os.path.join(os.path.dirname(directory), "RACE_min")
    os.makedirs(min_dir, exist_ok=True)
    
    # Copy files
    for filename, _ in longest:
        src = os.path.join(directory, filename)
        dst = os.path.join(min_dir, filename)
        import shutil
        shutil.copy2(src, dst)


if __name__ == "__main__":
    # missing = find_missing_indices()
    # if missing:
    #     print("Missing indices:", missing)
    # else:
    #     print("No missing indices found")

    longest = find_longest_passages()
    print(longest)
    # copy_longest_passages()