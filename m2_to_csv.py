import csv
import os

def parse_m2_file(m2_file_path, output_csv_path):
    """
    Parses an M2 file and creates a parallel dataset CSV (incorrect, correct).
    
    Args:
        m2_file_path: Path to the input .m2 file.
        output_csv_path: Path to the output .csv file.
    """
    print(f"Parsing M2 file: {m2_file_path}")
    
    if not os.path.exists(m2_file_path):
        print(f"Error: File not found at {m2_file_path}")
        return

    data_pairs = []
    
    current_sentence = None
    edits = []

    with open(m2_file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            
            if line.startswith('S '):
                # If there's a previous sentence being processed, apply its edits
                if current_sentence is not None:
                    corrected = apply_edits(current_sentence, edits)
                    data_pairs.append([current_sentence, corrected])
                
                # Start new sentence
                # Skip 'S ' (first 2 chars)
                current_sentence = line[2:].strip()
                edits = []
                
            elif line.startswith('A '):
                # Parse edit: A start end|||type|||correction|||...
                # We need start, end, and correction
                parts = line[2:].split('|||')
                offset_info = parts[0].split()
                start_token = int(offset_info[0])
                end_token = int(offset_info[1])
                correction = parts[2]
                
                # Store edit: (start, end, correction)
                edits.append((start_token, end_token, correction))
                
            elif line == "":
                # Empty line usually separates sentences, but strict M2 might not have it.
                # We handle the final flush after the loop or on next 'S'.
                pass

        # Handle the very last sentence
        if current_sentence is not None:
            corrected = apply_edits(current_sentence, edits)
            data_pairs.append([current_sentence, corrected])

    # Write to CSV
    print(f"Writing {len(data_pairs)} pairs to {output_csv_path}...")
    with open(output_csv_path, 'w', encoding='utf-8-sig', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['incorrect', 'correct'])
        writer.writerows(data_pairs)
    print("Done.")

def apply_edits(source_sentence, edits):
    """
    Applies edits to the source sentence.
    Crucial: Edits must be applied in reverse order of offsets to avoid shifting indices.
    """
    # Tokenize the source sentence by whitespace to match M2 token offsets
    tokens = source_sentence.split()
    
    # Sort edits by start offset in descending order (reverse order)
    # If multiple edits have same start, order doesn't strictly matter if they are disjoint,
    # but usually M2 edits are on specific spans.
    # What if they overlap? M2 usually implies one choice or alternative.
    # The prompt says "Apply All Edits... All 'A' lines".
    # We will assume they are compatible or take the standard GEC approach.
    # Sorting by start index descending is key.
    edits.sort(key=lambda x: x[0], reverse=True)
    
    for start, end, correction in edits:
        if correction == "-NONE-":
            continue
            
        # Replace the slice of tokens with the correction
        # Correction string usually needs to be tokenized if it replaces multiple tokens?
        # Or is it a string replacement?
        # M2 format: S The boy... -> tokens: ["The", "boy"]
        # A 1 1|||...|||boys|||... -> Replace token at index 1 ("boy") with "boys"
        
        # If correction is empty/deletion, it might be an empty string depending on M2 flavor
        # but usually it's just replacement content.
        
        # Handle "noop" or similar if any (usually just -NONE- which we skipped)
        
        # Replacement
        # Note: correction string from M2 is usually just the text "boys".
        # We replace the list slice.
        
        if start == -1: # Some M2 (CoNLL-2014) use -1 -1 for no-edits?
            continue
            
        # Calculate replacement tokens
        # The correction string itself might contain spaces, so we should arguably keep it as a single unit 
        # OR split it? 
        # Usually for reconstruction, we just put the string in.
        # But wait, if we reconstruct to a string at the end, we rejoin by space.
        
        # Let's simple split the correction by space to insert as tokens, 
        # so final join " " correction " " works cleanly.
        replacement_tokens = correction.split()
        
        tokens[start:end] = replacement_tokens

    # Reconstruct sentence
    return " ".join(tokens)

if __name__ == "__main__":
    # Define paths
    # Using absolute paths based on previous file finding
    # Double nesting confirmed
    base_dir = r"c:\Users\Mrh\Documents\Fam\QALB-0.9.1-Dec03-2021-SharedTasks\QALB-0.9.1-Dec03-2021-SharedTasks\data\2015\dev"
    m2_file = os.path.join(base_dir, "QALB-2015-L2-Dev.m2")
    output_file = r"c:\Users\Mrh\Documents\Fam\qalb_full_gec.csv"
    
    parse_m2_file(m2_file, output_file)
