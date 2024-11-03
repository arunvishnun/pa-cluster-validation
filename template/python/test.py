import os
from typing import List, Tuple
from submission import Solution  # Import Solution from submission.py

# Initialize Solution instance
solution = Solution()

# Define the folder path for test cases
TEST_CASES_FOLDER = "../../sample_test_cases"

def read_input(file_path: str) -> Tuple[int, List[Tuple[int, int]]]:
    """Reads the test input file and parses it into the appropriate format."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
        test_id = int(lines[0].strip())  # Read the test_id
        labels = [tuple(map(int, line.strip().split())) for line in lines[1:]]
    return test_id, labels

def read_output(file_path: str) -> str:
    """Reads the expected output file."""
    with open(file_path, 'r') as f:
        return f.read().strip()

def validate_test(test_id: int, labels: List[Tuple[int, int]], expected_output: str) -> bool:
    """Validates the output of the Solution methods against expected output."""
    true_labels, pred_labels = zip(*labels)
    if test_id == 0:  # Jaccard test
        output = solution.jaccard(list(true_labels), list(pred_labels))
        print(f"Test ID 0 (Jaccard) - Calculated: {output}, Expected: {float(expected_output)}")  # Debug output
        return abs(output - float(expected_output)) < 1e-4
    elif test_id == 1:  # NMI test
        output = solution.nmi(list(true_labels), list(pred_labels))
        print(f"Test ID 1 (NMI) - Calculated: {output}, Expected: {float(expected_output)}")  # Debug output
        return abs(output - float(expected_output)) < 1e-4
    elif test_id == 2:  # Confusion matrix test
        output = solution.confusion_matrix(list(true_labels), list(pred_labels))
        output_str = "\n".join(f"{i} {j} {v}" for (i, j), v in sorted(output.items()))
        print(f"Test ID 2 (Confusion Matrix) - Calculated:\n{output_str}\nExpected:\n{expected_output}")  # Debug output
        return output_str.strip() == expected_output
    return False

# List of test files
test_cases = [
    ("input00.txt", "output00.txt"),
    ("input01.txt", "output01.txt"),
    ("input02.txt", "output02.txt"),
    # Add more test cases if available
]

# Run the tests
for input_file, output_file in test_cases:
    # Construct full file paths
    input_path = os.path.join(TEST_CASES_FOLDER, input_file)
    output_path = os.path.join(TEST_CASES_FOLDER, output_file)
    
    # Read inputs and expected output
    test_id, labels = read_input(input_path)
    expected_output = read_output(output_path)
    
    # Validate the output
    if validate_test(test_id, labels, expected_output):
        print(f"Test {input_file} passed.")
    else:
        print(f"Test {input_file} failed.")
