import os
import json
import subprocess

def run_tests():
    # Folder paths based on the challenge structure
    input_dir = os.path.join("data", "test_cases", "inputs")
    expected_dir = os.path.join("data", "test_cases", "expected_outputs")
    
    # The command to run your solution (Windows format)
    solution_cmd = ["python", os.path.join("solution", "solution.py")]
    
    passed = 0
    failed = 0
    total = 0
    
    print("========================================================")
    print("          Box Box Box - Windows Test Runner             ")
    print("========================================================\n")
    
    # Loop through all 100 test files
    for filename in sorted(os.listdir(input_dir)):
        if not filename.endswith(".json"):
            continue
            
        total += 1
        input_path = os.path.join(input_dir, filename)
        expected_path = os.path.join(expected_dir, filename)
        
        test_id = filename.replace(".json", "").upper()
        
        # 1. Run your solution script and pipe the input file to it
        with open(input_path, 'r') as infile:
            result = subprocess.run(solution_cmd, stdin=infile, capture_output=True, text=True)
            
        # 2. Check the results
        try:
            # Parse your script's output
            output_data = json.loads(result.stdout.strip())
            predicted = output_data.get("finishing_positions", [])
            
            # Parse the actual correct answers
            with open(expected_path, 'r') as exp_file:
                expected_data = json.loads(exp_file.read())
                expected_answers = expected_data.get("finishing_positions", [])
                
            # Compare the arrays exactly
            if predicted == expected_answers:
                print(f"[✓ PASS] {test_id}")
                passed += 1
            else:
                print(f"[✗ FAIL] {test_id} - Incorrect Prediction")
                failed += 1
                
        except Exception as e:
            print(f"[! ERROR] {test_id} - Crash or Invalid Output: {e}")
            failed += 1

    # 3. Print Final Score
    print("\n========================================================")
    print("                       RESULTS                          ")
    print("========================================================")
    print(f"Total Tests:  {total}")
    print(f"Passed:       {passed}")
    print(f"Failed:       {failed}")
    
    score = (passed / total) * 100 if total > 0 else 0
    print(f"\nFinal Score:  {score:.1f}%")
    
    if score == 100.0:
        print("🏆 Perfect score! You cracked the F1 mechanics!")
    else:
        print("Keep grinding! We need to analyze the historical data to find the exact math.")

if __name__ == "__main__":
    run_tests()