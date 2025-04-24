#!/bin/bash
source ../../.venv/bin/activate

# Get the directory of this script
current_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Find all test files
test_files=$(find "$current_dir" -name "test_*.py")

# Run each test file
for test_file in $test_files; do
    echo -e "\nRunning tests in $(basename "$test_file")..."
    python "$test_file"
done
