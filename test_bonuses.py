#!/usr/bin/env python3
"""Test each bonus individually to see its impact."""

import subprocess
import sys

def test_configuration(description, modifications):
    """Test a specific bonus configuration."""
    print(f"\nTesting: {description}")
    print("-" * 50)

    # Read the clean file
    with open("aw5571_HW3_clean.py", "r") as f:
        content = f.read()

    # Apply modifications
    for old, new in modifications:
        content = content.replace(old, new)

    # Write test version
    with open("test_version.py", "w") as f:
        f.write(content)

    # Run and score
    subprocess.run([
        "python3", "test_version.py",
        "--train", "WSJ_02-21.pos",
        "--test", "WSJ_24.words",
        "--output", "test_output.pos",
        "--mode", "viterbi"
    ], capture_output=True)

    result = subprocess.run([
        "python3", "score.py",
        "WSJ_24.pos", "test_output.pos"
    ], capture_output=True, text=True)

    # Parse accuracy
    for line in result.stdout.split('\n'):
        if 'accuracy:' in line:
            accuracy = float(line.split()[-1])
            print(f"Accuracy: {accuracy:.6f}%")
            return accuracy

    return 0.0

# Test configurations
configs = [
    ("Baseline (all bonuses)", []),

    ("No numeric bonus", [
        ("    if is_numeric_token(word):\n        if tag == \"CD\":\n            bonus += math.log(1.6)\n        elif tag not in {\"LS\", \"JJ\"}:\n            bonus += math.log(0.4)",
         "    # DISABLED: numeric bonus")
    ]),

    ("No symbol bonus", [
        ("    if is_symbol_token(word):\n        if tag in PUNCT_TAGS:\n            bonus += math.log(1.6)\n        else:\n            bonus += math.log(0.3)",
         "    # DISABLED: symbol bonus")
    ]),

    ("No capitalization bonuses", [
        ("    if word[:1].isupper() and not word.isupper():\n        if tag in CAPITAL_TAGS:\n            bonus += math.log(1.2)\n        elif tag in {\"NN\", \"JJ\"}:\n            bonus += math.log(0.85)",
         "    # DISABLED: initial cap bonus"),
        ("    if word.isupper() and len(word) > 1:\n        if tag in CAPITAL_TAGS | {\"NN\"}:\n            bonus += math.log(1.15)\n        elif tag == \"JJ\":\n            bonus += math.log(0.8)",
         "    # DISABLED: all caps bonus")
    ]),

    ("No -ly adverb bonus", [
        ("    if lower.endswith(\"ly\"):\n        if tag == \"RB\":\n            bonus += math.log(1.25)\n        elif tag in {\"NN\", \"JJ\", \"VB\"}:\n            bonus += math.log(0.7)",
         "    # DISABLED: -ly bonus")
    ]),

    ("No adjective-like bonus", [
        ("    if tag == \"JJ\":\n        if is_adjective_like(word):\n            bonus += math.log(1.15)\n        else:\n            bonus += math.log(0.85)",
         "    # DISABLED: adjective-like bonus")
    ]),

    ("No frequency bonus", [
        ("    counts = params.word_tag_counts.get(word)",
         "    counts = None  # DISABLED frequency bonus\n    if False")
    ])
]

results = []
for desc, mods in configs:
    acc = test_configuration(desc, mods)
    results.append((desc, acc))

print("\n" + "="*60)
print("SUMMARY OF RESULTS:")
print("="*60)

baseline = results[0][1]
for desc, acc in results:
    diff = acc - baseline
    sign = "+" if diff >= 0 else ""
    print(f"{desc:40} {acc:8.4f}% ({sign}{diff:+.4f}%)")