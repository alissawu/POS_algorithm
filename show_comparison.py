#!/usr/bin/env python3
"""Show side-by-side comparison of gold vs predicted tags."""

def show_comparison(gold_file="WSJ_24.pos", pred_file="viterbi_dev.pos", output_file=None, max_lines=None):
    """Display or save comparison of gold vs predicted tags."""

    # Load files
    gold = []
    pred = []

    with open(gold_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) == 2:
                    gold.append(parts)
            else:
                gold.append(["", ""])  # Empty line for sentence break

    with open(pred_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) == 2:
                    pred.append(parts)
            else:
                pred.append(["", ""])  # Empty line for sentence break

    # Prepare output
    lines = []
    errors_only = []

    # Header
    lines.append("="*80)
    lines.append(f"{'WORD':<25} {'GOLD':<10} {'PREDICTED':<10} {'STATUS':<10}")
    lines.append("="*80)

    error_count = 0
    total_count = 0

    for i, (g, p) in enumerate(zip(gold, pred)):
        if g[0] == "":  # Sentence break
            lines.append("")  # Empty line for readability
            errors_only.append("")
            continue

        g_word, g_tag = g
        p_word, p_tag = p

        if g_word != p_word:
            lines.append(f"ERROR: Word mismatch at position {i}")
            continue

        total_count += 1
        status = "✓" if g_tag == p_tag else "✗"

        if g_tag != p_tag:
            error_count += 1
            status_text = f"✗ {g_tag}->{p_tag}"
        else:
            status_text = "✓"

        line = f"{g_word:<25} {g_tag:<10} {p_tag:<10} {status_text:<10}"
        lines.append(line)

        if g_tag != p_tag:
            errors_only.append(line)

        if max_lines and len(lines) > max_lines + 10:  # Account for headers
            break

    # Add summary
    lines.append("")
    lines.append("="*80)
    lines.append(f"SUMMARY: {error_count} errors out of {total_count} words")
    lines.append(f"Accuracy: {(total_count - error_count) / total_count * 100:.3f}%")
    lines.append("="*80)

    # Output
    if output_file:
        with open(output_file, 'w') as f:
            f.write('\n'.join(lines))
        print(f"Full comparison saved to {output_file}")

        # Also save errors-only file
        errors_file = output_file.replace('.txt', '_errors.txt')
        with open(errors_file, 'w') as f:
            f.write("ERRORS ONLY:\n")
            f.write("="*80 + "\n")
            f.write(f"{'WORD':<25} {'GOLD':<10} {'PREDICTED':<10} {'ERROR':<10}\n")
            f.write("="*80 + "\n")
            for line in errors_only:
                if line or line == "":  # Include empty lines for sentence breaks
                    f.write(line + "\n")
        print(f"Errors-only saved to {errors_file}")
    else:
        # Print to console
        print('\n'.join(lines))

def show_specific_errors(error_type, gold_file="WSJ_24.pos", pred_file="viterbi_dev.pos"):
    """Show only specific types of errors."""

    gold = []
    pred = []

    with open(gold_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) == 2:
                    gold.append(parts)

    with open(pred_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) == 2:
                    pred.append(parts)

    gold_tag, pred_tag = error_type.split('->')

    print(f"\n{'='*80}")
    print(f"Showing {gold_tag} -> {pred_tag} errors:")
    print(f"{'='*80}\n")

    for i, (g, p) in enumerate(zip(gold, pred)):
        if g[1] == gold_tag and p[1] == pred_tag:
            # Get context
            start = max(0, i-2)
            end = min(len(gold), i+3)

            context = []
            for j in range(start, end):
                if j == i:
                    context.append(f"[{gold[j][0]}/{gold_tag}->{pred_tag}]")
                else:
                    context.append(f"{gold[j][0]}/{gold[j][1]}")

            print(f"{' '.join(context)}")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if '->' in sys.argv[1]:
            # Show specific error type
            show_specific_errors(sys.argv[1])
        else:
            # Save to file
            show_comparison(output_file=sys.argv[1])
    else:
        # Interactive mode
        print("1. Show all (first 100 lines)")
        print("2. Save full comparison to file")
        print("3. Show specific error type (e.g., NN->JJ)")

        choice = input("\nChoice: ")

        if choice == "1":
            show_comparison(max_lines=100)
        elif choice == "2":
            filename = input("Output filename (e.g., comparison.txt): ")
            show_comparison(output_file=filename)
        elif choice == "3":
            error_type = input("Error type (e.g., NN->JJ): ")
            show_specific_errors(error_type)