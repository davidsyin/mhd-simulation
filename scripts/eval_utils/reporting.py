import os
import csv
from datetime import datetime


def create_output_filename(base_filename):
    """Create timestamped output filename"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if "." in base_filename:
        base = base_filename.rsplit(".", 1)[0]
        ext = "." + base_filename.rsplit(".", 1)[1]
    else:
        base = base_filename
        ext = ".csv"

    return f"{base}_{timestamp}{ext}"


def save_results(results, output_file, incremental=False):
    """
    Save results to CSV file

    Args:
        results: List of result dictionaries
        output_file: Path to output CSV file
        incremental: If True, note that this is an incremental save
    """
    if not results:
        return

    os.makedirs(
        os.path.dirname(output_file) if os.path.dirname(output_file) else ".",
        exist_ok=True,
    )

    with open(output_file, "w", newline="") as csvfile:
        fieldnames = set()
        for r in results:
            fieldnames.update(r.keys())

        writer = csv.DictWriter(csvfile, fieldnames=sorted(list(fieldnames)))
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    if incremental:
        print(f"Incremental results saved to {output_file}")
    else:
        print(f"\nFinal results saved to {output_file}")


def print_summary_table(results):
    """Print summary table of results"""
    if not results:
        print("No results to display.")
        return

    print("\nSummary of results:")
    print("-" * 110)
    print(
        f"{'Config':<30} | {'Re':<6} | {'Test Loss':<10} | {'Denorm L2':<10} | {'Denorm MSE':<10} | {'Epoch':<5}"
    )
    print("-" * 110)
    for result in results:
        print(
            f"{result['config']:<30} | {result['reynolds_number']:<6} | {result['test_loss']:<10.6f} | "
            f"{result['denorm_test_rel_l2']:<10.6f} | {result['denorm_test_mse']:<10.6f} | {result['checkpoint_epoch']:<5}"
        )
