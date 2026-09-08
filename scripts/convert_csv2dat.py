from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> None:
    # Example 1
    # input_path = r"datasets\raw\dog-wolf\dog-wolf.csv"
    # output_path = r"datasets\raw\dog-wolf\dog-wolf.dat"

    # Example 2
    # input_path = r"datasets\raw\gipertaniya\gipertaniya.csv"
    # output_path = r"datasets\raw\gipertaniya\gipertaniya.dat"

    # Example 3
    input_path = r"datasets\nominal\molecular-biology\Molecular-Biology (106, 57, 2).csv"
    output_path = r"datasets\nominal\molecular-biology\Molecular-Biology (106, 57, 2).dat"

    input_path = Path(input_path)
    output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Read CSV file
    df = pd.read_csv(input_path)

    # Save as DAT file with tab separator
    # df.to_csv(output_path, sep="\t", index=False)

    # Save as SPACE file with tab separator
    df.to_csv(output_path, sep=" ", index=False)

    print(f"File converted successfully! Saved as {output_path}")


if __name__ == "__main__":
    main()