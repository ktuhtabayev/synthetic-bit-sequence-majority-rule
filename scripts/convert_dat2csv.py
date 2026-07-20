from __future__ import annotations

import pandas as pd


def main() -> None:
    # Example 1
    # input_path = r"datasets\raw\dog-wolf\dog-wolf.dat"
    # output_path = r"datasets\raw\dog-wolf\dog-wolf.csv"

    # Example 2
    input_path = r"datasets\raw\gipertaniya\gipertaniya.dat"
    output_path = r"datasets\raw\gipertaniya\gipertaniya.csv"

    # df = pd.read_csv(input_path, delimiter="\t", decimal=",")
    df = pd.read_csv(input_path, delimiter="\t", decimal=".")

    df.to_csv(output_path, index=False)

    print(f"File converted successfully! Saved as {output_path}")


if __name__ == "__main__":
    main()
