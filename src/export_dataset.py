"""Export scikit-learn's bundled diabetes dataframe to the repository data folder."""

from pathlib import Path

from sklearn.datasets import load_diabetes


def main() -> None:
    destination = Path(__file__).resolve().parents[1] / "data" / "diabetes.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    dataset = load_diabetes(as_frame=True)
    frame = dataset.frame
    frame.to_csv(destination, index=False)
    print(f"Wrote {len(frame)} rows and {len(frame.columns)} columns to {destination}")


if __name__ == "__main__":
    main()
