"""Train Model B using Adam, SGD and momentum SGD, then evaluate and report."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from en3150_a03.model_b_experiment import main

if __name__ == "__main__":
    main()
