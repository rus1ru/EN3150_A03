"""Run the stronger Model B v2; original Model B workflows remain reproducible."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from en3150_a03.model_b_v2_train import main
if __name__=="__main__":
    main()
