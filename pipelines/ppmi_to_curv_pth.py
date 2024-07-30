import argparse
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
from graphIO.io import read_ppmi_data_as_tensors
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, help='Directory containing the PPMI data.')
    parser.add_argument('--destination', type=str, default='ppmi.pth', help='Save path for the processed data.')
    args = parser.parse_args()
    return args





if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ppmi_to_curv_pth.py <directory>")
        sys.exit(1)
    args = parse_args()
    source = args.source
    destination = args.destination
    curvatures = read_ppmi_data_as_tensors(source, method='new')
    torch.save(curvatures, destination)