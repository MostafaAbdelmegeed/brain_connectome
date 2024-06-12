import argparse
import sys
from graphIO import read_ppmi_data_as_tensors
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, help='Directory containing the PPMI data.')
    parser.add_argument('--method', type=str, default='correlation', help='Method used to compute the adjacency matrices.')
    parser.add_argument('--atlas', type=str, default='AAL116', help='Atlas used to compute the adjacency matrices.')
    parser.add_argument('--destination', type=str, default='ppmi.pth', help='Save path for the processed data.')
    args = parser.parse_args()
    return args





if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_ppmi_to_pth.py <directory>")
        sys.exit(1)
    args = parse_args()
    source = args.source
    destination = args.destination
    atlas = args.atlas
    method = args.method
    data = read_ppmi_data_as_tensors(source, atlas=atlas, method=method)
    torch.save(data, destination)