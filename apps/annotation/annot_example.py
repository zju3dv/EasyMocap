# This script shows an example of our annotator
from easymocap.annotator import ImageFolder
from easymocap.annotator import vis_point, vis_line
from easymocap.annotator import AnnotBase

def annot_example(path):
    # define datasets
    dataset = ImageFolder(path)
    # define visualize
    vis_funcs = [vis_point, vis_line]
    # construct annotations
    annotator = AnnotBase(
        dataset=dataset, 
        key_funcs={},
        vis_funcs=vis_funcs)
    while annotator.isOpen:
        annotator.run()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, default='/home/')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    annot_example(args.path)