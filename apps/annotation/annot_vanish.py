# This script shows an example to annotate vanishing lines
from easymocap.annotator import ImageFolder
from easymocap.annotator import plot_text, plot_skeleton_simple, vis_active_bbox, vis_line
from easymocap.annotator import AnnotBase
from easymocap.annotator.vanish_callback import get_record_vanish_lines, get_calc_intrinsic, clear_vanish_points, vanish_point_from_body, copy_edges, clear_body_points
from easymocap.annotator.vanish_visualize import vis_vanish_lines

def annot_example(path, annot, sub=None, step=100):
    # define datasets
    dataset = ImageFolder(path, sub=sub, annot=annot)
    key_funcs = {
        'X': get_record_vanish_lines(0),
        'Y': get_record_vanish_lines(1),
        'Z': get_record_vanish_lines(2),
        'k': get_calc_intrinsic('xy'),
        'K': get_calc_intrinsic('yz'),
        'b': vanish_point_from_body,
        'C': clear_vanish_points,
        'B': clear_body_points,
        'c': copy_edges,
    }
    # define visualize
    vis_funcs = [vis_line, plot_skeleton_simple, vis_vanish_lines]
    # construct annotations
    annotator = AnnotBase(
        dataset=dataset, 
        key_funcs=key_funcs,
        vis_funcs=vis_funcs,
        step=step)
    while annotator.isOpen:
        annotator.run()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--annot', type=str, default='annots')
    parser.add_argument('--step', type=int, default=100)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    annot_example(args.path, args.annot, step=args.step)