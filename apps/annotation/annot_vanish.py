'''
  @ Date: 2021-04-22 19:09:23
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-07-16 14:45:27
  @ FilePath: /EasyMocap/apps/annotation/annot_vanish.py
'''
# This script shows an example to annotate vanishing lines
from easymocap.annotator.file_utils import read_json, save_annot
from easymocap.annotator import ImageFolder
from easymocap.annotator import plot_text, vis_active_bbox, vis_line, plot_skeleton
from easymocap.annotator import AnnotBase
from easymocap.annotator.vanish_callback import get_record_vanish_lines, get_calc_intrinsic, clear_vanish_points, vanish_point_from_body, copy_edges, clear_body_points
from easymocap.annotator.vanish_visualize import vis_vanish_lines

edges_cache = {}

def copy_edges_from_cache(self, param, **kwargs):
    "copy the static edges from previous sub"
    annots = param['annots']
    for key in ['vanish_line', 'vanish_point']:
        if key not in edges_cache.keys():
            continue
        annots[key] = edges_cache[key]

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
        'c': copy_edges_from_cache,
        'v': copy_edges
    }
    # define visualize
    vis_funcs = [vis_line, plot_skeleton, vis_vanish_lines, plot_text]
    # construct annotations
    annotator = AnnotBase(
        dataset=dataset, 
        key_funcs=key_funcs,
        vis_funcs=vis_funcs,
        step=step)
    annots = annotator.param['annots']
    print(sub)
    annotator.run('X')
    annotator.run('Y')
    annotator.run('Z')
    annotator.run('k')
    if 'K' in annots.keys() and False:
        print('\n'.join([' '.join(['{:7.2f}'.format(i) for i in row]) for row in annots['K']]))
        return 0
    else:
        print('K is not caculated')
    while annotator.isOpen:
        annotator.run()
    for key in ['vanish_line', 'vanish_point']:
        edges_cache[key] = annotator.param['annots'][key]

if __name__ == "__main__":
    from easymocap.annotator import load_parser, parse_parser
    parser = load_parser()
    parser.add_argument('--skip', action='store_true')
    args = parse_parser(parser)
    for sub in args.sub:
        if False:
            from glob import glob
            from os.path import join
            annnames = sorted(glob(join(args.path, args.annot, sub, '*.json')))
            for annname in annnames:
                ann = read_json(annname)
                for key in ['vanish_line', 'vanish_points']:
                    if key in ann.keys():
                        val = ann[key]
                        val = [val[1], val[0], val[2]]
                        ann[key] = val
                save_annot(annname, ann)
        else:
            annot_example(args.path, annot=args.annot, sub=sub)