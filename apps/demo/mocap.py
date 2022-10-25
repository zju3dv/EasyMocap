import os
from os.path import exists
from os.path import join
from easymocap.config import Config, CfgNode
from glob import glob
from easymocap.mytools.debug_utils import run_cmd, check_exists, myerror, log, mywarn

def check_image(path):
    if not check_exists(join(path, 'images')):
        mywarn('Images not found in {}'.format(path))
        if exists(join(path, 'videos')):
            cmd = 'python3 apps/preprocess/extract_image.py {}'.format(path)
            run_cmd(cmd)

def check_camera(path, mode):
    if mode == 'scan':
        return 0
    if not os.path.exists(join(path, 'intri.yml')) or \
        not os.path.exists(join(path, 'extri.yml')):
        myerror('[error] No camera calibration found in {}'.format(path))
        raise FileNotFoundError

def format_subs(subs):
    subs = ', '.join(list(map(lambda x:"'{}'".format(x), subs)))
    subs = f'''"[{subs}]"'''
    return subs

def mocap_demo(path, mode, exp=None):
    # check images
    check_image(path)
    # check camera
    check_camera(path, mode)
    # run triangulation
    if mode in ['object3d']:
        dir_k3d = join(path, 'output-object3d')
    else:
        dir_k3d = join(path, 'output-keypoints3d')
    if not check_exists(join(dir_k3d, 'keypoints3d')) or args.restart_mocap:
        if 'half' in mode:
            cfg_data = 'config/recon/mv1p.yml'
            cfg_exp = 'config/recon/mv1p-half.yml'
        elif mode == 'object3d':
            cfg_data = 'config/recon/mvobj.yml'
            cfg_exp = 'config/recon/tri-mvobj.yml'
        elif mode.startswith('smpl-3d-mp-wild'):
            cfg_data = 'config/recon/mvmp.yml'
            cfg_exp = 'config/recon/mvmp-wild.yml'
        elif args.mp:
            # In this mode, we just perform triangulation on matched 3d keypoints
            cfg_data = 'config/recon/mvmp.yml'
            cfg_exp = 'config/recon/mvmp-match.yml'
        else:
            cfg_data = 'config/recon/mv1p.yml'
            cfg_exp = 'config/recon/mv1p-total.yml'
        opt_data = f'args.path {path} args.out {dir_k3d}'
        if args.subs is not None:
            opt_data += ' args.subs {}'.format(args.subs)
        if args.subs_vis is not None:
            opt_data += ' args.subs_vis {}'.format(format_subs(args.subs_vis))
        if args.disable_visdetec:
            opt_data += ' args.writer.visdetect.enable False'
        if args.vismatch:
            opt_data += ' args.writer.vismatch.enable True'
        if args.disable_visrepro:
            opt_data += ' args.writer.visrepro.enable False'        
        if args.disable_crop:
            opt_data += ' args.writer.vismatch.crop False args.writer.visdetect.crop False '
        if args.ranges is not None:
            opt_data += ' args.ranges {},{},{}'.format(*args.ranges)
        # config for experiment
        opt_exp = ' args.debug {}'.format('True' if args.debug else 'False')
        cmd = 'python3 apps/fit/triangulate1p.py --cfg_data {cfg_data} --opt_data {opt_data} --cfg_exp {cfg_exp} --opt_exp {opt_exp}'.format(
            cfg_data=cfg_data,
            cfg_exp=cfg_exp,
            opt_data=opt_data,
            opt_exp=opt_exp
        )
        run_cmd(cmd)
        # compose videos
        cmd = f'python3 -m easymocap.visualize.ffmpeg_wrapper {dir_k3d}/match --fps 50'
        run_cmd(cmd)
    # TODO: check triangulation
    # run reconstruction
    if mode in ['object3d']:
        return 0
    exp = mode if exp is None else exp
    if not check_exists(join(path, 'output-{}'.format(exp), 'smpl')) or args.restart:
        # load config
        config = config_dict[args.mode]
        cfg_data = config.data
        cfg_model = config.model
        cfg_exp = config.exp
        _config_data = Config.load(cfg_data)

        cmd = f'python3 apps/fit/fit.py --cfg_model {cfg_model} --cfg_data {cfg_data} --cfg_exp {cfg_exp}'

        # opt data
        output = join(path, 'output-{}'.format(exp))
        opt_data = ['args.path', path, 'args.out', output]
        opt_data += args.opt_data
        opt_data += config.get('opt_data', [])
        if 'camera' in _config_data.args.keys():
            opt_data.extend(['args.camera', path])
        if args.ranges is not None:
            opt_data.extend(['args.ranges', '{},{},{}'.format(*args.ranges)])
        if args.subs is not None:
            opt_data.extend(["args.subs", "{}".format(args.subs)])
        if args.disable_vismesh:
            opt_data += ['args.writer.render.enable', 'False']
        if args.vis_scale is not None:
            opt_data += ['args.writer.render.scale', '{}'.format(args.vis_scale)]
        if args.vis_mode is not None:
            opt_data += ['args.writer.render.mode', args.vis_mode]
        if args.pids is not None and args.mp:
            opt_data += ['args.pids', ','.join(map(str, args.pids))]
        cmd += ' --opt_data "{}"'.format('" "'.join(opt_data))
        # opt model
        opt_model = config.get('opt_model', [])
        if len(opt_model) > 0:
            cmd += ' --opt_model "{}"'.format('" "'.join(opt_model))
        # opt exp
        opt_exp = ['args.monitor.printloss', "True"] + args.opt_exp
        opt_exp += config.get('opt_exp', [])
        if len(opt_exp) > 0:
            cmd += ' --opt_exp "{}"'.format('" "'.join(opt_exp))

        log(cmd.replace(output, '${output}').replace(path, '${data}'))
        run_cmd(cmd)
    videoname = join(path, 'output-{}'.format(exp), 'smplmesh.mp4')
    if not exists(videoname) or args.restart:
        cmd = 'python3 -m easymocap.visualize.ffmpeg_wrapper {data}/output-{exp}/smplmesh --fps 50'.format(
            data=path, exp=exp
        )
        run_cmd(cmd)

def mono_demo(path, mode, exp=None):
    check_image(path)
    # check cameras
    if not os.path.exists(join(path, 'intri.yml')):
        cmd = f'python3 apps/calibration/create_blank_camera.py {path}'
        run_cmd(cmd)
    # run reconstruction
    exp = mode if exp is None else exp
    if args.subs is None:
        args.subs = sorted(os.listdir(join(path, 'images')))
    for sub in args.subs:
        outdir = join(path, 'output-{}'.format(exp), 'smplmesh')
        videoname = join(outdir, sub+'.mp4')
        if os.path.exists(videoname) and not args.restart:
            continue
        # load config
        config = config_dict[mode]
        cfg_data = config.data
        cfg_model = config.model
        cfg_exp = config.exp
        cmd = f'python3 apps/fit/fit.py --cfg_model {cfg_model} --cfg_data {cfg_data} --cfg_exp {cfg_exp}'
        _config_data = Config.load(cfg_data)

        # opt data
        output = join(path, 'output-{}'.format(exp))
        opt_data = ['args.path', path, 'args.out', output, 'args.subs', format_subs([sub]).replace('"', '')]
        opt_data += args.opt_data
        opt_data += config.get('opt_data', [])
        if 'camera' in _config_data.args.keys():
            opt_data.extend(['args.camera', path])
        if args.ranges is not None:
            opt_data.extend(['args.ranges', '{},{},{}'.format(*args.ranges)])
        if args.vis_scale is not None:
            opt_data += ['args.writer.render.scale', '{}'.format(args.vis_scale)]
        if args.vis_mode is not None:
            opt_data += ['args.writer.render.mode', args.vis_mode]
        if args.pids is not None and args.mp:
            opt_data += ['args.pids', ','.join(map(str, args.pids))]
        if args.render_side:
            opt_data += ['args.writer.render.mode', "left"]
        cmd += ' --opt_data "{}"'.format('" "'.join(opt_data))
        # opt model
        opt_model = config.get('opt_model', [])
        if len(opt_model) > 0:
            cmd += ' --opt_model "{}"'.format('" "'.join(opt_model))
        # opt exp
        opt_exp = [] + args.opt_exp
        if args.debug:
            opt_exp.extend(['args.monitor.printloss', "True", 'args.monitor.check', 'True'])
        opt_exp += config.get('opt_exp', [])
        if len(opt_exp) > 0:
            cmd += ' --opt_exp "{}"'.format('" "'.join(opt_exp))

        log(cmd.replace(output, '${output}').replace(path, '${data}'))
        run_cmd(cmd)

        cmd = 'python3 -m easymocap.visualize.ffmpeg_wrapper {data}/output-{exp}/smplmesh/{sub} --fps {fps}'.format(
            data=path, exp=exp, sub=sub, fps=30
        )
        run_cmd(cmd)

def run_triangulation(cfg_data, cfg_exp, path, out, args):
    opt_data = f'args.path {path} args.out {out}'
    if args.subs is not None:
        opt_data += ' args.subs "{}"'.format(format_subs(args.subs).replace('"', ''))
    if args.subs_vis is not None:
        opt_data += ' args.subs_vis {}'.format(format_subs(args.subs_vis))
    if args.pids is not None and 'mp' in args.work:
        opt_data += ' args.pids ' + ','.join(map(str, args.pids))
    if args.disable_visdetec:
        opt_data += ' args.writer.visdetect.enable False'
    if args.vismatch:
        opt_data += ' args.writer.vismatch.enable True'
    if args.disable_visrepro:
        opt_data += ' args.writer.visrepro.enable False'
    if args.disable_crop:
        opt_data += ' args.writer.vismatch.crop False args.writer.visdetect.crop False '
    if args.vis_scale is not None:
        opt_data += ' args.writer.visrepro.scale {}'.format(args.vis_scale)
        opt_data += ' args.writer.visdetect.scale {}'.format(args.vis_scale)
        opt_data += ' args.writer.vismatch.scale {}'.format(args.vis_scale)
    if args.ranges is not None:
        opt_data += ' args.ranges {},{},{}'.format(*args.ranges)
    # config for experiment
    opt_exp = ' args.debug {}'.format('True' if args.debug else 'False')
    if args.triangulator_min_views is not None:
        opt_exp += ' args.config.keypoints2d.min_view {view}'.format(view=args.triangulator_min_views)
    cmd = 'python3 apps/fit/triangulate1p.py --cfg_data {cfg_data} --opt_data {opt_data} --cfg_exp {cfg_exp} --opt_exp {opt_exp}'.format(
        cfg_data=cfg_data,
        cfg_exp=cfg_exp,
        opt_data=opt_data,
        opt_exp=opt_exp
    )
    run_cmd(cmd)
    cmd = f'python3 -m easymocap.visualize.ffmpeg_wrapper {out}/match --fps {args.fps}'
    run_cmd(cmd)

def append_mocap_flags(path, output, cfg_data, cfg_model, cfg_exp, config, args):
    cmd = f'python3 apps/fit/fit.py --cfg_model {cfg_model} --cfg_data {cfg_data} --cfg_exp {cfg_exp}'
    _config_data = Config.load(cfg_data)
    # opt data
    opt_data = ['args.path', path, 'args.out', output]
    if args.subs is not None:
        opt_data.extend(['args.subs', format_subs(args.subs).replace('"', '')])
    if args.subs_vis is not None:
        opt_data.extend(['args.subs_vis', format_subs(args.subs_vis).replace('"', '')])
    opt_data += args.opt_data
    opt_data += config.get('opt_data', [])
    if 'camera' in _config_data.args.keys():
        opt_data.extend(['args.camera', path])
    if args.ranges is not None:
        opt_data.extend(['args.ranges', '{},{},{}'.format(*args.ranges)])
    if args.disable_vismesh:
        opt_data += ['args.writer.render.enable', 'False']
    if args.vis_scale is not None:
        opt_data += ['args.writer.render.scale', '{}'.format(args.vis_scale)]
    if args.vis_mode is not None:
        opt_data += ['args.writer.render.mode', args.vis_mode]
    if args.disable_vismesh:
        opt_data += ['args.writer.render.enable', 'False']
    if args.pids is not None:
        opt_data += ['args.pids', ','.join(map(str, args.pids))]
        if len(args.pids) == 1:
            opt_data[-1] += ','
    if args.render_side:
        opt_data += ['args.writer.render.mode', "left"]
    cmd += ' --opt_data "{}"'.format('" "'.join(opt_data))
    # opt model
    opt_model = config.get('opt_model', [])
    if len(opt_model) > 0:
        cmd += ' --opt_model "{}"'.format('" "'.join(opt_model))
    # opt exp
    opt_exp = [] + args.opt_exp
    if args.debug:
        opt_exp.extend(['args.monitor.printloss', "True", 'args.monitor.check', 'True'])

    opt_exp += config.get('opt_exp', [])
    if len(opt_exp) > 0:
        cmd += ' --opt_exp "{}"'.format('" "'.join(opt_exp))
    run_cmd(cmd)
    outdir = join(output, 'smplmesh')
    filenames = os.listdir(outdir)
    filenames_ = [i for i in filenames if i.endswith('.jpg') and os.path.isfile(join(outdir, i))]
    subs_ = [i for i in filenames if os.path.isdir(join(outdir, i))]
    if len(filenames_) == 0:
        # try to find sub-folders
        if args.subs is not None:
            subs_ = args.subs
        for sub in subs_:
            cmd = f'python3 -m easymocap.visualize.ffmpeg_wrapper {output}/smplmesh/{sub} --fps {args.fps}'
            run_cmd(cmd)
    else:
        cmd = f'python3 -m easymocap.visualize.ffmpeg_wrapper {output}/smplmesh --fps {args.fps}'
        run_cmd(cmd)
    return cmd

def workflow(work, args):
    if not os.path.exists(join(args.path, 'images')):
        mywarn('Images not exists, extract it use default setting')
        cmd = f'python3 apps/preprocess/extract_image.py {args.path}'
        run_cmd(cmd)
    workflow_dict = Config.load('config/mocap_workflow.yml')
    for filename in glob(join('config', 'mocap_workflow_*.yml')):
        dict_ = Config.load(filename)
        workflow_dict.update(dict_)
    workflow = workflow_dict[work]
    for key_work in ['subs', 'pids']:
        if key_work in workflow.keys():
            if key_work == 'subs':
                args.subs = workflow[key_work]
            elif key_work == 'pids':
                args.pids = workflow[key_work]
    exp = work if args.exp is None else args.exp
    if 'extract_keypoints' in workflow.keys() and not args.skip_detect:
        if isinstance(workflow['extract_keypoints'], str):
            cmd = workflow['extract_keypoints'].replace('${data}', args.path)
            run_cmd(cmd)
        else:
            pass
    if 'calibration' in workflow.keys() and workflow['calibration'] != 'none':
        cmd = workflow['calibration'].replace('${data}', args.path)
        run_cmd(cmd)
    # check triangulation
    if 'triangulation' in workflow.keys():
        cfg_data = workflow.triangulation.data
        cfg_exp = workflow.triangulation.exp
        out = join(args.path, workflow.triangulation.out)
        # check output
        if not args.restart_mocap and os.path.exists(join(out, 'keypoints3d')) and len(os.listdir(join(out, 'keypoints3d'))) > 10:
            log('[Skip] Triangulation already done, skipping...')
        else:
            run_triangulation(cfg_data, cfg_exp, args.path, out, args)
    if 'fit' in workflow.keys():
        if isinstance(workflow.fit, str):
            workflow.fit = config_dict[workflow.fit]
        cfg_data = workflow.fit.data
        cfg_model = workflow.fit.model
        cfg_exp = workflow.fit.exp
        # check output
        path = args.path
        if 'output' in workflow.keys():
            output = join(args.path, workflow.output)
        else:
            output = join(args.path, 'output-{}'.format(exp))
        append_mocap_flags(path, output, cfg_data, cfg_model, cfg_exp, workflow.fit, args)
    if 'postprocess' in workflow.keys():
        for key, cmd in workflow.postprocess.items():
            cmd = cmd.replace('${data}', args.path).replace('${exp}', args.exp)
            if '${subs_vis}' in cmd:
                cmd = cmd.replace('${subs_vis}', ' '.join(args.subs_vis))
            if '${vis_scale}' in cmd:
                cmd = cmd.replace('${vis_scale}', '{}'.format(args.vis_scale))
            run_cmd(cmd)

if __name__ == '__main__':
    config_dict = Config.load('config/mocap_index.yml')
    modes = list(config_dict.keys())
    usage = '''This script helps you to motion capture from multiple views.

    The following modes are supported:
'''
    for mode, config in config_dict.items():
        usage += '\t{:20s}: {}\n'.format(mode, config.comment)
    import argparse
    parser = argparse.ArgumentParser(usage=usage)
    parser.add_argument('--work', type=str, default=None,
        help='This is the most top abstract of the workflow')
    parser.add_argument('path', type=str)
    parser.add_argument('--num', type=int, default=1)
    parser.add_argument('--fps', type=int, default=50)
    parser.add_argument('--ranges', type=int, default=None, nargs=3)
    parser.add_argument('--pids', type=int, default=None, nargs='+')
    parser.add_argument('--vis_scale', type=float, default=None)
    parser.add_argument('--vis_mode', type=str, default=None)
    parser.add_argument('--subs', type=str, default=None, nargs='+')
    parser.add_argument('--subs_vis', type=str, default=None, nargs='+')
    parser.add_argument('--mode', type=str, default='smpl-3d')
    parser.add_argument('--exp', type=str, default='output-smpl-3d')
    parser.add_argument('--opt_data', type=str, default=[], nargs='+')
    parser.add_argument('--opt_exp', type=str, default=[], nargs='+')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--mono', action='store_true')
    parser.add_argument('--mp', action='store_true', help='use multi-person')
    parser.add_argument('--skip_detect', action='store_true')
    parser.add_argument('--restart_mocap', action='store_true')
    parser.add_argument('--restart', action='store_true')
    parser.add_argument('--bodyonly', action='store_true')
    parser.add_argument('--disable_visdetec', action='store_true')
    parser.add_argument('--vismatch', action='store_true')
    parser.add_argument('--render_side', action='store_true', 
        help='render the mesh on the right')
    parser.add_argument('--disable_visrepro', action='store_true')
    parser.add_argument('--disable_vismesh', action='store_true')
    parser.add_argument('--disable_crop', action='store_true')
    parser.add_argument('--triangulator_min_views', type=int, default=None)
    args = parser.parse_args()

    if args.work is not None:
        workflow(args.work, args)
        exit()
    if args.mono:
        mono_demo(args.path, mode='mono-'+args.mode, exp=args.exp)
    else:
        if args.subs is not None:
            args.subs = format_subs(args.subs)
        mocap_demo(args.path, mode=args.mode, exp=args.exp)