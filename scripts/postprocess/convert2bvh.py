'''
  @ Date: 2020-07-27 16:51:24
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-03-13 21:54:03
  @ FilePath: /EasyMocapRelease/scripts/postprocess/convert2bvh.py
'''
import sys
import bpy
from os.path import join
import math
import numpy as np
from mathutils import Matrix, Vector, Quaternion, Euler

def deg2rad(angle):
    return -np.pi * (angle + 90) / 180.

part_match = {'root': 'root', 'bone_00': 'Pelvis', 'bone_01': 'L_Hip', 'bone_02': 'R_Hip',
              'bone_03': 'Spine1', 'bone_04': 'L_Knee', 'bone_05': 'R_Knee', 'bone_06': 'Spine2',
              'bone_07': 'L_Ankle', 'bone_08': 'R_Ankle', 'bone_09': 'Spine3', 'bone_10': 'L_Foot',
              'bone_11': 'R_Foot', 'bone_12': 'Neck', 'bone_13': 'L_Collar', 'bone_14': 'R_Collar',
              'bone_15': 'Head', 'bone_16': 'L_Shoulder', 'bone_17': 'R_Shoulder', 'bone_18': 'L_Elbow',
              'bone_19': 'R_Elbow', 'bone_20': 'L_Wrist', 'bone_21': 'R_Wrist', 'bone_22': 'L_Hand', 'bone_23': 'R_Hand'}

def init_location(cam, theta, r):
    # Originally, theta is negtivate
    # the center of circle coord is (-1, 0), r is np.random.normal(8, 1)
    x, z = (math.cos(theta) * r, math.sin(theta) * r)
    cam.location = Vector((x, -2, z))
    cam.rotation_euler = Euler((-np.pi / 20, -np.pi / 2 - theta, 0))
    cam.scale = Vector((-1, -1, -1))
    return cam

def init_scene(scene, params, gender='male', angle=0):
    # load fbx model
    bpy.ops.import_scene.fbx(filepath=join(params['smpl_data_folder'], 'basicModel_%s_lbs_10_207_0_v1.0.2.fbx' % gender[0]), axis_forward='-Y', axis_up='-Z', global_scale=100)
    print('success load')
    obname = '%s_avg' % gender[0]
    ob = bpy.data.objects[obname]
    ob.data.use_auto_smooth = False  # autosmooth creates artifacts

    # assign the existing spherical harmonics material
    ob.active_material = bpy.data.materials['Material']

    # delete the default cube (which held the material)
    bpy.ops.object.select_all(action='DESELECT')
    bpy.data.objects['Cube'].select_set(True)
    bpy.ops.object.delete(use_global=False)

    # set camera properties and initial position
    bpy.ops.object.select_all(action='DESELECT')
    cam_ob = bpy.data.objects['Camera']
    scn = bpy.context.scene
    bpy.context.view_layer.objects.active = cam_ob

    th = deg2rad(angle)
    # cam_ob = init_location(cam_ob, th, params['camera_distance'])

    '''
    cam_ob.matrix_world = Matrix(((0., 0., 1, params['camera_distance']+dis),
                                 (0., -1, 0., -1.0),
                                 (-1., 0., 0., 0.),
                                 (0.0, 0.0, 0.0, 1.0)))
    '''
    cam_ob.data.angle = math.radians(60)
    cam_ob.data.lens = 60
    cam_ob.data.clip_start = 0.1
    cam_ob.data.sensor_width = 32

    # setup an empty object in the center which will be the parent of the Camera
    # this allows to easily rotate an object around the origin
    scn.cycles.film_transparent = True
    bpy.context.view_layer.use_pass_vector = True
    bpy.context.view_layer.use_pass_normal = True
    bpy.context.view_layer.use_pass_emit = True
    bpy.context.view_layer.use_pass_material_index = True


    # set render size
    # scn.render.resolution_x = params['resy']
    # scn.render.resolution_y = params['resx']
    scn.render.resolution_percentage = 100
    scn.render.image_settings.file_format = 'PNG'

    # clear existing animation data
    ob.data.shape_keys.animation_data_clear()
    arm_ob = bpy.data.objects['Armature']
    arm_ob.animation_data_clear()

    return(ob, obname, arm_ob, cam_ob)

def setState0():
    for ob in bpy.data.objects.values():
       ob.select_set(False)
    bpy.context.view_layer.objects.active = None

def Rodrigues(rotvec):
    theta = np.linalg.norm(rotvec)
    r = (rotvec/theta).reshape(3, 1) if theta > 0. else rotvec
    cost = np.cos(theta)
    mat = np.asarray([[0, -r[2], r[1]],
                      [r[2], 0, -r[0]],
                      [-r[1], r[0], 0]])
    return(cost*np.eye(3) + (1-cost)*r.dot(r.T) + np.sin(theta)*mat)

def rodrigues2bshapes(pose):
    rod_rots = np.asarray(pose).reshape(24, 3)
    mat_rots = [Rodrigues(rod_rot) for rod_rot in rod_rots]
    bshapes = np.concatenate([(mat_rot - np.eye(3)).ravel()
                              for mat_rot in mat_rots[1:]])
    return(mat_rots, bshapes)

# apply trans pose and shape to character
def apply_trans_pose_shape(trans, pose, shape, ob, arm_ob, obname, scene, cam_ob, frame=None):
    # transform pose into rotation matrices (for pose) and pose blendshapes
    mrots, bsh = rodrigues2bshapes(pose)

    # set the location of the first bone to the translation parameter
    arm_ob.pose.bones[obname+'_Pelvis'].location = trans
    arm_ob.pose.bones[obname+'_root'].location = trans
    arm_ob.pose.bones[obname +'_root'].keyframe_insert('location', frame=frame)
    # set the pose of each bone to the quaternion specified by pose
    for ibone, mrot in enumerate(mrots):
        bone = arm_ob.pose.bones[obname+'_'+part_match['bone_%02d' % ibone]]
        bone.rotation_quaternion = Matrix(mrot).to_quaternion()
        if frame is not None:
            bone.keyframe_insert('rotation_quaternion', frame=frame)
            bone.keyframe_insert('location', frame=frame)

    # apply pose blendshapes
    for ibshape, bshape in enumerate(bsh):
        ob.data.shape_keys.key_blocks['Pose%03d' % ibshape].value = bshape
        if frame is not None:
            ob.data.shape_keys.key_blocks['Pose%03d' % ibshape].keyframe_insert(
                'value', index=-1, frame=frame)

    # apply shape blendshapes
    for ibshape, shape_elem in enumerate(shape):
        ob.data.shape_keys.key_blocks['Shape%03d' % ibshape].value = shape_elem
        if frame is not None:
            ob.data.shape_keys.key_blocks['Shape%03d' % ibshape].keyframe_insert(
                'value', index=-1, frame=frame)
import os
import json

def read_json(path):
    with open(path) as f:
        data = json.load(f)
    return data
    
def read_smpl(outname):
    assert os.path.exists(outname), outname
    datas = read_json(outname)
    outputs = []
    if isinstance(datas, dict):
        datas = datas['annots']
    for data in datas:
        for key in ['Rh', 'Th', 'poses', 'shapes']:
            data[key] = np.array(data[key])
        outputs.append(data)
    return outputs

def merge_params(param_list, share_shape=True):
    output = {}
    for key in ['poses', 'shapes', 'Rh', 'Th', 'expression']:
        if key in param_list[0].keys():
            output[key] = np.vstack([v[key] for v in param_list])
    if share_shape:
        output['shapes'] = output['shapes'].mean(axis=0, keepdims=True)
    return output

def load_motions(path):
    from glob import glob
    filenames = sorted(glob(join(path, '*.json')))
    print(filenames)
    motions = {}
    # for filename in filenames[300:900]:
    for filename in filenames:
        infos = read_smpl(filename)
        for data in infos:
            pid = data['id']
            if pid not in motions.keys():
                motions[pid] = []
            motions[pid].append(data)
    keys = list(motions.keys())
    # BUG: not strictly equal: (Rh, Th, poses) != (Th, (Rh, poses))
    for pid in motions.keys():
        motions[pid] = merge_params(motions[pid])
        motions[pid]['poses'][:, :3] = motions[pid]['Rh']
    return motions
    
def load_smpl_params(datapath):
    motions = load_motions(datapath)
    return motions

def main(params):
    scene = bpy.data.scenes['Scene']

    ob, obname, arm_ob, cam_ob = init_scene(scene, params, params['gender'])
    setState0()
    ob.select_set(True)
    bpy.context.view_layer.objects.active = ob

    # unblocking both the pose and the blendshape limits
    for k in ob.data.shape_keys.key_blocks.keys():
        bpy.data.shape_keys["Key"].key_blocks[k].slider_min = -10
        bpy.data.shape_keys["Key"].key_blocks[k].slider_max = 10
    
    bpy.context.view_layer.objects.active = arm_ob

    motions = load_smpl_params(params['path'])
    for pid, data in motions.items():
        # animation
        arm_ob.animation_data_clear()
        cam_ob.animation_data_clear()
        # load smpl params:
        nFrames = data['poses'].shape[0]
        for frame in range(nFrames):
            print(frame)
            scene.frame_set(frame)
            # apply
            trans = data['Th'][frame]
            shape = data['shapes'][0]
            pose = data['poses'][frame]
            apply_trans_pose_shape(Vector(trans), pose, shape, ob,
                                arm_ob, obname, scene, cam_ob, frame)
            bpy.context.view_layer.update()
        bpy.ops.export_anim.bvh(filepath=join(params['out'], '{}.bvh'.format(pid)), frame_start=0, frame_end=nFrames-1)
    return 0

if __name__ == '__main__':
    try:
        import argparse
        if bpy.app.background:
            parser = argparse.ArgumentParser(
                description='Create keyframed animated skinned SMPL mesh from VIBE output')
            parser.add_argument('path', type=str,
                help='Input file or directory')
            parser.add_argument('--out', dest='out', type=str, required=True,
                help='Output file or directory')
            parser.add_argument('--smpl_data_folder', type=str,
                default='./data/smplx/SMPL_maya',
                help='Output file or directory')
            parser.add_argument('--gender', type=str,
                default='male')
            args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
            print(vars(args))
            main(vars(args))
    except SystemExit as ex:

        if ex.code is None:
            exit_status = 0
        else:
            exit_status = ex.code

        print('Exiting. Exit status: ' + str(exit_status))

        # Only exit to OS when we are not running in Blender GUI
        if bpy.app.background:
            sys.exit(exit_status)
