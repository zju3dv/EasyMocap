import open3d as o3d
from easymocap.config.baseconfig import load_object_from_cmd
from easymocap.mytools.reader import read_smpl
from easymocap.visualize.o3dwrapper import Vector3dVector, Vector3iVector
from easymocap.vis3d.basegui import BaseWindow
from easymocap.mytools.vis_base import generate_colorbar
from easymocap.mytools.file_utils import myarray2string, write_smpl
import open3d.visualization.gui as gui
import os
import numpy as np
class SMPLControl(BaseWindow):
    def __init__(self, cfg_model, opts_model, out) -> None:
        super().__init__(out, panel_rate=0.4)
        self.body_model = load_object_from_cmd(cfg_model, opts_model)
        if args.param_path is not None and os.path.exists(args.param_path):
            from easymocap.mytools.reader import read_smpl
            params = read_smpl(args.param_path)[0]
        else:
            params = self.body_model.init_params(nFrames=1)
        self._params = params
        vertices = self.body_model(return_verts=True, return_tensor=False, **params)    
        joints = self.body_model(return_verts=False, return_tensor=False, return_smpl_joints=True, **self._params)[0]
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = Vector3dVector(vertices[0])
        mesh.triangles = Vector3iVector(self.body_model.faces)
        # 使用blendweight计算颜色
        if True:
            weights = self.body_model.weights.cpu().numpy()
            nJoints = weights.shape[-1]            
            colorbar = np.array(generate_colorbar(nJoints))[:, ::-1]/255.
            colors = weights @ colorbar
            mesh.vertex_colors = Vector3dVector(colors)
            mat = self.get_default_mat(color=[1,1,1,0.5])
        else:
            mat = self.get_default_mat()
        mesh.compute_vertex_normals()
        self.add_geometry('smpl', mesh, mat)
        colorbar = np.array(generate_colorbar(joints.shape[0]))[:, ::-1]/255.
        for nj in range(joints.shape[0]):
            sphere = o3d.geometry.TriangleMesh.create_sphere(
                radius=0.01, resolution=20)
            sphere.translate(joints[nj])
            sphere.compute_vertex_normals()
            sphere.paint_uniform_color(colorbar[nj])
            # self.add_geometry('joint{}'.format(nj), sphere, mat)
            # l = self.scene.add_3d_label(joints[nj], "{}".format(nj))
        self.add_control_smpl(self.panel, self.em)
        self.cnt = 0
    
    def update_smpl(self):
        self._params['id'] = 0
        write_smpl('/tmp/smpl.json', [self._params])
        self._params.pop('id')
        vertices = self.body_model(return_verts=True, return_tensor=False, **self._params)
        joints = self.body_model(return_verts=False, return_tensor=False, return_smpl_joints=True, **self._params)[0]
        for nj in range(joints.shape[0]):
            break
            sphere = o3d.geometry.TriangleMesh.create_sphere(
                radius=0.03, resolution=20)
            sphere.translate(joints[nj])
            self.factory['joint{}'.format(nj)]['geom'].vertices = sphere.vertices
            self.update_geometry('joint{}'.format(nj))
        self.factory['smpl']['geom'].vertices = Vector3dVector(vertices[0])
        self.factory['smpl']['geom'].compute_vertex_normals()
        self.update_geometry('smpl')

    def reset_params(self, ):
        for key, val in self._params.items():
            val[:] = 0.
            for slider in self.slider_dict[key]:
                slider.double_value = 0.
            print('Reset {}'.format(key))
        self.update_smpl()
    
    def export_params(self, ):
        for key, val in self._params.items():
            print('{}: {}'.format(key, myarray2string(self._params[key])))
        self._params['id'] = 0
        write_smpl('/tmp/smpl.json', [self._params])
    
    def read_params(self, filename):
        datas = read_smpl(filename)[0]
        for key in self._params.keys():
            if key in datas.keys():
                self._params[key] = datas[key]
        self.update_smpl()

    def import_params(self, ):
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Choose file to load",
                             self.window.theme)
        dlg.add_filter(
            ".json",
            "SMPL parameters")
        # A file dialog MUST define on_cancel and on_done functions
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_load_dialog_done)
        self.window.show_dialog(dlg)
    
    def _on_file_dialog_cancel(self):
        self.window.close_dialog()

    def _on_load_dialog_done(self, filename):
        self.window.close_dialog()
        self.read_params(filename)
        
    def create_callback(self, key, index):
        def change(value):
            self._params[key][0, index] = value
            self.update_smpl()
            if self.out is not None:
                self.scene.scene.scene.render_to_image(self.capture_callback)
        return change

    def add_control_smpl(self, panel, em):
        # Rh, Th
        print(self._params.keys())
        # add reset
        reset_button = gui.Button("reset")
        reset_button.set_on_clicked(self.reset_params)
        panel.add_child(reset_button)
        import_button = gui.Button("import")
        import_button.set_on_clicked(self.import_params)
        panel.add_child(import_button)        
        export_button = gui.Button("export")
        export_button.set_on_clicked(self.export_params)
        panel.add_child(export_button)
        self.slider_dict = {}
        for key in ['Rh', 'Th']:
            col = gui.CollapsableVert(key, 0.33 * em,
                gui.Margins(em, 0, 0, 0))
            for i in range(self._params[key].shape[1]//3):
                grid = gui.VGrid(5, 10)
                _label = gui.Label(key + '_{}'.format(i))
                grid.add_child(_label)
                self.slider_dict[key] = []
                for nj in range(3):
                    slider = gui.Slider(gui.Slider.DOUBLE)
                    slider.set_limits(-4., 4.)
                    slider.double_value = self._params[key][0, nj+3*i]
                    slider.set_on_value_changed(self.create_callback(key, nj+3*i))
                    self.slider_dict[key].append(slider)
                    grid.add_child(slider)
                col.add_child(grid)
            panel.add_child(col)
        for key in ['shapes', 'expression']:
            if key not in self._params.keys():
                continue
            nShape = self._params[key].shape[-1]
            col = gui.CollapsableVert(key, 0.33 * em,
                gui.Margins(em, 0, 0, 0))
            grid = gui.VGrid(2, 10)
            self.slider_dict[key] = []
            for nj in range(nShape):
                _label = gui.Label(key + '_{}'.format(nj))
                grid.add_child(_label)
                slider = gui.Slider(gui.Slider.DOUBLE)
                slider.set_limits(-4., 4.)
                slider.double_value = self._params[key][0, nj]
                slider.set_on_value_changed(self.create_callback(key, nj))
                self.slider_dict[key].append(slider)
                grid.add_child(slider)
            col.add_child(grid)
            panel.add_child(col)
        for key in ['poses', 'handl', 'handr']:
            if key not in self._params.keys():
                continue
            nShape = self._params[key].shape[-1]//3
            col = gui.CollapsableVert(key, 0.33 * em,
                gui.Margins(em, 0, 0, 0))
            grid = gui.VGrid(4, 10)
            self.slider_dict[key] = []
            for nj in range(nShape):
                _label = gui.Label(key + '_{}'.format(nj))
                grid.add_child(_label)
                for i in range(3):
                    slider = gui.Slider(gui.Slider.DOUBLE)
                    slider.set_limits(-3., 3.)
                    slider.double_value = self._params[key][0, 3*nj+i]
                    slider.set_on_value_changed(self.create_callback(key, 3*nj+i))
                    self.slider_dict[key].append(slider)
                    grid.add_child(slider)
            col.add_child(grid)
            panel.add_child(col)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, 
        default='config/model/smpl_neutral.yml')
    parser.add_argument('--opts', type=str, 
        default=[], nargs="+")
    
    parser.add_argument('--key', type=str, 
        default='poses')
    parser.add_argument('--max', type=float, default=1.57)
    parser.add_argument('--param_path', type=str, default=None)
    parser.add_argument('--out', type=str, default=None)
    parser.add_argument('--num', type=int, default=50)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--one', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    
    gui.Application.instance.initialize()

    w = SMPLControl(args.cfg, args.opts, out=args.out)
    # Run the event loop. This will not return until the last window is closed.
    gui.Application.instance.run()