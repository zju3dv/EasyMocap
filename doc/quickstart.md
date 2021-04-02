<!--
 * @Date: 2021-04-02 11:53:16
 * @Author: Qing Shuai
 * @LastEditors: Qing Shuai
 * @LastEditTime: 2021-04-02 11:53:16
 * @FilePath: /EasyMocapRelease/doc/quickstart.md
-->

We provide an example multiview dataset[[dropbox](https://www.dropbox.com/s/24mb7r921b1g9a7/zju-ls-feng.zip?dl=0)][[BaiduDisk](https://pan.baidu.com/s/1lvAopzYGCic3nauoQXjbPw)(vg1z)], which has 800 frames from 23 synchronized and calibrated cameras. After downloading the dataset, you can run the following example scripts.

```bash
data=path/to/data
out=path/to/output
# 0. extract the video to images
python3 scripts/preprocess/extract_video.py ${data}
# 1. example for skeleton reconstruction
python3 code/demo_mv1pmf_skel.py ${data} --out ${out} --vis_det --vis_repro --undis --sub_vis 1 7 13 19
# 2.1 example for SMPL reconstruction
python3 code/demo_mv1pmf_smpl.py ${data} --out ${out} --end 300 --vis_smpl --undis --sub_vis 1 7 13 19 --gender male
# 2.2 example for SMPL-X reconstruction
python3 code/demo_mv1pmf_smpl.py ${data} --out ${out} --undis --body bodyhandface --sub_vis 1 7 13 19 --start 400 --model smplx --vis_smpl --gender male
# 3.1 example for rendering SMPLX to ${out}/smpl
python3 code/vis_render.py ${data} --out ${out} --skel ${out}/smpl --model smplx --gender male --undis --start 400 --sub_vis 1
# 3.2 example for rendering skeleton of SMPL to ${out}/smplskel
python3 code/vis_render.py ${data} --out ${out} --skel ${out}/smpl --model smplx --gender male --undis --start 400 --sub_vis 1 --type smplskel --body bodyhandface
```