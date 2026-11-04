#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Blender 2.78 驱动：按 JSON 规格渲染一个受控场景状态（阶段二 Step 3）。

与官方 render_images.py 的关系：可见性检查与关系计算直接 import 官方实现
（clevr.pth 已把 image_generation 加入 sys.path），渲染设置逐行对齐；只有
三处不同——
1. 对象来自规格 JSON（形状/颜色/材质/尺寸/位置/旋转全部给定，不随机）；
2. 相机与三盏灯的抖动增量来自规格（官方为 random）；
3. 规格由 src/generation 带种子生成，本驱动只负责渲染。

约束：在 Blender 自带 Python 3.5 内运行，禁止 f-string / dataclass /
math.dist；不 import GraphDelta 代码（render_trial_2026-11-03.md 已知限制 1：
官方脚本无随机种子，正式数据必须由本驱动按规格渲染）。

可见性检查不通过时以退出码 3 结束（规格确定 → 重试结果相同，py 侧登记
失败并剔除该状态，不重试）。用法由 render_edit_pairs.py 调度：

  blender --background -noaudio --python blend_render_driver.py -- \
      --spec spec.json --output_image out.png --output_scene scene.json
"""
import argparse
import json
import os
import sys

import bpy
from mathutils import Vector

import utils
from render_images import check_visibility, compute_all_relationships


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--spec', required=True)
    p.add_argument('--output_image', required=True)
    p.add_argument('--output_scene', required=True)
    p.add_argument('--base_scene_blendfile', default='data/base_scene.blend')
    p.add_argument('--properties_json', default='data/properties.json')
    p.add_argument('--shape_dir', default='data/shapes')
    p.add_argument('--material_dir', default='data/materials')
    p.add_argument('--width', type=int, default=480)
    p.add_argument('--height', type=int, default=320)
    p.add_argument('--render_num_samples', type=int, default=512)
    p.add_argument('--render_tile_size', type=int, default=256)
    p.add_argument('--render_min_bounces', type=int, default=8)
    p.add_argument('--render_max_bounces', type=int, default=8)
    p.add_argument('--min_pixels_per_object', type=int, default=200)
    return p


def render_spec_state(args, spec):
    bpy.ops.wm.open_mainfile(filepath=args.base_scene_blendfile)
    utils.load_materials(args.material_dir)

    render_args = bpy.context.scene.render
    render_args.engine = 'CYCLES'
    render_args.filepath = args.output_image
    render_args.resolution_x = args.width
    render_args.resolution_y = args.height
    render_args.resolution_percentage = 100
    render_args.tile_x = args.render_tile_size
    render_args.tile_y = args.render_tile_size
    bpy.data.worlds['World'].cycles.sample_as_light = True
    bpy.context.scene.cycles.blur_glossy = 2.0
    bpy.context.scene.cycles.samples = args.render_num_samples
    bpy.context.scene.cycles.transparent_min_bounces = args.render_min_bounces
    bpy.context.scene.cycles.transparent_max_bounces = args.render_max_bounces

    scene_struct = {
        'split': 'e4',
        'image_index': 0,
        'image_filename': os.path.basename(args.output_image),
        'objects': [],
        'directions': {},
    }

    # 临时平面只为拿地面法线（官方同款），方向轴算完即删
    bpy.ops.mesh.primitive_plane_add(radius=5)
    plane = bpy.context.object

    # 相机抖动：规格给定增量，只动位置不动朝向（方向轴因此保持常量）
    for i in range(3):
        bpy.data.objects['Camera'].location[i] += spec['camera_jitter'][i]

    camera = bpy.data.objects['Camera']
    plane_normal = plane.data.vertices[0].normal
    cam_behind = camera.matrix_world.to_quaternion() * Vector((0, 0, -1))
    cam_left = camera.matrix_world.to_quaternion() * Vector((-1, 0, 0))
    cam_up = camera.matrix_world.to_quaternion() * Vector((0, 1, 0))
    plane_behind = (cam_behind - cam_behind.project(plane_normal)).normalized()
    plane_left = (cam_left - cam_left.project(plane_normal)).normalized()
    plane_up = cam_up.project(plane_normal).normalized()
    utils.delete_object(plane)
    scene_struct['directions']['behind'] = tuple(plane_behind)
    scene_struct['directions']['front'] = tuple(-plane_behind)
    scene_struct['directions']['left'] = tuple(plane_left)
    scene_struct['directions']['right'] = tuple(-plane_left)
    scene_struct['directions']['above'] = tuple(plane_up)
    scene_struct['directions']['below'] = tuple(-plane_up)

    # 灯光抖动：规格给定增量
    for key, lamp_name in (('key', 'Lamp_Key'), ('fill', 'Lamp_Fill'),
                           ('back', 'Lamp_Back')):
        for i in range(3):
            bpy.data.objects[lamp_name].location[i] += spec['lamp_jitter'][key][i]

    # 对象：全部来自规格（官方 add_random_objects 的受控版；含官方的
    # SmoothCube_v2 逻辑——内部名非 'Cube'，无需 r/sqrt(2) 修正）
    with open(args.properties_json, 'r') as f:
        properties = json.load(f)
    color_name_to_rgba = {}
    for name, rgb in properties['colors'].items():
        color_name_to_rgba[name] = [float(c) / 255.0 for c in rgb] + [1.0]
    # properties.json 的键即输出名（cube/rubber/...），值是 blend 资产名
    material_mapping = properties['materials']
    shape_mapping = properties['shapes']

    blender_objects = []
    for o in spec['objects']:
        r = properties['sizes'][o['size']]
        utils.add_object(args.shape_dir, shape_mapping[o['shape']], r,
                         (o['pos'][0], o['pos'][1]), theta=o['rotation'])
        obj = bpy.context.object
        blender_objects.append(obj)
        utils.add_material(material_mapping[o['material']],
                           Color=color_name_to_rgba[o['color']])
        pixel_coords = utils.get_camera_coords(camera, obj.location)
        scene_struct['objects'].append({
            'id': o['id'],
            'shape': o['shape'],
            'size': o['size'],
            'material': o['material'],
            '3d_coords': tuple(obj.location),
            'rotation': o['rotation'],
            'pixel_coords': pixel_coords,
            'color': o['color'],
        })

    if not check_visibility(blender_objects, args.min_pixels_per_object):
        print('VISIBILITY_CHECK_FAILED')
        sys.exit(3)

    scene_struct['relationships'] = compute_all_relationships(scene_struct)
    bpy.ops.render.render(write_still=True)
    with open(args.output_scene, 'w') as f:
        json.dump(scene_struct, f, indent=2)


def main():
    args = build_parser().parse_args(utils.extract_args())
    with open(args.spec) as f:
        spec = json.load(f)
    render_spec_state(args, spec)


if __name__ == '__main__':
    main()
