import argparse
import imageio
import scipy.misc
from models import hmr, SMPL
#import hmr, SMPL
import config, constants
import torch
from torchvision.transforms import Normalize
import numpy as np
#from utils.renderer import Renderer
import os
import pyvista as pv
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True, type=str, help='Path to network checkpoint')
parser.add_argument('--folder_path', required=True, type=str, help='Testing image path')


def size_to_scale(size):
    if size >= 224:
        scale = 0
    elif 128 <= size < 224:
        scale = 1
    elif 64 <= size < 128:
        scale = 2
    elif 40 <= size < 64:
        scale = 3
    else:
        scale = 4
    return scale


def get_render_results(vertices, cam_t, renderer):
    rendered_people_view_1 = renderer.visualize(vertices, cam_t, torch.ones((images.size(0), 3, 224, 224)).long() * 255)
    rendered_people_view_2 = renderer.visualize(vertices, cam_t, torch.ones((images.size(0), 3, 224, 224)).long() * 255,
                                                angle=90, rot_axis=[0, 1, 0])

    return rendered_people_view_1, rendered_people_view_2


if __name__ == '__main__':
    #args = parser.parse_args()
    #folder_path = args.folder_path
    folder_path = r"D:\dev\projects\football_AR\RSC-Net-master\RSC-Net-master\examples"
    #img_path = "D:\dev\projects\football AR\testing\shots\shot1_frames\0.jpg"
    #checkpoint_path = args.checkpoint
    checkpoint_path = r"D:\dev\projects\football_AR\RSC-Net-master\RSC-Net-master\pretrained\RSC-Net.pt"

    normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    hmr_model = hmr(config.SMPL_MEAN_PARAMS)
    checkpoint = torch.load(checkpoint_path)
    hmr_model.load_state_dict(checkpoint, strict=False)
    hmr_model.eval()
    hmr_model.to(device)

    smpl_neutral = SMPL(config.SMPL_MODEL_DIR, create_transl=False).to(device)
    #img_renderer = Renderer(focal_length=constants.FOCAL_LENGTH, img_res=constants.IMG_RES, faces=smpl_neutral.faces)
    
    #fi = []
    joints = []
    image_list = os.listdir(folder_path)

    for image in image_list:
    
        img_path = os.path.join(folder_path,image)
    
        img = imageio.imread(img_path)
        im_size = img.shape[0]
        im_scale = size_to_scale(im_size)
        img_up = scipy.misc.imresize(img, [224, 224])
        img_up = np.transpose(img_up.astype('float32'), (2, 0, 1)) / 255.0
        img_up = normalize_img(torch.from_numpy(img_up).float())
        images = img_up[None].to(device)

        with torch.no_grad():
            pred_rotmat, pred_betas, pred_camera, _ = hmr_model(images, scale=im_scale)
            pred_output = smpl_neutral(betas=pred_betas, body_pose=pred_rotmat[:, 1:],
                                       global_orient=pred_rotmat[:, 0].unsqueeze(1), pose2rot=False)
            #print(pred_output)
            #pred_vertices = pred_output.vertices
            pred_joints = pred_output.joints
            #fi.append(pred_vertices.cpu().numpy())
            joints.append(pred_joints.cpu().numpy())
            #print(pred_vertices.cpu().numpy())
            
            #pred_cam_t = torch.stack([pred_camera[:, 1],
            #                          pred_camera[:, 2],
            #                          2 * constants.FOCAL_LENGTH / (constants.IMG_RES * pred_camera[:, 0] + 1e-9)],
            #                         dim=-1)
            
        #view_1, view_2 = get_render_results(pred_vertices, pred_cam_t, img_renderer)
        #view_1 = view_1[0].permute(1, 2, 0).numpy()
        #view_2 = view_2[0].permute(1, 2, 0).numpy()

        #tmp = img_path.split('.')
        #name_1 = '.'.join(tmp[:-2] + [tmp[-2] + '_view1'] + ['png'])
        #name_2 = '.'.join(tmp[:-2] + [tmp[-2] + '_view2'] + ['png'])

        #imageio.imwrite(name_1, (view_1 * 255).astype(np.uint8))
        #imageio.imwrite(name_2, (view_2 * 255).astype(np.uint8))
    
        name1 = r"D:\dev\projects\football AR\testing\results\shots\view1.jpg"
        name2 = r"D:\dev\projects\football AR\testing\results\shots\view2.jpg"
    
        #imageio.imwrite(name1, (view_1 * 255).astype(np.uint8))
        #imageio.imwrite(name2, (view_2 * 255).astype(np.uint8))











# points = fi[0][0]


# x = []
# y = []
# z = []
# for row in points:
#     x.append(row[0])
#     y.append(row[1])
#     z.append(row[2])
    

# x = np.array(x)
# y = np.array(y)
# z = np.array(z)


# # A nice camera position
# cpos = [(-381.74, -46.02, 216.54), (74.8305, 89.2905, 100.0), (0.23, 0.072, 0.97)]
# point_cloud = pv.PolyData(points)
# point_cloud
# image = point_cloud.plot(volume=True,cpos=cpos,cmap="bone")

# from PIL import Image
# im = Image.fromarray(image)
# im.save(r'C:\Users\Asus\Pictures\Screenshots\airplane.png')



# surf = point_cloud.delaunay_2d()
# surf.plot()
# grid = pv.StructuredGrid(x,y,z)
# grid.plot()

# mesh = pv.StructuredGrid()
# # Set the coordinates from the numpy array
# mesh.points = points
# # set the dimensions
# # mesh.dimensions = [29, 32, 1]

# # and then inspect it!
# mesh.plot(show_edges=True, show_grid=True, cpos="xy")


# plotter = pv.Plotter(off_screen=True)
# plotter.add_mesh(mesh, color="orange")
# plotter.show(screenshot=r'C:\Users\Asus\Pictures\Screenshots\airplane.png')




#cv2.imwrite(r"D:\dev\projects\football_AR\RSC-Net-master\RSC-Net-master\examples\ex.jpg",image)













JOINT_NAMES = [
# 25 OpenPose joints (in the order provided by OpenPose)
'OP Nose',
'OP Neck',
'OP RShoulder',
'OP RElbow',
'OP RWrist',
'OP LShoulder',
'OP LElbow',
'OP LWrist',
'OP MidHip',
'OP RHip',
'OP RKnee',
'OP RAnkle',
'OP LHip',
'OP LKnee',
'OP LAnkle',
'OP REye',
'OP LEye',
'OP REar',
'OP LEar',
'OP LBigToe',
'OP LSmallToe',
'OP LHeel',
'OP RBigToe',
'OP RSmallToe',
'OP RHeel',
# 24 Ground Truth joints (superset of joints from different datasets)
'Right Ankle',
'Right Knee',
'Right Hip',
'Left Hip',
'Left Knee',
'Left Ankle',
'Right Wrist',
'Right Elbow',
'Right Shoulder',
'Left Shoulder',
'Left Elbow',
'Left Wrist',
'Neck (LSP)',
'Top of Head (LSP)',
'Pelvis (MPII)',
'Thorax (MPII)',
'Spine (H36M)',
'Jaw (H36M)',
'Head (H36M)',
'Nose',
'Left Eye',
'Right Eye',
'Left Ear',
'Right Ear'
]

JOINT_IDS = {JOINT_NAMES[i]: i for i in range(len(JOINT_NAMES))}




15,16,17,18,
























k = joints[0][0]



def joints_collector(joints):
    
    joints = joints[0][0]
    final_joints = []
    for i in range(25):
        
        #excluding right eye, left eye, right ear, left ear
        if i not in [15,16,17,18]:
            final_joints.append(joints[i])
        
    return final_joints
    


fin = joints_collector(k)
fin = [joints[24+i] for i in range(1,25)]

cpos = [(-381.74, -46.02, 216.54), (74.8305, 89.2905, 100.0), (0.23, 0.072, 0.97)]
poly = pv.PolyData(fin)
poly["My Labels"] = [f"{i}" for i in range(poly.n_points)]
poly
#image = poly.plot()

plotter = pv.Plotter()
plotter.add_point_labels(poly, "My Labels", point_size=20, font_size=10)
plotter.show()







joints = joints[0][0]
print(joints)
print(len(joints))

# A nice camera position











