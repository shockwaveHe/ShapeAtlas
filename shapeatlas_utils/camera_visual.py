# %%
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
# four corner pose

#  *-----*   z
#  |  o  |   x------> x
#  *-----*   |
#            |
#            Y
ul_corner = np.array([-0.75, -0.5, 1])
ur_corner = np.array([ 0.75, -0.5, 1])
bl_corner = np.array([-0.75,  0.5, 1])
br_corner = np.array([ 0.75,  0.5, 1])
x_end = np.array([0.5, 0, 0])
y_end = np.array([0, 0.5, 0])
z_end = np.array([0, 0, 1.5])

origin = np.array([0, 0, 0])
x_n = np.array([1, 0, 0])
y_n = np.array([0, 1, 0])
z_n = np.array([0, 0, 1])

class camera:
    def __init__(self, center_pose, center_orientation, color='purple'):
        self.center_pose = center_pose  # center pose in the world
        self.center_orientation = center_orientation  # orientation in the world
        
        self.ul_w = self.center_orientation @ ul_corner + center_pose
        self.ur_w = self.center_orientation @ ur_corner + center_pose
        self.bl_w = self.center_orientation @ bl_corner + center_pose
        self.br_w = self.center_orientation @ br_corner + center_pose

        self.x_end = self.center_orientation @ x_end + center_pose
        self.y_end = self.center_orientation @ y_end + center_pose
        self.z_end = self.center_orientation @ z_end + center_pose
        self.color = color
    
    def set_color(self, color):
        self.color = color
        
    def plot_line(self, ax, p1, p2, color_='purple'):
        
        xvalues = [p1[0], p2[0]]
        yvalues = [p1[1], p2[1]]
        zvalues = [p1[2], p2[2]]

        ax.plot(xvalues, yvalues, zvalues, color=color_)

    def plot_camera(self, ax):
        self.plot_line(ax, self.ul_w, self.ur_w,"blue")
        self.plot_line(ax, self.ul_w, self.bl_w,"green")
        self.plot_line(ax, self.bl_w, self.br_w,self.color)
        self.plot_line(ax, self.br_w, self.ur_w,self.color)
        self.plot_line(ax, self.ul_w, self.center_pose,self.color)
        self.plot_line(ax, self.ur_w, self.center_pose,self.color)
        self.plot_line(ax, self.bl_w, self.center_pose,self.color)
        self.plot_line(ax, self.br_w, self.center_pose,self.color)
    
    def plot_axis(self, ax):

        self.plot_line(ax, self.center_pose, self.x_end, color_='blue')
        ax.scatter([self.x_end[0]],[self.x_end[1]],[self.x_end[2]], marker=">", color="blue", s=20)

        self.plot_line(ax, self.center_pose, self.y_end, color_='green')
        ax.scatter([self.y_end[0]],[self.y_end[1]],[self.y_end[2]], marker=">", color="green", s=20)

        self.plot_line(ax, self.center_pose, self.z_end, color_='red')
        ax.scatter([self.z_end[0]],[self.z_end[1]],[self.z_end[2]], marker=">", color="red", s=20)


class Trajcetory:
    def __init__(self, color="black", marker="o", line="solid", lsize=10):
        self.color = color
        self.marker = marker
        self.line = line
        self.lsize = lsize
        self.xs = []
        self.ys = []
        self.zs = []
    
    def expand(self, pose):
        self.xs.append(pose[0])
        self.ys.append(pose[1])
        self.zs.append(pose[2])
    
    def plot(self, ax):
        ax.plot(self.xs, self.ys, self.zs, color=self.color)

if __name__ == "__main__":

    no_orientation = R.from_dcm([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]])
    center_pose0 = np.array([1,0,0])
    center_pose1 = np.array([0,1,0])
    center_orientation = R.from_dcm([[0, -1, 0],
                    [1, 0, 0],
                    [0, 0, 1]])

    # %%

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ar1 = [0, 1]
    ar2 = [0, 0]

    ax.set_xticks(np.arange(-5, 10, 1))
    ax.set_yticks(np.arange(-5, 10, 1))
    ax.set_zticks(np.arange(-5, 10, 1))
    ax.set_xlim([-5, 10])
    ax.set_ylim([-5, 10])
    ax.set_zlim([-5, 10])
    plt.plot(ar1, ar2, ar2, color="blue")
    plt.plot(ar2, ar1, ar2, color="green")
    plt.plot(ar2, ar2, ar1, color="red")
    ax.scatter([0], [0], [0], marker="o", color="black", s=20)
    ax.scatter([1], [0], [0], marker=">", color="blue", s=20)
    ax.scatter([0], [1], [0], marker=">", color="green", s=20)
    ax.scatter([0], [0], [1], marker=">", color="red", s=20)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    initial_t = np.array([1, 1, 1])
    initial_R = np.array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]])

    traj = Trajcetory()

    # %%
    cam1 = camera(initial_t, initial_R)
    traj.expand(cam1.center_pose)

    dt1 = np.array([2,3,1])
    # dq1 =  R.from_euler('zyx', [
    # [90, 0, 0],
    # [0, 45, 0],
    # [0, 0, 30]], degrees=True)
    dq1 =  R.from_euler('zyx', 
    [90, 45, 30], degrees=True)

    t1 = dq1.as_dcm() @ initial_t + dt1
    R1 = dq1.as_dcm() @ initial_R
    print(dq1.as_euler('zyx', degrees=True))

    cam2 = camera(t1, R1)
    traj.expand(cam2.center_pose)

    dt2 = np.array([2, 1, 3])
    # dq2 =  R.from_euler('zyx', [
    # [-90, 0, 0],
    # [0, -45, 0],
    # [0, 0, -30]], degrees=True)

    dq2 =  R.from_euler('zyx', 
    [-90, -45, -30], degrees=True)

    print(dq2.as_euler('zyx', degrees=True))
    print(dq2.as_dcm())

    # %%
    t2 = dq2.as_dcm() @ t1 + dt2
    R2 = dq2.as_dcm() @ R1
    cam3 = camera(t2, R2)
    traj.expand(cam3.center_pose)

    cam1.plot_camera(ax)
    cam1.plot_axis(ax)

    cam2.plot_camera(ax)
    cam2.plot_axis(ax)

    cam3.plot_camera(ax)
    cam3.plot_axis(ax)

    traj.plot(ax)
    # plot trajectory

    # %%

    r_c_w = R.from_euler('zyx', [45, 90, 60], degrees=True)

    t_c_w = np.array([6, 9, 6])

    # print(r_c_w.as_dcm()) 
    # plot new axis
    new_origin = r_c_w.as_dcm() @ origin + t_c_w

    new_x_n = r_c_w.as_dcm() @ x_n + t_c_w
    new_y_n = r_c_w.as_dcm() @ y_n + t_c_w
    new_z_n = r_c_w.as_dcm() @ z_n + t_c_w

    print(new_origin)
    print(new_x_n)
    print(new_y_n)
    print(new_z_n)

    plt.plot([new_origin[0], new_x_n[0]],[new_origin[1], new_x_n[1]], [new_origin[2], new_x_n[2]], "blue" )
    plt.plot([new_origin[0], new_y_n[0]],[new_origin[1], new_y_n[1]], [new_origin[2], new_y_n[2]], "green" )
    plt.plot([new_origin[0], new_z_n[0]],[new_origin[1], new_z_n[1]], [new_origin[2], new_z_n[2]], "red" )
    # %%

    plt.show()
    # %%
    # calculate relative pose


