from typing import Optional

import numpy as np
import vtk
from matplotlib import pyplot as plt

from mouse_skull_pipelines.analysis.mesh_stats import read_vtk_polydata_points_faces
def produce_gif_coh_comparison(ref_mesh: vtk.vtkPolyData,
                               ks_results: list,
                               output_gif_file: str):
    from matplotlib import cm, animation
    from mpl_toolkits.mplot3d import Axes3D

    # Extract points and faces
    print("ref_mesh", ref_mesh)
    pts, faces = read_vtk_polydata_points_faces(ref_mesh)
    print("PTS", pts.shape)
    print("FACES", faces.shape)

    # Color map based on p-value < 0.05
    p_values = np.array([p for _, p in ks_results])

    colors =  [ 0.0 if all(p_values[p,:] > 0.05) else 1.0 for p in range(p_values.shape[0])]
    #
    print( "Ratio of significant p-values", np.sum(colors) / len(colors))
    #colors = np.where(p_values < 0.05, 1.0, 0.0)  # binary coloring
    face_colors = np.array([colors[face[0]] for face in faces])

    fig = plt.figure(figsize=(8, 6), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    cmap = plt.get_cmap('coolwarm')

    norm = plt.Normalize(0, 1)
    trisurf = ax.plot_trisurf(pts[:, 0], pts[:, 1], pts[:, 2],
                               triangles=faces,
                               linewidth=0.2,
                               antialiased=True)
    from matplotlib.colors import ListedColormap


    cmap = ListedColormap(['lightgray', 'crimson'])  # 0 = gray, 1 = red
    norm = plt.Normalize(0, 1)
    print(len(colors))
    print("colors[0]", colors[0])
    trisurf.set_fc(cmap(norm(colors)))

    ax.set_box_aspect([np.ptp(i) for i in [ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]])
    #ax.axis('off')
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, pad=0.1)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['n.s.', 'p < 0.05'])
    cbar.set_label('Kolmogorovâ€“Smirnov p-value threshold')

    def animate(frame):
        ax.view_init(elev=20, azim=frame * 36)

    ani = animation.FuncAnimation(fig, animate, frames=10)
    ani.save(output_gif_file, writer='pillow', fps=1)
    plt.close(fig)


def produce_gif_ttest_comparison(ref_mesh: vtk.vtkPolyData,
                                    t_test_results: list,
                                    output_gif_file: str):
    from matplotlib import cm, animation
    from mpl_toolkits.mplot3d import Axes3D

        # Extract points and faces
    print("ref_mesh", ref_mesh)
    pts, faces = read_vtk_polydata_points_faces(ref_mesh)
    print("PTS", pts.shape)
    print("FACES", faces.shape)

        # Color map based on p-value < 0.05
    p_values = np.array([p.pvalue for p in t_test_results])

    colors =  [ 0.0 if all(p_values[p,:] > 0.05) else 1.0 for p in range(p_values.shape[0])]
        #
    print( "Ratio of significant p-values", np.sum(colors) / len(colors))
    #colors = np.where(p_values < 0.05, 1.0, 0.0)  # binary coloring
    face_colors = np.array([colors[face[0]] for face in faces])

    fig = plt.figure(figsize=(8, 6), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    cmap = plt.get_cmap('coolwarm')

    norm = plt.Normalize(0, 1)
    trisurf = ax.plot_trisurf(pts[:, 0], pts[:, 1], pts[:, 2],
                            triangles=faces,
                            linewidth=0.2,
                            antialiased=True)
    from matplotlib.colors import ListedColormap


    cmap = ListedColormap(['lightgray', 'crimson'])  # 0 = gray, 1 = red
    norm = plt.Normalize(0, 1)
    print(len(colors))
    print("colors[0]", colors[0])
    trisurf.set_fc(cmap(norm(colors)))

    ax.set_box_aspect([np.ptp(i) for i in [ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]])
    #ax.axis('off')
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, pad=0.1)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['n.s.', 'p < 0.05'])
    cbar.set_label("T-test p-value threshold")

    def animate(frame):
        ax.view_init(elev=20, azim=frame * 36)

    ani = animation.FuncAnimation(fig, animate, frames=10)
    ani.save(output_gif_file, writer='pillow', fps=1)
    plt.close(fig)


def produce_gif_mean_diff(ref_mesh: vtk.vtkPolyData,
                          mean_diff: list,
                            output_gif_file: str):
    from matplotlib import cm, animation
    from mpl_toolkits.mplot3d import Axes3D

    mean_norms = np.linalg.norm(mean_diff,axis=1)
    print(mean_norms)

    pts, faces = read_vtk_polydata_points_faces(ref_mesh)

    # ????????? ???? ?????? ?????? (?? ?????)
    face_colors = mean_norms  # one value per face
    p95 = np.percentile(mean_norms, 95)

    # ????????: ??? ???? 95-?? ? ???????????? ? 95-??
    mean_norms_clipped = np.clip(mean_norms, a_min=None, a_max=p95)
    cmap = plt.get_cmap('viridis')
    norm = plt.Normalize(vmin=face_colors.min(), vmax=face_colors.max())

    # ??????? ?????? ? ???
    fig = plt.figure(figsize=(8, 6), dpi=300)
    ax = fig.add_subplot(111, projection='3d')

    trisurf = ax.plot_trisurf(pts[:, 0], pts[:, 1], pts[:, 2],
                              triangles=faces,
                              linewidth=0.2,
                              antialiased=True)
    trisurf.set_facecolors(cmap(norm(face_colors)))

    # ???????? ?????
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array(mean_norms)  # ?? ?????? ??????!
    plt.colorbar(sm, ax=ax)


    # ????????? ?????
    ax.set_box_aspect([np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), np.ptp(pts[:, 2])])
    #ax.axis('off')

    # ???????? ????????
    def animate(frame):
        ax.view_init(elev=20, azim=frame * 36)

    ani = animation.FuncAnimation(fig, animate, frames=10)
    ani.save(output_gif_file, writer='pillow', fps=1)
    plt.close(fig)


def uproduce_gif_std(ref_mesh: vtk.vtkPolyData,
                    std_results: list,
                    output_gif_file: str,
                     upper_limit: Optional[float] = None):

    from matplotlib import cm, animation
    from mpl_toolkits.mplot3d import Axes3D
    std_norms = np.linalg.norm(std_results,axis=1)
    print(std_norms)
    pts, faces = read_vtk_polydata_points_faces(ref_mesh)
    # ????????? ???? ?????? ?????? (?? ?????)
    face_colors = std_norms  # one value per face
    p95 = np.percentile(std_norms, 95)
    # ????????: ??? ???? 95-?? ? ???????????? ? 95-??
    std_norms_clipped = np.clip(std_norms, a_min=None, a_max=p95)
    cmap = plt.get_cmap('viridis')
    if upper_limit is not None:
        norm = plt.Normalize(vmin=0, vmax=upper_limit)
    else:
        norm = plt.Normalize(vmin=0, vmax=face_colors.max())

    # ??????? ?????? ? ???

    fig = plt.figure(figsize=(8, 6), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    trisurf = ax.plot_trisurf(pts[:, 0], pts[:, 1], pts[:, 2],
                                triangles=faces,
                                linewidth=0.2,
                                antialiased=True)
    trisurf.set_facecolors(cmap(norm(face_colors)))
    # ???????? ?????
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array(std_norms)  # ?? ?????? ??????!
    plt.colorbar(sm, ax=ax)
    # ????????? ?????
    ax.set_box_aspect([np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), np.ptp(pts[:, 2])])
    #ax.axis('off')
    # ???????? ????????
    def animate(frame):
        ax.view_init(elev=20, azim=frame * 36)
    ani = animation.FuncAnimation(fig, animate, frames=10)
    ani.save(output_gif_file, writer='pillow', fps=1)
    plt.close(fig)


def produce_gif_scalar_map(ref_mesh: vtk.vtkPolyData,
                           face_values: np.ndarray,
                           output_gif_file: str,
                           cmap_name: str = 'viridis',
                           clip_percentile: Optional[float] = 95.0,
                           max_value: Optional[float] = None,
                           colorbar_label: str = 'Value'):
    from matplotlib import cm, animation
    # Prepare points and faces
    pts, faces = read_vtk_polydata_points_faces(ref_mesh)
    values = np.asarray(face_values).astype(float)
    if max_value is not None:
        vmax = float(max_value)
        values = np.clip(values, a_min=None, a_max=vmax)
    elif clip_percentile is not None:
        vmax = np.percentile(values, clip_percentile)
        values = np.clip(values, a_min=None, a_max=vmax)
    else:
        vmax = float(np.max(values))

    vmin = float(np.min(values)) if np.isfinite(values).all() else 0.0
    if vmax <= vmin:
        vmax = vmin + 1e-6

    fig = plt.figure(figsize=(8, 6), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    trisurf = ax.plot_trisurf(pts[:, 0], pts[:, 1], pts[:, 2],
                              triangles=faces,
                              linewidth=0.2,
                              antialiased=True)
    cmap = plt.get_cmap(cmap_name)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    trisurf.set_facecolors(cmap(norm(values)))
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array(values)
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(colorbar_label)
    ax.set_box_aspect([np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), np.ptp(pts[:, 2])])

    def animate(frame):
        ax.view_init(elev=20, azim=frame * 36)

    from matplotlib import animation as _anim
    ani = _anim.FuncAnimation(fig, animate, frames=10)
    ani.save(output_gif_file, writer='pillow', fps=1)
    plt.close(fig)


