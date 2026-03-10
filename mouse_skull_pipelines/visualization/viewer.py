from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional, Tuple

import vtk

from mouse_skull_pipelines.io.discovery import collect_registered_meshes
from mouse_skull_pipelines.scripts.gpa import compute_mean_mesh, gpa_new
def _make_polydata_actor(polydata: vtk.vtkPolyData,
                         color: Tuple[float, float, float],
                         opacity: float,
                         wireframe: bool = False) -> vtk.vtkActor:
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    mapper.ScalarVisibilityOff()
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    prop = actor.GetProperty()
    prop.SetColor(*color)
    prop.SetOpacity(opacity)
    if wireframe:
        prop.SetRepresentationToWireframe()
        prop.SetLineWidth(1.5)
    else:
        prop.SetRepresentationToSurface()
    return actor


def launch_procrustes_viewer(subject_name: str,
                             subject_mesh: vtk.vtkPolyData,
                             mean_mesh: Optional[vtk.vtkPolyData],
                             peer_meshes: Dict[str, vtk.vtkPolyData]) -> None:
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0.08, 0.08, 0.1)

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetWindowName(f"Procrustes viewer - {subject_name}")
    render_window.SetSize(1200, 900)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

    subject_actor = _make_polydata_actor(subject_mesh, (0.88, 0.36, 0.30), 0.9)
    renderer.AddActor(subject_actor)

    mean_actor = None
    if mean_mesh is not None:
        mean_actor = _make_polydata_actor(mean_mesh, (0.25, 0.65, 1.0), 0.35)
        renderer.AddActor(mean_actor)

    peer_order = list(peer_meshes.keys())
    peer_actors = []
    for name in peer_order:
        actor = _make_polydata_actor(peer_meshes[name], (0.25, 0.9, 0.55), 0.3)
        actor.VisibilityOff()
        renderer.AddActor(actor)
        peer_actors.append(actor)

    text_actor = vtk.vtkTextActor()
    text_actor.SetPosition(10, 10)
    text_prop = text_actor.GetTextProperty()
    text_prop.SetFontSize(14)
    text_prop.SetColor(1.0, 1.0, 1.0)
    text_prop.SetFontFamilyToArial()
    renderer.AddActor2D(text_actor)

    state = {
        "subject_visible": True,
        "mean_visible": mean_actor is not None,
        "peer_visible": False,
        "peer_index": 0,
        "wireframe": False,
    }

    def update_peer_visibility():
        if not peer_actors:
            return
        for idx, actor in enumerate(peer_actors):
            actor.SetVisibility(state["peer_visible"] and idx == state["peer_index"])

    def rebuild_text():
        lines = [
            f"Subject: {subject_name} ({'visible' if state['subject_visible'] else 'hidden'})"
        ]
        if mean_actor is not None:
            lines.append(f"Mean: {'visible' if state['mean_visible'] else 'hidden'}")
        if peer_order:
            if state["peer_visible"]:
                current_peer = peer_order[state["peer_index"]]
                lines.append(f"Peer: {current_peer}")
            else:
                lines.append("Peer: hidden")
        lines.append("")
        lines.append("Keys: s toggle subject, m toggle mean, space toggle peer, n/p cycle peer, w wireframe, q quit")
        text_actor.SetInput("\n".join(lines))

    def set_subject_wireframe(enabled: bool):
        prop = subject_actor.GetProperty()
        if enabled:
            prop.SetRepresentationToWireframe()
            prop.SetLineWidth(1.5)
        else:
            prop.SetRepresentationToSurface()
            prop.SetLineWidth(1.0)

    def on_key_press(obj, _event):
        key = obj.GetKeySym()
        if key == "s":
            state["subject_visible"] = not state["subject_visible"]
            subject_actor.SetVisibility(state["subject_visible"])
        elif key == "m" and mean_actor is not None:
            state["mean_visible"] = not state["mean_visible"]
            mean_actor.SetVisibility(state["mean_visible"])
        elif key in ("space", "v"):
            if peer_actors:
                state["peer_visible"] = not state["peer_visible"]
                update_peer_visibility()
        elif key in ("n", "Right"):
            if peer_actors:
                state["peer_index"] = (state["peer_index"] + 1) % len(peer_actors)
                state["peer_visible"] = True
                update_peer_visibility()
        elif key in ("p", "Left"):
            if peer_actors:
                state["peer_index"] = (state["peer_index"] - 1) % len(peer_actors)
                state["peer_visible"] = True
                update_peer_visibility()
        elif key == "w":
            state["wireframe"] = not state["wireframe"]
            set_subject_wireframe(state["wireframe"])
        elif key in ("q", "Escape"):
            render_window.Finalize()
            obj.TerminateApp()
            return
        else:
            return
        rebuild_text()
        render_window.Render()

    interactor.AddObserver("KeyPressEvent", on_key_press)
    rebuild_text()
    set_subject_wireframe(False)
    renderer.ResetCamera()
    render_window.Render()
    interactor.Initialize()
    interactor.Start()


def visualize_subject_interactive(subject_path: Path,
                                  reference_cohort: Path) -> None:
    """
    Launch an interactive VTK viewer for a single subject against a reference cohort.

    subject_path:
        Path to a subject directory or directly to its head_affine_registered.obj file.
    reference_cohort:
        Path to a cohort directory that already contains registered meshes.
    """
    subject_path = Path(subject_path)
    reference_cohort = Path(reference_cohort)
    if not reference_cohort.exists():
        raise FileNotFoundError(f"Reference cohort directory not found: {reference_cohort}")

    if subject_path.is_file():
        if subject_path.name != 'head_affine_registered.obj':
            raise ValueError('Subject file must be head_affine_registered.obj if a file path is provided.')
        subject_dir = subject_path.parent
    else:
        subject_dir = subject_path
        subject_path = subject_dir / 'head_affine_registered.obj'

    if not subject_path.exists():
        raise FileNotFoundError(f"Subject mesh not found: {subject_path}")

    subject_name = subject_dir.name

    ref_meshes = collect_registered_meshes(reference_cohort)

    gpa_input: "OrderedDict[str, Path]" = OrderedDict()
    for name, mesh_path in ref_meshes.items():
        gpa_input[f"ref::{name}"] = mesh_path
    subject_key = f"query::{subject_name}"
    gpa_input[subject_key] = subject_path

    print(f"Running Procrustes alignment against {len(ref_meshes)} reference meshes...")
    aligned_meshes = gpa_new(gpa_input, modes='rigid')

    print("Finished alignment. Collecting meshes for visualization...")

    aligned_subject = aligned_meshes[subject_key]
    aligned_reference = OrderedDict(
        (key, aligned_meshes[key]) for key in aligned_meshes.keys() if key.startswith('ref::')
    )
    print(f"Aligned subject: {aligned_subject.GetNumberOfPoints()} points, {aligned_subject.GetNumberOfCells()} cells")

    
    mean_mesh = compute_mean_mesh(aligned_reference) if aligned_reference else None
    print("Computed mean mesh.")
    peer_meshes: "OrderedDict[str, vtk.vtkPolyData]" = OrderedDict(
        (key.split('::', 1)[1], aligned_reference[key]) for key in aligned_reference.keys()
    )

    print("Launching interactive viewer. Close the window or press 'q' to exit.")
    launch_procrustes_viewer(subject_name, aligned_subject, mean_mesh, peer_meshes)





