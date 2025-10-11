import sys
import slicer

def load_model_ras(path):
    # Prefer the generic loader that accepts properties in all recent Slicer builds
    try:
        success, node = slicer.util.loadNodeFromFile(
            path,
            "ModelFile",
            {"coordinateSystem": slicer.vtkMRMLStorageNode.RAS},
            returnNode=True,
        )
    except TypeError:
        # Fallback for very old builds: load without properties (models assume RAS anyway)
        success, node = slicer.util.loadModel(path, returnNode=True)
    if not success or node is None:
        raise RuntimeError(f"Failed to load model: {path}")
    return node

def load_transform(path):
    # Generic + compatible
    success, node = slicer.util.loadNodeFromFile(path, "TransformFile", {}, returnNode=True)
    if not success or node is None:
        raise RuntimeError(f"Failed to load transform: {path}")
    return node

def apply_transform(node, transform_path):
    tnode = load_transform(transform_path)
    node.SetAndObserveTransformNodeID(tnode.GetID())
    slicer.vtkSlicerTransformLogic().hardenTransform(node)
    slicer.mrmlScene.RemoveNode(tnode)

def main():
    if len(sys.argv) < 4:
        raise SystemExit("Usage: convert_mesh.py <mesh_in> <mesh_out> <transform1> [<transform2> ...]")

    mesh_in = sys.argv[1]
    mesh_out = sys.argv[2]
    transform_paths = [p for p in sys.argv[3:] if p and p.lower() != "none"]

    if not transform_paths:
        raise RuntimeError("At least one transform path must be provided.")

    model_node = load_model_ras(mesh_in)
    print(f"Loaded model: {mesh_in}")

    for transform_path in transform_paths:
        print(f"Applying transform: {transform_path}")
        apply_transform(model_node, transform_path)

    if not slicer.util.saveNode(model_node, mesh_out):
        raise RuntimeError(f"Failed to save transformed mesh to {mesh_out}")
    print(f"Saved: {mesh_out}")

if __name__ == "__main__":
    try:
        main()
    finally:
        slicer.app.quit()
