import numpy as np
import vtk






def read_vtk_obj(filename: str):
    reader = vtk.vtkOBJReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()

def center_of_mesh(mesh):
    centerOfMassFilter = vtk.vtkCenterOfMass()
    centerOfMassFilter.SetInputData(mesh)
    centerOfMassFilter.SetUseScalarsAsWeights(False)
    centerOfMassFilter.Update()
    center = centerOfMassFilter.GetCenter()
    return center


def get_mesh_vertices( vtk_mesh_object ):
    points = vtk_mesh_object.GetPoints()
    res_mesh = []
    for i in range(points.GetNumberOfPoints()):
        pt = points.GetPoint(i)
        res_mesh.append(pt)

    return np.array(res_mesh)

def get_mesh_faces( vtk_mesh_object ):
    polygons = vtk_mesh_object.GetPolys()
    faces = []
    # Iterate over the polygons to extract the faces
    polygons.InitTraversal()
    while True:
        faceIds = vtk.vtkIdList()
        if polygons.GetNextCell(faceIds):
            face = [faceIds.GetId(i) for i in range(faceIds.GetNumberOfIds())]
            faces.append(face)
        else:
            break
    return faces


