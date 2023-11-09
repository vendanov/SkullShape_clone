

import os
import vtk
import sys


print(sys.argv)
def_field_file = sys.argv[1]


print(def_field_file)
###load mesh
mesh_in_file = sys.argv[2] #input mesh
mesh_out_file = sys.argv[3] # output mesh
reader = vtk.vtkOBJReader()
reader.SetFileName(mesh_in_file)
reader.Update()
mesh = reader.GetOutput()

##### load deformation field
trans = slicer.util.loadTransform(def_field_file)
loaded_transf = trans.GetTransformToParent()

####transofrm points
points = mesh.GetPoints()
out_points = vtk.vtkPoints()

for i in range(points.GetNumberOfPoints()):
    pt = points.GetPoint(i)
    transf_pt = loaded_transf.TransformFloatPoint(pt)
    out_points.InsertNextPoint(transf_pt)
#### write output
mesh.SetPoints(out_points)
writer = vtk.vtkOBJWriter()
writer.SetFileName(mesh_out_file)
writer.SetInputData(mesh)
writer.Update()



