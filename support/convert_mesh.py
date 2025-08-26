

import os
import vtk
import sys
import slicer

print(sys.argv)
def_field_file = sys.argv[1]


print(def_field_file)
###load mesh
mesh_in_file = sys.argv[2] #input mesh
mesh_out_file = sys.argv[3] # output mesh

model_node = slicer.util.loadModel(mesh_in_file)

##### load deformation field
trans = slicer.util.loadTransform(def_field_file)

####transofrm points
print (model_node.SetAndObserveTransformNodeID(trans.GetID()))

slicer.vtkSlicerTransformLogic().hardenTransform(model_node)

slicer.util.saveNode(model_node, mesh_out_file)



