
import os
import numpy as np

DTYPE_MAPPING = {
    "int8": "char",
    "uint8": "uchar",
    "int16": "short",
    "uint16": "ushort",
    "int32": "int",
    "uint32": "uint",
    "float32": "float",
    "float64": "double"
}

class PointCloud:
    def __init__(self, vertices, vcolors=None):
        
        self.vertices = vertices
        self.num_vert = self.vertices.shape[0]
        if self.vertices.shape[1] != 3:
            raise ValueError("The point cloud must have the shape [N, 3]")

        self.vcolors = vcolors

        if self.vcolors is not None:
            if self.num_vert != self.vcolors.shape[0]:
                raise ValueError("There must be as much as vertices as vertex colors.")
            
            if len(self.vcolors.shape) == 1:
                self.vcolors = self.vcolors[:, None]
            
            if self.vcolors.shape[1] != 1 and self.vcolors.shape[1] != 3:
                raise ValueError("The vertex colors must have the shape [N, 1] or [N, 3]")
            
            if self.vcolors.shape[1] == 1:
                self.vcolors = self.vcolors.repeat(3, axis=1)
            

            if f"{vcolors.dtype}" not in DTYPE_MAPPING:
                raise ValueError(f"The vertex color data type {vcolors.dtype} is not supported")
            
            self.color_dtype = DTYPE_MAPPING[f"{vcolors.dtype}"]
        else:
            self.color_dtype = None

        
    def save(self, path):
        with open(path,'w') as pc_file:
            pc_file.write("ply" + os.linesep)
            pc_file.write("format ascii 1.0" + os.linesep)
            pc_file.write(f"element vertex {self.num_vert}" + os.linesep)
            pc_file.write("property float x" + os.linesep)
            pc_file.write("property float y" + os.linesep)
            pc_file.write("property float z" + os.linesep)
            if self.vcolors is not None:
                pc_file.write(f"property {self.color_dtype} red" + os.linesep)
                pc_file.write(f"property {self.color_dtype} green" + os.linesep)
                pc_file.write(f"property {self.color_dtype} blue" + os.linesep)
            pc_file.write("end_header" + os.linesep)

            for i in range(self.num_vert):
                x, y, z = self.vertices[i, :]
                if self.vcolors is not None:
                    r, g, b = self.vcolors[i, :]
                    pc_file.write(f"{x} {y} {z} {r} {g} {b}" + os.linesep)
                else:
                    pc_file.write(f"{x} {y} {z}" + os.linesep)




        
