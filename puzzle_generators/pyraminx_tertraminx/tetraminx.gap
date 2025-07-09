# /home/username/.bun/bin/bun /home/username/Desktop/cayley/cubing.js/src/bin/puzzle-geometry-bin.ts --gap --verbose --moves R,L,B,U tetraminx
# PuzzleGeometry 0.1 Copyright 2018 Tomas Rokicki.
# 

# Rotations: 12
# Base planes: 4
# Face vertices: 3
# Boundary is Q[1,0.5773502691896258,0.5773502691896258,0.5773502691896258]
# Distances: face 1 edge 1.732050807568877 vertex 2.999999999999999
# Faces is now 6
# Short edge is 1.6329931618554514
# Total stickers is now 24
# Move plane sets: 1,1,1,1
# Cubies: 10
# Cubie orbit sizes 6,4
# PuzzleGeometry 0.1 Copyright 2018 Tomas Rokicki.
# 
M_R:=(5,9,11)(6,10,12)(19,21,20);
M_L:=(1,6,7)(2,5,8)(22,24,23);
M_B:=(3,8,12)(4,7,11)(16,18,17);
M_U:=(1,3,9)(2,4,10)(13,15,14);
Gen:=[
M_R,M_L,M_B,M_U
];
ip:=[[1],[3],[5],[7],[9],[11],[13],[16],[19],[22]];
# Size(Group(Gen));
# Size(Stabilizer(Group(Gen), ip, OnTuplesSets));

