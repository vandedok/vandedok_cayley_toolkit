# /home/username/.bun/bin/bun /home/username/Desktop/cayley/cubing.js/src/bin/puzzle-geometry-bin.ts --gap --verbose --moves r,2r,l,2l,b,2b,u,2u pyraminx
# PuzzleGeometry 0.1 Copyright 2018 Tomas Rokicki.
# 

# Rotations: 12
# Base planes: 4
# Face vertices: 3
# Boundary is Q[1,0.5773502691896258,0.5773502691896258,0.5773502691896258]
# Distances: face 1 edge 1.732050807568877 vertex 2.999999999999999
# Faces is now 9
# Short edge is 1.6329931618554514
# Total stickers is now 36
# Move plane sets: 2,2,2,2
# Cubies: 14
# Cubie orbit sizes 6,4,4
# PuzzleGeometry 0.1 Copyright 2018 Tomas Rokicki.
# 
M_r:=(31,33,32);
M_2r:=(5,9,11)(6,10,12)(19,21,20);
M_l:=(34,36,35);
M_2l:=(1,6,7)(2,5,8)(22,24,23);
M_b:=(28,30,29);
M_2b:=(3,8,12)(4,7,11)(16,18,17);
M_u:=(25,27,26);
M_2u:=(1,3,9)(2,4,10)(13,15,14);
Gen:=[
M_r,M_2r,M_l,M_2l,M_b,M_2b,M_u,M_2u
];
ip:=[[1],[3],[5],[7],[9],[11],[13],[16],[19],[22],[25],[28],[31],[34]];
# Size(Group(Gen));
# Size(Stabilizer(Group(Gen), ip, OnTuplesSets));

