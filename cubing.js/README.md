# cubing.js based scripts

This directory contains some scripts written with cubing.js. They are not designed to be properly working in any setting, being rather some quick-and-dirty tools for the momentarily needs.

## get_puzzles_names.js

Copy this script to `cubing.js` root dir and run:

```bash
bun get_puzzles_names.js
```

A list of all puzzles supported by `cubing.js` should appear


## puzzle-geometry-all-json.ts

Copy this file to `cubing.js/src/bin/`, go to the root `cubing.js` diretory and run:

```bash
src/bin/puzzle-geometry-bin-all-json.ts --gap gaps.json
```

This should generate a json file, where their keys are the puzzles and the values are `gap` (or `ksolve` or elsewhat...) outputs of `puzzle-geometry-bin.ts` script provided by `cubing.js`. All the arguments of `puzzle-geometry-bin.ts` but the puzzle name should be functional and used across all puzzles. 