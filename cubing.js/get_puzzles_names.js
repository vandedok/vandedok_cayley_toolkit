import { getPG3DNamedPuzzles } from "cubing/puzzle-geometry";

const puzzles = getPG3DNamedPuzzles();
const puzzleNames = Object.keys(puzzles);

// Print all puzzle names in one line, space-separated
console.log(puzzleNames.join("\n"));