from rubiks_cube import cube

rubiks_cube = cube.RubiksCube()
print(rubiks_cube.stringify())
rubiks_cube.shuffle()
print(rubiks_cube.get_f_score())

