from rubiks_cube import cube

rubiks_cube = cube.RubiksCube()
print(rubiks_cube.stringify())
print(rubiks_cube.get_f_score())
rubiks_cube.shuffle()
print(rubiks_cube.get_f_score())

rubiks_cube2 = cube.RubiksCube(state = rubiks_cube.stringify())
print(rubiks_cube2.stringify() == rubiks_cube.stringify())


print(rubiks_cube2.get_f_score_per_rotation())

def check_do_move():
    bot = cube.RubiksCube()
    for i in range(18):
        bot.do_move(i)
        bot.do_move(17-i)
    
    print(bot.get_f_score())

