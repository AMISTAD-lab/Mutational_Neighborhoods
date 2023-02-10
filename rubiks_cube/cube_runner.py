import cube
import numpy as np
import matplotlib.pyplot as plt
import random


def test():
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
    bot.shuffle()
    for i in range(18):
        bot.do_move(i)
        bot.do_inverse_move(i)
        print(bot.stringify())
    

def get_best_f_score_path(initial_configuration = None,  depth = 50, break_at_end = True, allow_repeats=True):
    # also named "best_f_score" We are following the path of best f-score
    current_configuration = cube.RubiksCube()
    current_configuration.shuffle()
    if(initial_configuration): current_configuration = initial_configuration

    f_scores = []
    visited = {}
    for i in range(depth):
        f_scores += [current_configuration.get_f_score()]
        if(f_scores[-1] == 1.0 and break_at_end): break #break if we are at the end
        temp_f_scores_and_configuration = []
        best_f = 0
        visited[(current_configuration).stringify()] = 1
        new_i = -1
        new_j = -1
        for i in range(18):
            current_configuration.do_move(i)
            if(not allow_repeats): 
                if(current_configuration.stringify() in visited): 
                    continue
            new_f = current_configuration.get_f_score()
            temp_f_scores_and_configuration += [(new_f, i)]
            if(new_f > best_f): best_f = new_f
            current_configuration.do_inverse_move(i)
        if(len(temp_f_scores_and_configuration) == 0): 
            break
        
        np.random.shuffle(temp_f_scores_and_configuration)
        for temp in temp_f_scores_and_configuration:    
            if(temp[0]  == best_f):
                current_configuration.do_move(temp[1])
                break
    return f_scores


def get_high_f_score_cube(num_swaps):
    configuration = cube.RubiksCube()
    for i in range(num_swaps):
        configuration.do_move(random.randint(0,17))
    return configuration
    
def get_mult_cube_results(num_cubes, random_cube = False, num_splits = 5, depth = 20, plot=True):
    #if the number of splits are 5, we want to go from 0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0
    results = [[] for i in range(num_splits)]
    num_in_each = [0 for i in range(num_splits)]
    print(results)

    size_of_split = 1.0/num_splits
    initial_configuration = None
    for i in range(num_cubes):
        if(random_cube): 
            initial_configuration = cube.RubiksCube()
            initial_configuration.shuffle()
        else:
            initial_configuration = get_high_f_score_cube(random.randint(0,20))
    
        initial_f = initial_configuration.get_f_score()
        f_scores = get_best_f_score_path(initial_configuration=initial_configuration,  depth = depth, break_at_end = False)
        index = min(num_splits - 1, int(initial_f/size_of_split))
        print(index)
        if(results[index] == []):
            results[index] = f_scores
        else:
            results[index] = [results[index][i] + f_scores[i] for i in range(depth)]
        num_in_each[index] += 1
    
    
    if(plot):
        for i in range(num_splits):
            if(num_in_each[i] == 0): continue
            data = results[i]
            print("before", data)
            data = [data[j]/num_in_each[i] for j in range(depth)]
            print(data)
            label = round(i*size_of_split,2)
            plt.plot(data, label=str(label))
        #plt.show()
        plt.legend()
        plt.xlabel('Number of iterations')
        plt.ylabel('F-score')
        #plt.title("Random Transformator: Solvable Puzzles")
        plt.title("Rubiks Cube: Best-F-score - Repeats in Configurations")
        plt.savefig('./plot0.png') 


def run():
    get_mult_cube_results(100)


run()