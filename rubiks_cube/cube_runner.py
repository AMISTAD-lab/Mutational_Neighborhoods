import cube
import numpy as np
import matplotlib.pyplot as plt
import random
import math


def test():
    rubiks_cube = cube.RubiksCube()
    print(rubiks_cube.stringify())
    print(rubiks_cube.get_f_score_based_on_center())
    rubiks_cube.shuffle()
    print(rubiks_cube.get_f_score_based_on_center())

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


def get_high_f_score_cube(num_swaps, solvable=True):
    configuration = cube.RubiksCube()
    if(not solvable):
        #we want to swtich the configuration of the twist
        configuration.cube = [[['w', 'w', 'w'], ['w', 'w', 'w'], ['o', 'w', 'w']], [['o', 'o', 'g'], ['o', 'o', 'o'], ['o', 'o', 'o']], [['w', 'g', 'g'], ['g', 'g', 'g'], ['g', 'g', 'g']], [['r', 'r', 'r'], ['r', 'r', 'r'], ['r', 'r', 'r']], [['b', 'b', 'b'], ['b', 'b', 'b'], ['b', 'b', 'b']], [['y', 'y', 'y'], ['y', 'y', 'y'], ['y', 'y', 'y']]]
    for i in range(num_swaps):
        configuration.do_move(random.randint(0,17))
    return configuration
    
def get_mult_cube_results(num_cubes, random_cube = False, num_splits = 20, depth = 20, plot=True, use_original_f_score=True, allow_repeats=False, solvable=True):
    #if the number of splits are 5, we want to go from 0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0
    results = [[] for i in range(num_splits)]
    results_per_depth = [[] for i in range(num_splits)]

    #we will calculate error as the standard deviation / sqrt (measurements)
    for i in range(num_splits):
        for j in range(depth):
            results_per_depth[i] += [[]]
    
    num_in_each = [0 for i in range(num_splits)]

    size_of_split = 1.0/num_splits
    initial_configuration = None
    number_of_configurations_that_gained_higher_fscore = 0
    for i in range(num_cubes):
        if(random_cube): 
            initial_configuration = cube.RubiksCube()
            initial_configuration.shuffle()
        else:
            initial_configuration = get_high_f_score_cube(random.randint(0,20), solvable=solvable)
    
        
        if(use_original_f_score): initial_f = initial_configuration.get_f_score_based_on_center()
        else: initial_f = initial_configuration.get_f_score()
        f_scores = get_best_f_score_path(initial_configuration=initial_configuration,  depth = depth, break_at_end = False, allow_repeats=allow_repeats)
        if(f_scores[-1] >= initial_f or f_scores[-2] >= initial_f): number_of_configurations_that_gained_higher_fscore+=1
        index = min(num_splits - 1, int(initial_f/size_of_split))
        index = max(index, 0)
        if(results[index] == []):
            results[index] = f_scores
        else:
            results[index] = [results[index][i] + f_scores[i] for i in range(depth)]
        
        for j in range(depth):
            results_per_depth[index][j] += [f_scores[j]]
        num_in_each[index] += 1
    
    if plot: plt.figure()
    variance_of_variances = []
    variance_of_variances_x = []
    for i in range(num_splits):
        if(num_in_each[i] < 5): continue
        data = results[i]
        data = [data[j]/num_in_each[i] for j in range(depth)]
        label = round(i*size_of_split,2)
        y_error = []
        for j in range(depth):
            standard_deviation = np.std(results_per_depth[i][j])
            error = standard_deviation/ math.sqrt(num_in_each[i])
            y_error +=[error]
        if plot: 
            plt.errorbar(x=range(depth), y=data, yerr=y_error, label=str(label))

        variance_of_variances += [np.std(data)]
        variance_of_variances_x += [label]
    print(num_in_each)
    f = open("cube_runner.txt", "a")
    f.write("---------------------------------------------\n")
    f.write("The number of cubes is " + str(num_cubes) + "\n")
    f.write("Getting high cubes by chance instead of going bacwkwards from solved cube " + str(random_cube) + "\n")
    f.write("The number of splits is " + str(num_splits) + "\n")
    f.write("Are we using the original_f_score: " + str(use_original_f_score) + "\n")
    f.write("Are we allowing repeats: " + str(allow_repeats) + "\n")
    f.write("Are we using solvable cubes " + str(solvable)+ "\n")
    std_of_std = np.std(variance_of_variances)
    f.write("The standard deviation of standard deviations is " + str(std_of_std) + "\n")
    average_std = np.average(variance_of_variances)
    f.write("The average standard deviation is " + str(average_std) + "\n")
    percent_gained_higher_f = (number_of_configurations_that_gained_higher_fscore/num_cubes) * 100
    f.write("The number of configurations that gained a higher f_score: " + str(percent_gained_higher_f) + "% \n")
    f.write("")
    f.close()
        
    if plot:
        print("Variance of variances is \n")
        print(variance_of_variances)
        #plt.show()
        plt.legend(loc="lower right")
        plt.xlabel('Number of Moves')
        plt.ylabel('F-score')
        #plt.title("Random Transformator: Solvable Puzzles")
        title = "Rubiks Cube: "
        if(use_original_f_score): title += "Original F-Score, "
        else: title += "Best F-Score, "

        if(allow_repeats): title += "Allow Repeats, "
        else: title += "Repeats, "

        if(solvable): title += "Solvable"
        else: title += "Not Solvable"
        plt.title(title)
        plt.savefig('./plot0.png') 

        #now we will plot the variance over the trap size and see how much it differs
        plt.figure()
        plt.xlabel('Initial F-score of Traps')
        plt.ylabel('Standard Deviation')
        plt.scatter(variance_of_variances_x, variance_of_variances)
        plt.title("Rubiks Cube: Variance over Original F-score - No Repeats in Configurations")
        plt.savefig('./plot0_std.png')




get_mult_cube_results(2000, plot=True, num_splits=10, allow_repeats=True)


def run():
    arr = [False, False, True, True, False, False, True, True]
    arr2 = [False, True, False, True, False, True, False, True]
    arr3 = [False, False, False, False, True, True, True, True]
    for i in range(8):
        get_mult_cube_results(2000, plot=False, use_original_f_score=arr[i], allow_repeats=arr2[i], solvable=arr3[i])


# run()