from scipy.stats import gaussian_kde
import copy
from mimetypes import init 
import matplotlib.pyplot as plt
import numpy as np
import random

class Probability_Tree(object):
    "Generic tree node."
    def __init__(self, f, arr, parent = None, children=None):
        self.f = f
        self.arr = arr
        self.children = []
        self.parent = None
        if children is not None:
            for child in children:
                self.add_child(child)
    def __repr__(self):
        return self.arr
    def add_child(self, node):
        assert isinstance(node, Probability_Tree)
        self.children.append(node)
    def set_parent(self, node):
        assert isinstance(node, Probability_Tree)
        self.parent = node

def f(current_configuration, desired_configuration):
    #f is defined by the number of elements in the correct spot / number of elements in the array
    num_correct = 0

    for i in range(len(desired_configuration)):
        for j in range(len(desired_configuration[i])):
            if(desired_configuration[i][j]  == current_configuration[i][j]): num_correct += 1
    
    return num_correct / (len(desired_configuration) * len(desired_configuration[0]))

def get_high_f_score_configuration_using_random_transformer(num_swaps, desired_configuration):
    col_1 = -1
    row_1 = -1

    col_2 = -1
    row_2 = -1

    new_configuration = copy.deepcopy(desired_configuration)

    for i in range(num_swaps):
        col_1 = random.randint(0, len(new_configuration)-1)
        col_2 = random.randint(0, len(new_configuration)-1)

        row_1 = random.randint(0, len(new_configuration[0])-1)
        row_2 = random.randint(0, len(new_configuration[0])-1)

        #we swap the two positions
        temp = new_configuration[col_1][row_1]
        new_configuration[col_1][row_1] = new_configuration[col_2][row_2]
        new_configuration[col_2][row_2] = temp

    return new_configuration



def get_inv_count(N, configuration_1D):


    inv_count = 0
    for i in range((N*N) - 1):
    
        for j in range(i+1, N*N):
            # count pairs(arr[i], arr[j]) such that i < j but arr[i] > arr[j]
            if (configuration_1D[j] != 0 and configuration_1D[i]  != 0 and configuration_1D[i] > configuration_1D[j]):
                inv_count+=1
        
    
    return inv_count


def get_1D_configuration(configuration):
    np_configuration = np.array(configuration)
    return np_configuration.ravel()

# This function returns true if given
# instance of N*N - 1 puzzle is solvable
def is_solvable(configuration):

    N = len(configuration)
    configuration_1D = get_1D_configuration(configuration)
    inv_count = get_inv_count(N, configuration_1D)
 
    #If grid is odd, return true if inversion count is even.
    if (N % 2 == 1):
        return (inv_count % 2 == 0)
    else: #if N is even   
    
        pos_x, pos_y = find_zero(configuration)
        pos_x = N - pos_x
        if (pos_x % 2 == 1):
            return (inv_count % 2 == 0)
        else:
            return (inv_count & 2 == 1)
    



def get_solvable_configuration(num_swaps, desired_configuration, solvable=True):

    new_configuration = get_high_f_score_configuration_using_random_transformer(num_swaps, desired_configuration)
    i =0
    while(True):
        if(is_solvable(new_configuration) == solvable): break
        new_configuration = get_high_f_score_configuration_using_random_transformer(num_swaps, desired_configuration)
        i +=1
        if (i % 2000 == 0):
            print(str(i) + "we are tryong to get " + str(solvable) + " configuration\n")
    
    return new_configuration




def get_high_f_score_configuration_using_original_transformer(num_swaps, desired_configuration):
    """we are swapping 2 positions by moving the 0 block to a random place"""
    col_1, row_1 = find_zero(desired_configuration)
    di = [-1, 0, 0, 1]
    dj = [0, -1, 1, 0]

    new_configuration = copy.deepcopy(desired_configuration)

    for i in range(num_swaps):
        col_2 = -1
        row_2 = -1
        while(True):
            i = random.randint(0, 4-1)
            col_2 = col_1 + di[i]
            row_2 = row_1 + dj[i]
            if(col_2 < len(new_configuration) and col_2 >= 0 and row_2 < len(new_configuration[0]) and row_2 >= 0): break

        #we swap the two positions
        temp = new_configuration[col_1][row_1]
        new_configuration[col_1][row_1] = new_configuration[col_2][row_2]
        new_configuration[col_2][row_2] = temp

        #here we set col_1 and row_1 to be the place of the current space
        col_1 = col_2
        row_1 = row_2

    return new_configuration





# number of moves for the best configuration, answering the question, do we need to dercrease f -score to get to the desired configuration
# can we plot the f-score leading up to a successful configuration --> will tell us if we need to go in a valley to be sucess
# if we were to follow the path of high_f, how likelky are we to get a sucessful configuration 
def recursive_gram(current_configuration, desired_configuration, loc_space_i, loc_space_j, tree, depth=18):

    sucessful_trips = []
    if(depth <= 0): return
    di = [-1, 0, 0, 1]
    dj = [0, -1, 1, 0]

    for i in range(4):
        new_i = loc_space_i + di[i]
        new_j = loc_space_j + dj[i]

        if(new_i < 0 or new_j < 0 or new_i >= len(current_configuration) or new_j >= len(current_configuration)): continue
        temp_configuration = copy.deepcopy(current_configuration)
        temp_configuration[loc_space_i][loc_space_j] = temp_configuration[new_i][new_j]
        temp_configuration[new_i][new_j] = 0
        new_f = f(temp_configuration, desired_configuration)
        
        if(new_f == 1 or temp_configuration == desired_configuration): 
            #print("\n")
            current_node = tree
            arr = [new_f]
            while(current_node != None):
                arr = [current_node.f] + arr
                current_node = current_node.parent
            
            #print(arr)
            sucessful_trips += [arr]
            continue
        child = Probability_Tree(new_f, temp_configuration)
        child.set_parent(tree)
        tree.add_child(child)
        
        if(not tree.parent or child.arr != tree.parent.arr):
            ret = recursive_gram(temp_configuration, desired_configuration, new_i, new_j, child, depth = depth - 1)
            if(ret): sucessful_trips += ret
    return sucessful_trips


def get_configuration(random = True, side_length = 3):

    list_of_elem = list(range(0,side_length*side_length))
    list_of_elem = list_of_elem[1:] + [list_of_elem[0]]
    if(random): np.random.shuffle(list_of_elem)
    conf = []
    counter = 0
    for i in range(side_length):
        conf_row = []
        for j in range(side_length):
            conf_row += [list_of_elem[counter]]
            counter += 1
        conf += [conf_row]
    
    if not random: return conf
    return conf


def find_zero(configuration):
    loc_space_i = -1
    loc_space_j = -1
    for i in range(len(configuration)):
        for j in range(len(configuration[i])):
            if(configuration[i][j] == 0):
                loc_space_i = i
                loc_space_j = j
                break
    
    return (loc_space_i, loc_space_j)


def get_tuple_configuration(current_configuration):
    return tuple(get_1D_configuration(current_configuration))



def get_best_f_score_path(initial_configuration = None, side_length = 3, depth = 50, break_at_end = True, plot=True, allow_repeats=False):
    # We are following the path of best f-score
    current_configuration = get_configuration(random=True, side_length=side_length)
    if(initial_configuration): current_configuration = initial_configuration
    desired_configuration = get_configuration(random=False, side_length=side_length)

    (loc_space_i, loc_space_j) = find_zero(current_configuration)
    f_scores = []

    visited = {}
    
    for i in range(depth):
        f_scores += [f(current_configuration, desired_configuration)]
        if(f_scores[-1] == 1.0 and break_at_end): break #break if we are at the end
        di = [-1, 0, 0, 1]
        dj = [0, -1, 1, 0]
        temp_f_scores_and_configuration = []
        best_f = 0
        visited[get_tuple_configuration(current_configuration)] = 1

        new_i = -1
        new_j = -1
        for i in range(4):
            new_i = loc_space_i + di[i]
            new_j = loc_space_j + dj[i]


            if(new_i < 0 or new_j < 0 or new_i >= len(current_configuration) or new_j >= len(current_configuration)): continue
            temp_configuration = copy.deepcopy(current_configuration)
            temp_configuration[loc_space_i][loc_space_j] = temp_configuration[new_i][new_j]
            temp_configuration[new_i][new_j] = 0
            if(not allow_repeats): 
                if(get_tuple_configuration(temp_configuration) in visited): continue
            new_f = f(temp_configuration, desired_configuration)
            temp_f_scores_and_configuration += [(new_f, temp_configuration, new_i, new_j)]
            if(new_f > best_f): best_f = new_f
    
        if(new_i == -1): 
            print("Reached the end of search space")
            break
        np.random.shuffle(temp_f_scores_and_configuration)
        for temp in temp_f_scores_and_configuration:    
            if(temp[0]  == best_f):
                current_configuration = temp[1]
                loc_space_i = temp[2]
                loc_space_j = temp[3]
                break

    if(plot):
        plt.plot(f_scores, alpha=1)
        plt.title("8-puzzle: Graph of the path picking best F-score")
        plt.savefig("best_f_score.png")
    
    return f_scores


    
def get_mult_puzzle_results(num_traps, artificial_start_traps = True, num_splits = 20, depth = 20, plot=True, allow_repeats=False, use_solvable_puzzles=True):
    #if the number of splits are 5, we want to go from 0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0
    results = [[] for i in range(num_splits)]
    num_in_each = [0 for i in range(num_splits)]
    print(results)

    size_of_split = 1.0/num_splits
    initial_configuration = get_configuration(random=True, side_length= 3)
    desired_configuration = get_configuration(random=False, side_length=3)
    number_of_configurations_that_gained_higher_fscore = 0
    for i in range(num_traps):
        if(artificial_start_traps):
            #initial_configuration = get_high_f_score_configuration_using_original_transformer(random.randint(0,20), desired_configuration)
            initial_configuration = get_solvable_configuration(random.randint(10,30), desired_configuration, solvable=use_solvable_puzzles)
            initial_f = f(initial_configuration, desired_configuration)
            f_scores = get_best_f_score_path(initial_configuration, side_length = 3, depth = depth, 
                                             break_at_end = False, plot=False, allow_repeats=allow_repeats)
            if(f_scores[-1] >= initial_f or f_scores[-2] >= initial_f): number_of_configurations_that_gained_higher_fscore+=1
            index = min(num_splits - 1, int(initial_f/size_of_split))
            if(results[index] == []):
                results[index] = f_scores
            else:
                results[index] = [results[index][i] + f_scores[i] for i in range(depth)]
            num_in_each[index] += 1
    
    if plot: plt.figure()
    variance_of_variances = []
    variance_of_variances_x = []
    for i in range(num_splits):
        if(num_in_each[i] == 0): continue
        data = results[i]
        data = [data[j]/num_in_each[i] for j in range(depth)]
        label = round(i*size_of_split,2)
        variance_of_variances += [np.std(data)]
        variance_of_variances_x += [label]
        if plot: plt.plot(data, label=str(label))
    
    file = open("n-puzzle.txt", "a")
    file.write("---------------------------------------------\n")
    file.write("The number of cubes is " + str(num_traps) + "\n")
    file.write("Getting high cubes by going bacwkwards from solved configuration " + str(artificial_start_traps) + "\n")
    file.write("The number of splits is " + str(num_splits) + "\n")
    file.write("Solvable Puzzles " + str(use_solvable_puzzles) + "\n")
    file.write("Are we allowing repeats: " + str(allow_repeats) + "\n")
    std_of_std = np.std(variance_of_variances)
    file.write("The standard deviation of standard deviations is " + str(std_of_std) + "\n")
    average_std = np.average(variance_of_variances)
    file.write("The average standard deviation is " + str(average_std) + "\n")
    percent_gained_higher_f = (number_of_configurations_that_gained_higher_fscore/num_traps) * 100
    file.write("The number of configurations that gained a higher f_score: " + str(percent_gained_higher_f) + "% \n")
    file.write("")
    file.close()

    if plot:
        #plt.show()
        plt.legend()
        plt.xlabel('Number of iterations')
        plt.ylabel('F-score')
        #plt.title("Random Transformator: Solvable Puzzles")
        plt.title("Original Transformator: No Repeats in Configurations")
        plt.savefig('./plot5.png') 


def run():
    arr = [False, False, True, True]
    arr2 = [False, True, False, True]
    for i in range(4):
        get_mult_puzzle_results(2000, artificial_start_traps=True, allow_repeats=arr[i], use_solvable_puzzles=arr2[i], plot=False)
        print("Done")



    

def two_gram_recursive(initial_configuration = None):
    # 1 2
    # 0 3
    #
    

    initial_configuration = get_configuration(random=True, side_length= 3)
    desired_configuration = get_configuration(random=False, side_length=3)

    f_init = f(initial_configuration, desired_configuration)

    tree = Probability_Tree(f_init, initial_configuration)
    (loc_space_i, loc_space_j) = find_zero(initial_configuration)
    
    print('The space is starting in ({0}, {1})'.format(loc_space_i, loc_space_j))

    trips = recursive_gram(initial_configuration, desired_configuration, loc_space_i, loc_space_j, tree)
    print("the amount of sucesses is", len(trips))

    counter = 0
    trips = sorted(trips, key = lambda ele : len(ele), reverse = False)
    fig = plt.figure()  
    for trip in trips:
        if(counter > 3): break
        counter += 1
        plt.plot(trip, alpha=0.2)
    plt.title("8-puzzle: Graph of F-score for Multiple Paths")
    plt.savefig("trips2.png")



    
    #we can only move 0, and we want to see the probability tree of the configuration



#two_gram_recursive()
#best_f_score()


# not_solvable_configuration = [[3, 9, 1, 15],
#                     [14, 11, 4, 6],
#                     [13, 0, 10, 12],
#                     [2, 7, 8, 5]]
# print(is_solvable(not_solvable_configuration))

# solvable_configuration = [[6, 13, 7, 10],
#                     [8, 9, 11, 0],
#                     [15, 2, 12, 5],
#                     [14, 3, 1, 4]]

# print(is_solvable(solvable_configuration))

run()