The code explores multiple toy systems: Rubik's Cube, N-puzzle, Wordgame. Our goal is to use multiple configurations of each system and understand if taking mutations can improve the configuration. To score a mutation, we use a $f$ score. 

Each system has at least one specific $f$-score function where $f(x) = 1$ for a configuration $x$ represents the configuration is in the desired state. 


To run the code on either the n-puzzle system or the rubik's cube system you would use the following functions. 

```get_mult_cube_results``` in ```cube_runner.py``` gives you the results for the Rubik's cube.

```get_mult_puzzle_results``` in ```n-puzzle.py``` gives you the results for the N-puzzle. 

Below are the standard configurations for both functions:

```
get_mult_cube_results(num_cubes, random_cube = False, num_splits = 20, depth = 20, plot=True, use_original_f_score=True, allow_repeats=False, solvable=True):
```

```
def get_mult_puzzle_results(num_puzzles, max_iterations = 100000, artificial_start_traps = True, num_splits = 20, depth = 20, plot=True, allow_repeats=True, use_solvable_puzzles=True):
```

```num_cubes``` or ```num_puzzles``` specifies the number of configurations to start out with

```num_splits``` splits all the randomly picked configurations based on their initial f-score. If there are 2 splits, the splits would be from 0-0.5 and 0.5-1.0

```depth``` specifies the number of mutations per configuration 

```allow_repeats``` allows each configuration to pick a repeated configuration. 

```solvable``` lets the user pick either originally solvable or un-solvable configurations 

