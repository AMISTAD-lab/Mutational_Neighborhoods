from typing import List, Tuple, Set, Dict, Any, Union
from english_words import get_english_words_set
import matplotlib.pyplot as plt

def f(current: str, target: str):
    '''
    Finds the proportion of correct letters in the current word in comparison to the target word
    '''
    common = list(set(current) & set(target))
    return (float) (len(common) / len(target))


def shortest_chain_len(start: str, target: str, D: Set[str]) -> int:
    f_score = [f(start, target)]
    temp_score = []

    if start == target:
        f_score += [f(start, target)]

        xs = [x for x in range(len(f_score))]

        plt.plot(xs, f_score)
        plt.show()
        plt.close()

        return 0
 
    # Map of intermediate words and the list of original words
    umap: Dict[str, List[str]] = {}
 
    # Initialize umap with empty lists
    for i in range(len(start)):
        intermediate_word = start[:i] + "*" + start[i+1:]
        umap[intermediate_word] = []
 
    # Find all the intermediate words for the words in the given Set
    for word in D:
        for i in range(len(word)):
            intermediate_word = word[:i] + "*" + word[i+1:]
            if intermediate_word not in umap:
                umap[intermediate_word] = []
            umap[intermediate_word].append(word)
 
    # Perform BFS and push (word, distance)
    q = [(start, 1)]
    visited = {start: 1}
    
    i  = 0
    # Traverse until queue is empty
    while (q or i < 30):
        i += 1
        word, dist = q.pop(0)

        # If target word is found
        if word == target:
            print("Distance is ", dist)
            break
 
        # Finding intermediate words for the word in front of queue
        for i in range(len(word)):
            intermediate_word = word[:i] + '*' + word[i+1:]
            vect = umap[intermediate_word]
            for k in range(len(vect)):
               
                # If the word is not visited
                if vect[k] not in visited:
                    visited[vect[k]] = 1
                    q.append((vect[k], dist + 1))

                # Add value to f_score array 
                temp_score += [f(vect[k], target)]
                #print("intermediate word", vect[k])
                #print(target)
                #print(temp_score)
            f_score += [max(temp_score)]
            #print(f_score)
    #print(f_score)
 
    xs = [x for x in range(len(f_score))]

    plt.plot(xs, f_score)
    plt.savefig("plot1.png")
    plt.show()
    plt.close()

    return 0
  
# Make dictionary
D = get_english_words_set(['web2'], lower=True)
start = "cart"
target = "salt"
print(f"Length of shortest chain is: {shortest_chain_len(start, target, D)}")