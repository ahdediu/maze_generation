# Maze Generation by Random Walks
This document outlines a procedure to generate a maze using the random walk process. It is intended to document the code from generate_maze.py and also as a base for genMaze.tex.

## Formal Definition of a Maze

 $$x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$$


## The Random Walk Algorithm

A random walk is a mathematical formalism that describes a succession of random steps within a mathematical space. Here, we will leverage this principle to create an algorithm that generates mazes.

The steps involved in this process are outlined below:

1. **Initialization**: Select an arbitrary, open cell in the maze surface to start from and designate it as visited. This single node forms our initial tree `T`, i.e., `T = (V', E')` where `V'` consists of the initial cell and `E'` is initially empty. 

2. **Walk construction**: From the current cell, walk randomly to an open neighbouring cell. If the neighbouring cell has not been visited before, add it to `V'` and the line joining the neighbouring cell and the current cell is added to `E'`.

3. **Continuation**: Continually perform the step above until a previously visited cell is encountered.

4. **Unvisited Detection**: In the case that the walk construction comes to a halt, select an unvisited cell in the maze and begin another random walk as described in step 2. 

5. **Walk Addition**: As the new walk is developed, append the nodes and the edges connecting them to the respective `V'` and `E'`.

6. **Interative Process**: Repeat Step 4 and 5 until all cells in the maze have been visited. This step assures that a complete and unbroken graph labyrinth has been created.

This iterative process guarantees that a path exists between every pair of cells in the labyrinth. 