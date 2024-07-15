import math
from collections import deque
from typing import List, Tuple

import pandas as pd
from matplotlib import pyplot as plt

import random
import networkx as nx
from matplotlib.patches import Circle

''' Initial walls: mxnx2 (L shape walls for each cell)+n+n 
	to close the top and the right borders. 
	Now each edge can be considered as visiting one cell minus the last one 
	that visits 2 cells. so mxm-1 edges.  '''


class Maze:
	def __init__(self, m, n, wall_chance=0.5):
		self.m = m
		self.n = n
		self.wall_chance = wall_chance
		self.graph = nx.Graph()
		self.visited = [[False] * n for _ in range(m)]
		self.visited[0][0] = True
		self.graph.add_node((0, 0))
		self.begin_node = (0, 0)
		self.end_node = (m-1, n-1)
		self.directions = 6
		self.scale_factor = 100
		self.loops = []
		self.make_maze() #it modifies the end_node




	def make_maze(self):
		self.end_node = self.random_walk_maze((0, 0))

	def random_walk_maze(self, start):


		# dx, dy = [0, 1, 0, -1], [1, 0, -1, 0]  # directions to get the four neighbors
		dx = [round(self.scale_factor*math.cos(i * (2 * math.pi / self.directions))) for i in range(self.directions)]
		dy = [round(self.scale_factor*math.sin(i * (2 * math.pi / self.directions))) for i in range(self.directions)]


		stack = [start]
		node_at_max_depth = None
		max_depth = 0
		visited = {(start): True}

		while stack:
			current = stack[-1]
			if len(stack) > max_depth:
				max_depth = len(stack)
				node_at_max_depth = current
			# Get list of unvisited neighbors
			neighbors = [(current[0] + dx[i], current[1] + dy[i]) for i in range(self.directions) if
						 0 <= current[0] + dx[i] < self.m*self.scale_factor and
						 0 <= current[1] + dy[i] < self.n*self.scale_factor and
						 (current[0] + dx[i], current[1] + dy[i]) not in visited]

			if neighbors:
				# Choose a random neighboring cell and mark as visited
				next_cell = random.choice(neighbors)
				visited[(next_cell)] = True
				stack.append(next_cell)

				# remove the wall between cells
				self.graph.add_edge(current, next_cell)
			else:
				# Additional branching can occur here with 5% chance
				branching_chance = 0.
				if random.random() < branching_chance and len(self.graph[current]) < self.directions-1:
					# Find all neighbours that haven't reached maximum branching factor
					branching_candidates = [(current[0] + dx[i], current[1] + dy[i]) for i in range(self.directions) if
											0 <= current[0] + dx[i] < self.m * self.scale_factor and
											0 <= current[1] + dy[i] < self.n * self.scale_factor and
											len(self.graph[(current[0] + dx[i], current[1] + dy[i])]) < self.directions -1]
					if branching_candidates:
						next_cell = random.choice(branching_candidates)
						self.graph.add_edge(current, next_cell)
						self.loops.append((next_cell, current))
				# Backtrack
				stack.pop()
		return node_at_max_depth

	def _dfs(self, node): # old version directions should be altered
		directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
		random.shuffle(directions)
		x, y = node
		self.visited[x][y] = True
		for dx, dy in directions:
			new_x, new_y = x + dx, y + dy
			if 0 <= new_x < self.m and 0 <= new_y < self.n and not self.visited[new_x][new_y]:
				self.graph.add_edge(node, (new_x, new_y))
				self._dfs((new_x, new_y))

	# def plot_maze(self):
	# 	pos = {(x, y): (y, -x) for x in range(self.n) for y in range(self.n)}
	# 	nx.draw(self.graph, pos=pos, with_labels=True, node_color='lightblue', font_color='black')
	def find_solution(self,start,end) -> List[Tuple[int, int]]:
		path: List[Tuple[int, int]] = []  # Initialize path as an empty list
		try:
			# Use NetworkX's shortest path function
			path = nx.shortest_path(self.graph, start, end)
		except nx.NetworkXNoPath:
			print("No path between start and end.")
		return path

	def plot_maze(self, solution=False):
		pos = {(x, y): (x, y) for x in range(self.m*self.scale_factor) for y in range(self.n*self.scale_factor)}  # positions for all nodes

		# edges
		nx.draw_networkx_edges(self.graph, pos, width=0.5)

		# nodes (drawn as small dots)
		nx.draw_networkx_nodes(self.graph, pos, node_size=4, node_color='blue')

		if solution:

			# max_x = max(node[0] for node in self.graph.nodes)
			# filtered_nodes = [node for node in self.graph.nodes if node[0] == max_x]
			# max_y = max(node[1] for node in filtered_nodes)

			solution_path = self.find_solution(self.begin_node, self.end_node)
			# We change solution_path from nodes list to edge list
			edge_list = [(solution_path[i], solution_path[i + 1]) for i in range(len(solution_path) - 1)]

			# Draw solution edges with color red
			nx.draw_networkx_edges(self.graph, pos, edgelist=edge_list, edge_color='red', width=0.75)
			nx.draw_networkx_nodes(self.graph, pos, nodelist=[self.end_node], node_color='red', node_size=20)

		plt.axis('off')
		plt.savefig("graph.png")
		plt.show()
		pass

	def count_states_within_distance(self, start_node, k):
		visited = {start_node: 0}
		count = 1 if k >= 0 else 0
		queue = deque([(start_node, 0)])

		while queue:
			node, level = queue.popleft()
			if level < k:
				for neighbor in self.graph.neighbors(node):
					if neighbor not in visited:
						visited[neighbor] = level + 1
						queue.append((neighbor, level + 1))
						if visited[neighbor] <= k:
							count += 1

		return count

	#from here we do statistics
	def test_states_within_distance(self):
		distance_data = []
		for state in self.graph.nodes():
			distances = [self.count_states_within_distance(state, k) for k in range(1, 8)]
			distance_data.append([state] + distances)

		# Create DataFrame
		df = pd.DataFrame(distance_data,
						  columns=['State', 'Dist_1', 'Dist_2', 'Dist_3', 'Dist_4', 'Dist_5', 'Dist_6', 'Dist_7'])

		# Calculate average
		average = df.iloc[:, 1:].mean(axis=0)  # Calculate mean along columns, ignoring 'State'
		average_row = pd.DataFrame(average.values.reshape(1, -1), columns=df.columns[1:])
		average_row.insert(0, 'State', 'Average')

		# Append average row
		df = pd.concat([df, average_row], ignore_index=True)
		pd.set_option('display.max_columns', None)
		print(df)

# g =automaton_graph(100, 2, maze=False)
# g.draw_graph()
maze = Maze(25, 20)
# print(maze.find_solution((0, 0), (6, 8)))
# maze.plot_maze(maze.find_solution((0, 0), (maze.n-1, maze.n-1)))
maze.plot_maze(True)
maze.test_states_within_distance()

# visited_rooms = maze.run_simulation(n_simulations=100, l=5)

# Compute the average number of visited rooms
# average_visited_rooms = sum(visited_rooms) / len(visited_rooms)
# print(f"Average number of rooms visited: {average_visited_rooms}")
