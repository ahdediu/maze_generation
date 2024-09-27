import math
from collections import deque
from typing import List, Tuple, Set

import pandas as pd
from matplotlib import pyplot as plt

import random
import networkx as nx

''' Initial walls: mxnx2 (L shape walls for each cell)+n+n 
	to close the top and the right borders. 
	Now each edge can be considered as visiting one cell minus the last one 
	that visits 2 cells. so mxn-1 edges.  '''


class automaton_graph:
	def inverse_dir(self, direction):
		"""Return the complementary direction."""
		return (direction + 2) % 4

	def __init__(self, vertices: int, alphabet_size: int, maze: bool = True):
		"""
		Args:
		  m: vertices in the graph.
		  k: alphabet_size, Number of edges per vertex.
		  maze: If True, the graph is maze. For maze for each edge direction there is a complementary one
		  	practically we have 2*k-1 alphabet size.
		  	however, k stays the same only the edges labels will be greater than k
		"""
		# Direction mappings
		dx = [0, 1, 0, -1]  # for directions N(0), E(1), S(2), W(3) respectively
		dy = [1, 0, -1, 0]  # for directions N(0), E(1), S(2), W(3) respectively
		self.m = vertices
		self.k = alphabet_size
		self.maze = maze

		if self.maze:
			# various experiments; un-succesfull.
			pass
		else:
			self.graph = nx.DiGraph()
			self.vertex_level = {}
			for i in range(self.m):
				# self.graph.add_node(i)
				self.vertex_level[i] = self.m + 1  # we place all nodes outside the reachable level

			# edges later as for an edge we need 2 nodes.
			#  self.graph.add_edge(u,v)

			# no need to keep the edges on levels
			self.vertex_level[0] = 0
			self.graph.add_node(0)
			level = 1
			current_level = [0]
			while True:
				next_level = []
				for v in current_level:
					for i in range(self.k):
						random_vertex = random.choice(range(self.m))
						# self.graph.add_edge(v, random_vertex,label=i)
						self.graph.add_edge(v, random_vertex, label=i)
						if self.vertex_level[random_vertex] > level:
							self.graph.add_node(random_vertex)
							self.vertex_level[random_vertex] = level
							next_level.append(random_vertex)
				if next_level == []:
					for edge in self.graph.edges(data=True):
						print(edge)
					break
				else:
					current_level = next_level
					level += 1
			if self.graph.number_of_nodes() < self.m:
				print(f"Only {self.graph.number_of_nodes()} nodes are connected. Max level: {level}")

	def draw_graph(self):
		if self.maze:
			pass
		# pos = {(x, y): (x, y) for (x, y) in self.graph.nodes()}
		# nx.draw_networkx_edges(self.graph, pos, width=0.5)
		#
		# # nodes
		# nx.draw_networkx_nodes(self.graph, pos, node_size=4, node_color='blue')
		# plt.axis('off')
		# plt.savefig("graph.png")
		# plt.show()
		else:
			pos = nx.fruchterman_reingold_layout(
				self.graph)  # options: kamada_kawai_layout or nx.fruchterman_reingold_layout(self.graph)
			# pos = nx.spring_layout(self.graph, k=1.2)
			# nodes
			nx.draw_networkx_nodes(self.graph, pos)

			# edges
			nx.draw_networkx_edges(self.graph, pos)

			# labels for nodes
			nx.draw_networkx_labels(self.graph, pos)

			# labels for edges
			if self.maze:
				edge_labels = {(u, v): d['label'] for u, v, d in self.graph.edges(data=True) if d['label'] < self.k}
				nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_size=10, font_color='red')

			plt.axis('off')
			plt.savefig("graph.png")
			plt.show()


# g =automaton_graph(10, 2, maze=True)
# g.draw_graph()

class Maze:
	def __init__(self, m, n, directions=4):
		self.m = m
		self.n = n

		self.graph = nx.Graph()
		self.visited = [[False] * n for _ in range(m)]
		self.visited[0][0] = True
		self.graph.add_node((0, 0))
		self.begin_node = (0, 0)

		self.directions = directions
		self.scale_factor = 100

		self.dx = [round(self.scale_factor * math.cos(i * (2 * math.pi / self.directions))) for i in
				   range(self.directions)]
		self.dy = [round(self.scale_factor * math.sin(i * (2 * math.pi / self.directions))) for i in
				   range(self.directions)]
		self.loops = []
		self.make_maze()  #it modifies the end_node

	def make_maze(self):
		# algorithm = 'dfs'
		# algorithm = 'Wilson'
		algorithm = 'breadth_first'
		if algorithm == 'dfs':
			self.end_node = self.depth_first((0, 0))
		if algorithm == 'Wilson':
			self.Wilson_algorithm((0, 0))
		if algorithm == 'breadth_first':
			self.breadth_first()

	def Wilson_algorithm(self, r: Tuple[int, int]):
		''' from https://dl.acm.org/doi/10.1145/237814.237880 then we go on PDF
		RandomTreeWithRoot(r)
		for i in 1 to n
		  InTree[i] = false
	    Next[r] = nil
	    InTree[r] = true
	    for i in 1 to n // perform Self Erasing Loops Random Walk
	      u = i
	      while not InTree[u]
	        Next[u] = RandomSuccessor(u) //We update Next[u]; if we have a loop, Next[u] keeps any way only one element
	        u = Next[u]
		  u = i
		  while not InTree[u]
		    InTree[u] = true // we update the tree, considering the Next elements.
		    u = Next[u]
		return Next'''
		# first we find the set of vertices.
		possible_nodes = set()
		current_nodes = [r]

		while current_nodes:
			new_nodes = set()
			for x, y in current_nodes:
				for i in range(self.directions):
					nx, ny = x + self.dx[i], y + self.dy[i]
					if 0 <= nx < self.m * self.scale_factor and 0 <= ny < self.n * self.scale_factor:
						new_nodes.add((nx, ny))
			current_nodes = new_nodes - possible_nodes
			possible_nodes.update(new_nodes)
		InTree = {node: False for node in possible_nodes}
		Next = {node: None for node in possible_nodes}

		Next[r] = None
		InTree[r] = True

		def RandomSuccessor(u):
			x, y = u
			neighbors = [(x + self.dx[i], y + self.dy[i]) for i in range(self.directions) if
						 (x + self.dx[i], y + self.dy[i]) in possible_nodes]
			return random.choice(neighbors)

		for i in possible_nodes:
			u = i
			while not InTree[u]:
				Next[u] = RandomSuccessor(u)
				# We update Next[u];
				# if we have a loop, Next[u] keeps any way only one element
				u = Next[u]
			u = i
			while not InTree[u]:
				InTree[u] = True  # we update the InTree,
				u = Next[u]
		# we have to update the edges, + the self.end_node
		for i in possible_nodes:
			if Next[i]:
				self.graph.add_edge(i, Next[i])

		max_x = max(node[0] for node in self.graph.nodes)
		nodes_with_max_x = [node for node in self.graph.nodes if node[0] == max_x]
		self.end_node = max(nodes_with_max_x, key=lambda node: node[1])

	def depth_first(self, start):
		# algorithm based on dfs
		# dx, dy = [0, 1, 0, -1], [1, 0, -1, 0]  # directions to get the four neighbors

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
			neighbors = [(current[0] + self.dx[i], current[1] + self.dy[i]) for i in range(self.directions) if
						 0 <= current[0] + self.dx[i] < self.m * self.scale_factor and
						 0 <= current[1] + self.dy[i] < self.n * self.scale_factor and
						 (current[0] + self.dx[i], current[1] + self.dy[i]) not in visited]

			if neighbors:
				# Choose a random neighboring cell and mark as visited
				next_cell = random.choice(neighbors)
				visited[(next_cell)] = True
				stack.append(next_cell)

				# remove the wall between cells
				self.graph.add_edge(current, next_cell)
			else:
				# Additional looping can occur here with loop_chance %
				loop_chance = 0.25
				if random.random() < loop_chance and len(self.graph[current]) < self.directions - 1:
					# Find all neighbours that haven't reached maximum branching factor
					branching_candidates = [(current[0] + self.dx[i], current[1] + self.dy[i]) for i in
											range(self.directions) if
											0 <= current[0] + self.dx[i] < self.m * self.scale_factor and
											0 <= current[1] + self.dy[i] < self.n * self.scale_factor and
											len(self.graph[(
												current[0] + self.dx[i],
												current[1] + self.dy[i])]) < self.directions - 1]
					if branching_candidates:
						next_cell = random.choice(branching_candidates)
						self.graph.add_edge(current, next_cell)
						self.loops.append((next_cell, current))
				# Backtrack
				stack.pop()
		return node_at_max_depth

	def _dfs(self, node):  # old version directions should be altered
		directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
		random.shuffle(directions)
		x, y = node
		self.visited[x][y] = True
		for dx, dy in directions:
			new_x, new_y = x + dx, y + dy
			if 0 <= new_x < self.m and 0 <= new_y < self.n and not self.visited[new_x][new_y]:
				self.graph.add_edge(node, (new_x, new_y))
				self._dfs((new_x, new_y))

	def condition(self, x: int, y: int, config: int) -> bool:
		# we receive an x and an y, + a configuration and we say if it is valid or not
		# configurations are binary values using the directions starting from N
		# a 0 bit means a wall, a 1 means passage.
		# for self.direction == 3 configuration in [1..6]
		# for self.direction == 4 configuration in [1..14]
		# self.direction == 6 configuration in [1..62]
		# we write first for self.direction == 4
		'''			if config == 0b0001:
			    return False
			elif config == 0b0010:
			    return False
			elif config == 0b0011:
			    return False
			elif config == 0b0100:
			    return True
			elif config == 0b0101:
			    return False
			elif config == 0b0110:
			    return False
			elif config == 0b0111:
			    return False
			elif config == 0b1000:
			    return True
			elif config == 0b1001:
			    return False
			elif config == 0b1010:
			    return False
			elif config == 0b1011:
			    return False
			elif config == 0b1100:
			    return True
			elif config == 0b1101:
			    return False
			elif config == 0b1110:
			    return False
			else:'''
		#0 N, 1 E, 2 S, 3 W
		if (x, y) == (0, 0):
			return (config & 0b0011) == 0
		else:
			print("Something went wrong")
			return False

	def decode(self, value: int) -> Set[int]:
		directions_set = set()
		for i in range(self.directions - 1, -1, -1):  # iterate over directions bits
			if value & (1 << i):
				directions_set.add(self.directions - 1 - i)
		return directions_set

	def breadth_first(self):
		possible_configurations = [c for c in range(2 ** self.directions)]
		direction_bits = [2 ** (self.directions - 1 - i) for i in range(self.directions)]
		complementary_direction_bits = [2 ** self.directions - 1 - b for b in direction_bits]
		# vertex_level = {(x, y): self.m * self.n + 1 for x in range(self.m) for y in range(self.n)}
		vertex_level = {}
		vertex_level[(0, 0)] = 0
		self.graph.add_node((0, 0))
		level = 1
		current_level = [(0, 0)]
		while True:
			next_level = []

			for x, y in current_level:
				valid_configurations = set(possible_configurations)
				for i in range(self.directions):
					nx, ny = x + self.dx[i], y + self.dy[i]
					if 0 <= nx < self.m * self.scale_factor and \
							0 <= ny < self.n * self.scale_factor:
						if (nx, ny) in vertex_level :
							# if we have an edge we do not add another
							if self.graph.has_edge((x,y),(nx, ny)):
								valid_configurations = {config & complementary_direction_bits[i] for config in
														valid_configurations}
							else: # if less than all paths even after adding one edge
								if self.graph.degree[(nx,ny)]<self.directions/2:
									pass # we allow this possible llop
								else:
									valid_configurations = {config & complementary_direction_bits[i] for config in
															valid_configurations}

					else:
						# the neighbour is outside of the grid margins
						# we do not go there.
						valid_configurations = {config & complementary_direction_bits[i] for config in
												valid_configurations}
				valid_configurations.discard(0) # all walls
				valid_configurations.discard(2 ** self.directions - 1) # all passages
				if valid_configurations: # that is not empty
					selected_configuration = random.choice(list(valid_configurations))
					decoded_set = self.decode(selected_configuration)
					for i in decoded_set:
						nx, ny = x + self.dx[i], y + self.dy[i]
						next_level.append((nx, ny))
						self.graph.add_edge((x, y), (nx, ny))
						vertex_level[(nx, ny)] = level
			if next_level == []:
				max_x = max(node[0] for node in self.graph.nodes)
				nodes_with_max_x = [node for node in self.graph.nodes if node[0] == max_x]
				self.end_node = max(nodes_with_max_x, key=lambda node: node[1])

				break
			else:
				current_level = next_level
				level += 1

	def find_solution(self, start, end) -> List[Tuple[int, int]]:
		path: List[Tuple[int, int]] = []  # Initialize path as an empty list
		try:
			# Use NetworkX's shortest path function
			path = nx.shortest_path(self.graph, start, end)
		except nx.NetworkXNoPath:
			print("No path between start and end.")
		return path

	def plot_maze(self, solution=False):
		pos = {(x, y): (x, y) for x in range(self.m * self.scale_factor) for y in
			   range(self.n * self.scale_factor)}  # positions for all nodes

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
maze = Maze(25, 20, directions=3)
# print(maze.find_solution((0, 0), (6, 8)))
# maze.plot_maze(maze.find_solution((0, 0), (maze.n-1, maze.n-1)))
maze.plot_maze(solution=True)
# maze.test_states_within_distance()

# visited_rooms = maze.run_simulation(n_simulations=100, l=5)

# Compute the average number of visited rooms
# average_visited_rooms = sum(visited_rooms) / len(visited_rooms)
# print(f"Average number of rooms visited: {average_visited_rooms}")
