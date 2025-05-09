import copy
import random
from collections import defaultdict
import heapq
from collections import deque
import math
import time

class PuzzleSolver:
    def __init__(self, goal_state):
        self.goal_state = goal_state

    def is_goal(self, state):
        return state == self.goal_state

    def find_blank(self, state):
        for i in range(3):
            for j in range(3):
                if state[i][j] == 0:
                    return i, j

    def get_neighbor(self, state):
        neighbors = []
        row, col = self.find_blank(state)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < 3 and 0 <= new_col < 3:
                new_state = copy.deepcopy(state)
                new_state[row][col], new_state[new_row][new_col] = new_state[new_row][new_col], new_state[row][col]
                neighbors.append(new_state)
        return neighbors

    def manhattan_distance(self, state):
        distance = 0
        for i in range(3):
            for j in range(3):
                value = state[i][j]
                if value != 0:
                    goal_i, goal_j = divmod(value - 1, 3)
                    distance += abs(i - goal_i) + abs(j - goal_j)
        return distance

    def generate_random_state(self):
        state = copy.deepcopy(self.goal_state)
        for _ in range(100):
            neighbors = self.get_neighbor(state)
            if neighbors:
                state = random.choice(neighbors)
        return state

    def bfs(self, initial_state):
        queue = deque([(initial_state, [])])
        visited = {str(initial_state)}
        start_time = time.time()
        while queue:
            state, path = queue.popleft()
            if self.is_goal(state):
                elapsed_time = time.time() - start_time
                return path + [state], (len(visited), len(path), elapsed_time)
            for neighbor in self.get_neighbor(state):
                neighbor_str = str(neighbor)
                if neighbor_str not in visited:
                    visited.add(neighbor_str)
                    queue.append((neighbor, path + [state]))
        elapsed_time = time.time() - start_time
        return None, (len(visited), 0, elapsed_time)

    def dfs(self, initial_state):
        stack = [(initial_state, [])]
        visited = {str(initial_state)}
        start_time = time.time()
        while stack:
            state, path = stack.pop()
            if self.is_goal(state):
                elapsed_time = time.time() - start_time
                return path + [state], (len(visited), len(path), elapsed_time)
            for neighbor in self.get_neighbor(state):
                neighbor_str = str(neighbor)
                if neighbor_str not in visited:
                    visited.add(neighbor_str)
                    stack.append((neighbor, path + [state]))
        elapsed_time = time.time() - start_time
        return None, (len(visited), 0, elapsed_time)

    def iddfs(self, initial_state, max_depth):
        def dls(state, path, depth, visited):
            if depth < 0:
                return None
            if self.is_goal(state):
                return path + [state]
            visited.add(str(state))
            for neighbor in self.get_neighbor(state):
                neighbor_str = str(neighbor)
                if neighbor_str not in visited:
                    result = dls(neighbor, path + [state], depth - 1, visited)
                    if result is not None:
                        return result
            return None

        total_visited = set()
        start_time = time.time()
        for depth in range(max_depth + 1):
            visited = set()
            result = dls(initial_state, [], depth, visited)
            total_visited.update(visited)
            if result is not None:
                elapsed_time = time.time() - start_time
                return result, (len(total_visited), len(result) - 1, elapsed_time)
        elapsed_time = time.time() - start_time
        return None, (len(total_visited), 0, elapsed_time)

    def greedy_search(self, initial_state):
        queue = [(self.manhattan_distance(initial_state), id(initial_state), initial_state, [])]
        visited = {str(initial_state)}
        start_time = time.time()
        while queue:
            _, _, state, path = heapq.heappop(queue)
            if self.is_goal(state):
                elapsed_time = time.time() - start_time
                return path + [state], (len(visited), len(path), elapsed_time)
            for neighbor in self.get_neighbor(state):
                neighbor_str = str(neighbor)
                if neighbor_str not in visited:
                    visited.add(neighbor_str)
                    heuristic = self.manhattan_distance(neighbor)
                    heapq.heappush(queue, (heuristic, id(neighbor), neighbor, path + [state]))
        elapsed_time = time.time() - start_time
        return None, (len(visited), 0, elapsed_time)

    def ucs(self, initial_state):
        queue = [(0, id(initial_state), initial_state, [])]
        visited = {str(initial_state)}
        start_time = time.time()
        while queue:
            cost, _, state, path = heapq.heappop(queue)
            if self.is_goal(state):
                elapsed_time = time.time() - start_time
                return path + [state], (len(visited), len(path), elapsed_time)
            for neighbor in self.get_neighbor(state):
                neighbor_str = str(neighbor)
                if neighbor_str not in visited:
                    visited.add(neighbor_str)
                    new_cost = cost + 1
                    heapq.heappush(queue, (new_cost, id(neighbor), neighbor, path + [state]))
        elapsed_time = time.time() - start_time
        return None, (len(visited), 0, elapsed_time)

    def astar_search(self, initial_state):
        start_h = self.manhattan_distance(initial_state)
        queue = [(start_h, 0, id(initial_state), initial_state, [])]
        visited = {str(initial_state): 0}
        start_time = time.time()
        while queue:
            _, g, _, state, path = heapq.heappop(queue)
            state_str = str(state)
            if self.is_goal(state):
                elapsed_time = time.time() - start_time
                return path + [state], (len(visited), len(path), elapsed_time)
            for neighbor in self.get_neighbor(state):
                neighbor_str = str(neighbor)
                new_g = g + 1
                if neighbor_str not in visited or new_g < visited[neighbor_str]:
                    visited[neighbor_str] = new_g
                    h = self.manhattan_distance(neighbor)
                    f = new_g + h
                    heapq.heappush(queue, (f, new_g, id(neighbor), neighbor, path + [state]))
        elapsed_time = time.time() - start_time
        return None, (len(visited), 0, elapsed_time)

    def idastar_search(self, initial_state):
        def search(path, g, threshold):
            state = path[-1]
            f = g + self.manhattan_distance(state)
            if f > threshold:
                return f, None
            if self.is_goal(state):
                return None, path
            min_threshold = float('inf')
            for neighbor in self.get_neighbor(state):
                if neighbor not in [p for p in path]:
                    path.append(neighbor)
                    t, result = search(path, g + 1, threshold)
                    if result is not None:
                        return None, result
                    path.pop()
                    if t is not None:
                        min_threshold = min(min_threshold, t)
            return min_threshold, None

        threshold = self.manhattan_distance(initial_state)
        path = [initial_state]
        visited = set()
        start_time = time.time()
        while True:
            t, result = search(path, 0, threshold)
            visited.add(str(path[-1]))
            if result is not None:
                elapsed_time = time.time() - start_time
                return result, (len(visited), len(result) - 1, elapsed_time)
            if t == float('inf'):
                elapsed_time = time.time() - start_time
                return None, (len(visited), 0, elapsed_time)
            threshold = t

    def simple_hill_climbing(self, initial_state):
        current_state = initial_state
        current_path = [current_state]
        current_h = self.manhattan_distance(current_state)
        visited = {str(current_state)}
        max_iterations = 1000
        iteration = 0
        start_time = time.time()
        while iteration < max_iterations:
            iteration += 1
            neighbors = self.get_neighbor(current_state)
            found_better = False
            for neighbor in neighbors:
                neighbor_h = self.manhattan_distance(neighbor)
                if self.is_goal(neighbor):
                    elapsed_time = time.time() - start_time
                    return current_path + [neighbor], (len(visited), len(current_path), elapsed_time)
                if neighbor_h < current_h:
                    current_state = neighbor
                    current_h = neighbor_h
                    current_path.append(current_state)
                    visited.add(str(current_state))
                    found_better = True
                    break
            if not found_better:
                elapsed_time = time.time() - start_time
                return None, (len(visited), len(current_path) - 1, elapsed_time)
        elapsed_time = time.time() - start_time
        return None, (len(visited), len(current_path) - 1, elapsed_time)

    def steepest_hill_climbing(self, initial_state):
        beam_width = 2
        initial_h = self.manhattan_distance(initial_state)
        current_beam = [(initial_h, initial_state, [initial_state])]
        visited = {str(initial_state)}
        max_iterations = 1000
        iteration = 0
        prev_best_h = initial_h
        start_time = time.time()
        while iteration < max_iterations and current_beam:
            iteration += 1
            all_neighbors = []
            for _, current_state, current_path in current_beam:
                neighbors = self.get_neighbor(current_state)
                for neighbor in neighbors:
                    neighbor_str = str(neighbor)
                    if neighbor_str not in visited:
                        neighbor_h = self.manhattan_distance(neighbor)
                        if self.is_goal(neighbor):
                            elapsed_time = time.time() - start_time
                            return current_path + [neighbor], (len(visited), len(current_path), elapsed_time)
                        all_neighbors.append((neighbor_h, neighbor, current_path + [neighbor]))
                        visited.add(neighbor_str)
            if not all_neighbors:
                elapsed_time = time.time() - start_time
                return None, (len(visited), 0, elapsed_time)
            all_neighbors.sort(key=lambda x: x[0])
            current_beam = all_neighbors[:beam_width]
            best_h, _, _ = current_beam[0]
            if best_h >= prev_best_h:
                if iteration > 1:
                    elapsed_time = time.time() - start_time
                    return None, (len(visited), 0, elapsed_time)
            prev_best_h = best_h
        elapsed_time = time.time() - start_time
        return None, (len(visited), 0, elapsed_time)

    def stochastic_hill_climbing(self, initial_state):
        current_state = initial_state
        current_path = [current_state]
        current_h = self.manhattan_distance(current_state)
        visited = {str(current_state)}
        max_iterations = 1000
        iteration = 0
        start_time = time.time()
        while iteration < max_iterations:
            iteration += 1
            neighbors = self.get_neighbor(current_state)
            better_neighbors = []
            for neighbor in neighbors:
                neighbor_str = str(neighbor)
                if neighbor_str not in visited:
                    neighbor_h = self.manhattan_distance(neighbor)
                    if self.is_goal(neighbor):
                        elapsed_time = time.time() - start_time
                        return current_path + [neighbor], (len(visited), len(current_path), elapsed_time)
                    if neighbor_h < current_h:
                        better_neighbors.append((neighbor, neighbor_h))
            if better_neighbors:
                improvements = [current_h - h for _, h in better_neighbors]
                total_improvement = sum(improvements)
                if total_improvement > 0:
                    probabilities = [imp / total_improvement for imp in improvements]
                    index = random.choices(range(len(better_neighbors)), weights=probabilities, k=1)[0]
                else:
                    index = random.randint(0, len(better_neighbors) - 1)
                current_state, current_h = better_neighbors[index]
                current_path.append(current_state)
                visited.add(str(current_state))
            else:
                initial_temp = 1.0
                final_temp = 0.01
                alpha = 0.99
                temp = initial_temp
                while temp > final_temp and iteration < max_iterations:
                    iteration += 1
                    neighbors = self.get_neighbor(current_state)
                    if not neighbors:
                        break
                    next_state = random.choice(neighbors)
                    next_h = self.manhattan_distance(next_state)
                    if self.is_goal(next_state):
                        elapsed_time = time.time() - start_time
                        return current_path + [next_state], (len(visited), len(current_path), elapsed_time)
                    delta_e = current_h - next_h
                    if delta_e > 0 or random.random() < math.exp(delta_e / temp):
                        current_state = next_state
                        current_h = next_h
                        current_path.append(current_state)
                        visited.add(str(current_state))
                    temp *= alpha
                if iteration >= max_iterations or temp <= final_temp:
                    elapsed_time = time.time() - start_time
                    return None, (len(visited), len(current_path) - 1, elapsed_time)
        elapsed_time = time.time() - start_time
        return None, (len(visited), len(current_path) - 1, elapsed_time)

    def simulated_annealing(self, initial_state):
        current_state = initial_state
        current_path = [current_state]
        current_h = self.manhattan_distance(current_state)
        visited = {str(current_state)}
        initial_temp = 1.0
        final_temp = 0.01
        alpha = 0.99
        max_iterations = 1000
        temp = initial_temp
        iteration = 0
        start_time = time.time()
        while temp > final_temp and iteration < max_iterations:
            iteration += 1
            neighbors = self.get_neighbor(current_state)
            if not neighbors:
                break
            next_state = random.choice(neighbors)
            next_h = self.manhattan_distance(next_state)
            if self.is_goal(next_state):
                elapsed_time = time.time() - start_time
                return current_path + [next_state], (len(visited), len(current_path), elapsed_time)
            delta_e = current_h - next_h
            if delta_e > 0 or random.random() < math.exp(delta_e / temp):
                current_state = next_state
                current_h = next_h
                current_path.append(current_state)
                visited.add(str(current_state))
            temp *= alpha
        elapsed_time = time.time() - start_time
        return None, (len(visited), len(current_path) - 1, elapsed_time)

    def beam_search(self, initial_state):
        beam_width = 3
        current_beam = [(self.manhattan_distance(initial_state), initial_state, [initial_state])]
        visited = {str(initial_state)}
        max_iterations = 1000
        iteration = 0
        prev_best_h = self.manhattan_distance(initial_state)
        start_time = time.time()
        while iteration < max_iterations and current_beam:
            iteration += 1
            all_neighbors = []
            for _, current_state, current_path in current_beam:
                neighbors = self.get_neighbor(current_state)
                for neighbor in neighbors:
                    neighbor_str = str(neighbor)
                    if neighbor_str not in visited:
                        neighbor_h = self.manhattan_distance(neighbor)
                        if self.is_goal(neighbor):
                            elapsed_time = time.time() - start_time
                            return current_path + [neighbor], (len(visited), len(current_path), elapsed_time)
                        all_neighbors.append((neighbor_h, neighbor, current_path + [neighbor]))
                        visited.add(neighbor_str)
            if not all_neighbors:
                elapsed_time = time.time() - start_time
                return None, (len(visited), 0, elapsed_time)
            all_neighbors.sort(key=lambda x: x[0])
            current_beam = all_neighbors[:beam_width]
            best_h, _, _ = current_beam[0]
            if best_h >= prev_best_h and iteration > 1:
                elapsed_time = time.time() - start_time
                return None, (len(visited), 0, elapsed_time)
            prev_best_h = best_h
        elapsed_time = time.time() - start_time
        return None, (len(visited), 0, elapsed_time)

    def genetic_algorithm(self, initial_state):
        population_size = 100
        max_generations = 50
        mutation_rate = 0.3
        tournament_size = 5
        elite_size = 10
        visited = {str(initial_state)}
        start_time = time.time()

        def generate_move_sequence(length):
            return [random.randint(0, 3) for _ in range(length)]

        def apply_moves(state, move_sequence):
            current = copy.deepcopy(state)
            path = [current]
            for move in move_sequence:
                blank_row, blank_col = self.find_blank(current)
                directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                dr, dc = directions[move]
                new_row, new_col = blank_row + dr, blank_col + dc
                if not (0 <= new_row < 3 and 0 <= new_col < 3):
                    continue
                new_state = copy.deepcopy(current)
                new_state[blank_row][blank_col], new_state[new_row][new_col] = new_state[new_row][new_col], new_state[blank_row][blank_col]
                current = new_state
                path.append(current)
                visited.add(str(current))
                if self.is_goal(current):
                    return path, True
            return path, self.is_goal(path[-1]) if path else False

        def fitness(chromosome):
            path, found_solution = apply_moves(initial_state, chromosome)
            if found_solution:
                return -1000 + len(path)
            if not path:
                return 1000
            return self.manhattan_distance(path[-1]) + len(path) * 0.1

        def selection(population_with_fitness):
            selected = []
            elite = sorted(population_with_fitness, key=lambda x: x[0])[:elite_size]
            selected.extend([chromosome for _, chromosome in elite])
            while len(selected) < population_size:
                tournament = random.sample(population_with_fitness, tournament_size)
                winner = min(tournament, key=lambda x: x[0])
                selected.append(winner[1])
            return selected

        def crossover(parent1, parent2):
            if len(parent1) <= 1 or len(parent2) <= 1:
                return copy.deepcopy(parent1)
            crossover_point = random.randint(1, min(len(parent1), len(parent2)) - 1)
            child = parent1[:crossover_point] + parent2[crossover_point:]
            return child

        def mutate(chromosome):
            if random.random() < mutation_rate:
                mutated = chromosome.copy()
                mutation_type = random.randint(0, 2)
                if mutation_type == 0:
                    if mutated:
                        index = random.randint(0, len(mutated) - 1)
                        mutated[index] = random.randint(0, 3)
                elif mutation_type == 1:
                    for _ in range(random.randint(1, 3)):
                        mutated.append(random.randint(0, 3))
                else:
                    if len(mutated) > 5:
                        for _ in range(min(random.randint(1, 3), len(mutated) - 1)):
                            index = random.randint(0, len(mutated) - 1)
                            mutated.pop(index)
                return mutated
            return chromosome

        population = [generate_move_sequence(random.randint(20, 50)) for _ in range(population_size)]
        best_solution = None
        for generation in range(max_generations):
            population_with_fitness = [(fitness(chromosome), chromosome) for chromosome in population]
            for fit, chromosome in population_with_fitness:
                if fit < 0:
                    path, _ = apply_moves(initial_state, chromosome)
                    if not best_solution or len(path) < len(best_solution):
                        best_solution = path
            if best_solution:
                elapsed_time = time.time() - start_time
                return best_solution, (len(visited), len(best_solution) - 1, elapsed_time)
            selected = selection(population_with_fitness)
            next_population = []
            while len(next_population) < population_size:
                parent1, parent2 = random.sample(selected, 2)
                child = crossover(parent1, parent2)
                child = mutate(child)
                next_population.append(child)
            population = next_population
            if generation > 10 and min(population_with_fitness, key=lambda x: x[0])[0] > 5:
                mutation_rate = min(0.5, mutation_rate * 1.05)
        if best_solution:
            elapsed_time = time.time() - start_time
            return best_solution, (len(visited), len(best_solution) - 1, elapsed_time)
        best_chromosome = min(population, key=fitness)
        best_path, _ = apply_moves(initial_state, best_chromosome)
        elapsed_time = time.time() - start_time
        return None, (len(visited), len(best_path) - 1 if best_path else 0, elapsed_time)

    def nondeterministic_search(self, initial_state):
        max_depth = 20
        visited = {str(initial_state)}
        state_count = 1
        start_time = time.time()

        def dfs_and(state, depth, path):
            nonlocal state_count
            if depth > max_depth:
                return None
            if self.is_goal(state):
                return path + [state]
            blank_row, blank_col = self.find_blank(state)
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            for dr, dc in directions:
                new_row, new_col = blank_row + dr, blank_col + dc
                if not (0 <= new_row < 3 and 0 <= new_col < 3):
                    continue
                possible_outcomes = []
                outcome1 = copy.deepcopy(state)
                outcome1[blank_row][blank_col], outcome1[new_row][new_col] = outcome1[new_row][new_col], outcome1[blank_row][blank_col]
                outcome1_str = str(outcome1)
                if outcome1_str not in visited:
                    visited.add(outcome1_str)
                    state_count += 1
                    possible_outcomes.append(outcome1)
                outcome2 = copy.deepcopy(state)
                outcome2_str = str(outcome2)
                if outcome2_str not in visited:
                    visited.add(outcome2_str)
                    state_count += 1
                    possible_outcomes.append(outcome2)
                if possible_outcomes:
                    all_paths = []
                    valid_for_all = True
                    for outcome in possible_outcomes:
                        result = dfs_and(outcome, depth + 1, path + [state])
                        if result is None:
                            valid_for_all = False
                            break
                        all_paths.append(result)
                    if valid_for_all and all_paths:
                        return all_paths[0]
            return None

        result = dfs_and(initial_state, 0, [])
        elapsed_time = time.time() - start_time
        return result, (state_count, len(result) - 1 if result else 0, elapsed_time)

    def actions(self):
        return [0, 1, 2, 3]

    def results(self, state, action):
        blank_row, blank_col = self.find_blank(state)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        dr, dc = directions[action]
        new_row, new_col = blank_row + dr, blank_col + dc
        result_states = []
        if 0 <= new_row < 3 and 0 <= new_col < 3:
            new_state = copy.deepcopy(state)
            new_state[blank_row][blank_col], new_state[new_row][new_col] = new_state[new_row][new_col], new_state[blank_row][blank_col]
            result_states.append(new_state)
        else:
            result_states.append(copy.deepcopy(state))
        return result_states

    def no_observations_search(self, initial_state):
        initial_belief = {str(initial_state): initial_state}
        queue = deque([(initial_belief, [])])
        visited_beliefs = {frozenset(initial_belief.keys())}
        max_depth = 15
        states_explored = 1
        start_time = time.time()
        while queue:
            belief, plan = queue.popleft()
            all_goals = True
            for state_str, state in belief.items():
                if not self.is_goal(state):
                    all_goals = False
                    break
            if all_goals:
                solution_path = [initial_state]
                current_state = initial_state
                for action in plan:
                    results = self.results(current_state, action)
                    if results:
                        current_state = results[0]
                        solution_path.append(current_state)
                elapsed_time = time.time() - start_time
                return solution_path, (states_explored, len(solution_path) - 1, elapsed_time)
            if len(plan) >= max_depth:
                continue
            for action in self.actions():
                new_belief = {}
                for state_str, state in belief.items():
                    possible_results = self.results(state, action)
                    states_explored += len(possible_results)
                    for result_state in possible_results:
                        result_str = str(result_state)
                        new_belief[result_str] = result_state
                belief_key = frozenset(new_belief.keys())
                if not new_belief or belief_key in visited_beliefs:
                    continue
                visited_beliefs.add(belief_key)
                queue.append((new_belief, plan + [action]))
        elapsed_time = time.time() - start_time
        return None, (states_explored, 0, elapsed_time)

    def partially_observable_search(self, initial_state):
        def get_observation(state):
            blank_row, blank_col = self.find_blank(state)
            return (blank_row, blank_col)

        initial_observation = get_observation(initial_state)
        belief_states = {str(initial_state): initial_state}
        queue = deque([(belief_states, [])])
        visited_beliefs = set()
        max_depth = 30
        states_explored = 1
        start_time = time.time()
        while queue:
            belief, plan = queue.popleft()
            belief_key = frozenset(belief.keys())
            if belief_key in visited_beliefs:
                continue
            visited_beliefs.add(belief_key)
            all_goals = all(self.is_goal(state) for state in belief.values())
            if all_goals:
                solution_path = [initial_state]
                current_state = initial_state
                for action in plan:
                    results = self.results(current_state, action)
                    if results:
                        current_state = results[0]
                        solution_path.append(current_state)
                elapsed_time = time.time() - start_time
                return solution_path, (states_explored, len(solution_path) - 1, elapsed_time)
            if len(plan) >= max_depth:
                continue
            for action in self.actions():
                next_belief = {}
                for state_str, state in belief.items():
                    possible_results = self.results(state, action)
                    states_explored += len(possible_results)
                    for result_state in possible_results:
                        result_str = str(result_state)
                        if result_str not in next_belief:
                            next_belief[result_str] = result_state
                if next_belief:
                    queue.append((next_belief, plan + [action]))
        elapsed_time = time.time() - start_time
        return None, (states_explored, 0, elapsed_time)
    
    def is_goal_for_CSPs(self, state):
        for i in range(3):
            row = [state[i][j] for j in range(3) if state[i][j] != 0]
            col = [state[j][i] for j in range(3) if state[j][i] != 0]
            if len(set(row)) != len(row) or len(set(col)) != len(col):
                return False
        return len([x for row in state for x in row if x != 0]) == 9
    
    def backtracking_search(self, initial_state, ui=None):
        variables = [(i,j) for i in range(3) for j in range(3)]
        domains = {v: list(range(9)) for v in variables}
        assignment = {}
        steps = 0

        def is_consistent(var, val, assign):
            temp = [[assign.get((i,j), 0) for j in range(3)] for i in range(3)]
            r, c = var
            temp[r][c] = val
            vals = [temp[x][y] for x in range(3) for y in range(3) if temp[x][y] != 0]
            return len(vals) == len(set(vals))

        def backtrack(assign):
            nonlocal steps
            steps += 1
            if ui:
                ui.update_status(f"Backtracking: Step {steps}")
                ui.update_board_from_assignment(assign)
                time.sleep(0.5)  # Pause to allow user to observe
            if len(assign) == 9:
                state = [[assign[(i,j)] for j in range(3)] for i in range(3)]
                if self.is_goal_for_CSPs(state):
                    if ui:
                        ui.update_status("Solution found!")
                    return state
                else:
                    if ui:
                        ui.update_status("Assignment does not satisfy goal, backtracking...")
                    return None

            var = next(v for v in variables if v not in assign)
            for val in domains[var]:
                if is_consistent(var, val, assign):
                    assign[var] = val
                    result = backtrack(assign)
                    if result is not None:
                        return result
                    del assign[var]
                    if ui:
                        ui.update_status(f"Backtracking from {var} with value {val}")
                else:
                    if ui:
                        ui.update_status(f"{val} is not consistent with {var}")
            if ui:
                ui.update_status(f"No suitable value for {var}, backtracking...")
            return None

        solution = backtrack(assignment)
        if solution is None and ui:
            ui.update_status("No solution found.")
        return solution, steps

    def forward_checking(self, initial_state, ui=None):
        if not isinstance(initial_state, list) or not all(isinstance(row, list) for row in initial_state):
            raise ValueError("initial_state must be a 3x3 matrix")
        if len(initial_state) != 3 or any(len(row) != 3 for row in initial_state):
            raise ValueError("initial_state must be 3x3")

        variables = [(i, j) for i in range(3) for j in range(3)]
        domains = {}
        assignment = {}
        for i, j in variables:
            if initial_state[i][j] == 0:
                domains[(i, j)] = list(range(1, 10))  # Assuming values from 1-9
            else:
                val = initial_state[i][j]
                domains[(i, j)] = [val]
                assignment[(i, j)] = val

        steps = 0

        def inference(var, val, doms, assign):
            inf = {}
            if ui:
                ui.update_status(f"Forward checking for {var} with value {val}")
            for other in doms:
                if other in assign or other == var:
                    continue
                if val in doms[other]:
                    doms[other].remove(val)
                    inf.setdefault(other, []).append(val)
                    if ui:
                        ui.update_status(f"Removed {val} from domain of {other}: {doms[other]}")
                    if not doms[other]:
                        if ui:
                            ui.update_status(f"Domain of {other} is empty, failure!")
                        return 'failure'
            return inf

        def undo(inf, doms):
            for v, lst in inf.items():
                doms[v].extend(lst)
                if ui:
                    ui.update_status(f"Restored domain for {v}: {doms[v]}")

        def backtrack(assign, doms):
            nonlocal steps
            steps += 1
            if ui:
                ui.update_status(f"Forward Checking: Step {steps}")
                ui.update_board_from_assignment(assign)
                time.sleep(0.5)  # Pause to allow user to observe
            if len(assign) == 9:
                state = [[assign.get((i, j), 0) for j in range(3)] for i in range(3)]
                if self.is_goal_for_CSPs(state):
                    if ui:
                        ui.update_status("Solution found!")
                    return state
                else:
                    if ui:
                        ui.update_status("Assignment does not satisfy goal, backtracking...")
                    return None

            var = min((v for v in doms if v not in assign), key=lambda x: len(doms[x]))
            for val in sorted(doms[var], key=lambda v: self.manhattan_distance(
                    [[assign.get((i, j), v if (i, j) == var else 0) for j in range(3)] for i in range(3)])):
                assign[var] = val
                inf = inference(var, val, doms, assign)
                if inf != 'failure':
                    result = backtrack(assign, doms)
                    if result is not None:
                        return result
                    undo(inf, doms)
                    if ui:
                        ui.update_status(f"Backtracking from {var} with {val}, restoring domains")
                else:
                    if ui:
                        ui.update_status(f"Forward checking failed for {var} with {val}")
                del assign[var]
                if ui:
                    ui.update_status(f"Removed assignment {val} from {var}")
            if ui:
                ui.update_status(f"No suitable value for {var}, backtracking...")
            return None

        solution = backtrack(assignment, domains)
        if solution is None and ui:
            ui.update_status("No solution found.")
        return solution, steps

    def conflicts(self, state, pos, val):
        r,c = pos
        temp = copy.deepcopy(state)
        temp[r][c] = val
        rv = [x for x in temp[r] if x]
        cv = [temp[i][c] for i in range(3) if temp[i][c]]
        dup = (len(rv)-len(set(rv))) + (len(cv)-len(set(cv)))
        return dup + self.manhattan_distance(temp)

    def min_conflicts(self, initial_state, max_steps=1000):
        curr = copy.deepcopy(initial_state)
        vars = [(i,j) for i in range(3) for j in range(3)]
        domains = {v: list(range(1,9)) for v in vars if initial_state[v[0]][v[1]]==0}

        for step in range(max_steps):
            if self.is_goal_for_CSPs(curr):
                return curr, step
            conflicted = [v for v in vars if curr[v[0]][v[1]]!=0 and self.conflicts(curr, v, curr[v[0]][v[1]])>0]
            var = random.choice(conflicted)
            best = min(domains.get(var,[]), key=lambda x: self.conflicts(curr,var,x))
            curr[var[0]][var[1]] = best
        return None, max_steps

    def q_learning(self, initial_state):
        Q_table = defaultdict(lambda: [0, 0, 0, 0])
        alpha = 0.1     # learning rate
        gamma = 0.9     # discount factor
        epsilon_start = 1.0
        epsilon_end = 0.1
        total_episodes = 1000
        max_steps = 100
        start_time = time.time()

        for episode in range(total_episodes):
            state = self.generate_random_state()
            state_tuple = tuple(map(tuple, state))
            epsilon = max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * episode / total_episodes)

            for step in range(max_steps):
                possible_actions = self.get_possible_actions(state)
                if random.random() < epsilon:
                    action = random.choice(possible_actions)
                else:
                    q_values = [Q_table[state_tuple][a] for a in possible_actions]
                    max_q = max(q_values)
                    best_actions = [a for a, q in zip(possible_actions, q_values) if q == max_q]
                    action = random.choice(best_actions)

                next_state = self.results(state, action)[0]
                reward = 100 if self.is_goal(next_state) else -1
                next_state_tuple = tuple(map(tuple, next_state))

                if self.is_goal(next_state):
                    target = reward
                else:
                    next_possible_actions = self.get_possible_actions(next_state)
                    max_q_next = max(Q_table[next_state_tuple][a] for a in next_possible_actions) if next_possible_actions else 0
                    target = reward + gamma * max_q_next

                Q_table[state_tuple][action] += alpha * (target - Q_table[state_tuple][action])
                state = next_state
                state_tuple = next_state_tuple
                if self.is_goal(state):
                    break

        state = initial_state
        path = [state]
        max_path_length = 100
        for _ in range(max_path_length):
            if self.is_goal(state):
                break
            state_tuple = tuple(map(tuple, state))
            possible_actions = self.get_possible_actions(state)
            if possible_actions:
                q_values = [Q_table[state_tuple][a] for a in possible_actions]
                max_q = max(q_values)
                best_actions = [a for a, q in zip(possible_actions, q_values) if q == max_q]
                action = random.choice(best_actions)
                next_state = self.results(state, action)[0]
                path.append(next_state)
                state = next_state
            else:
                break
        else:
            path = None

        elapsed_time = time.time() - start_time
        states_explored = len(Q_table)
        return path, (states_explored, len(path) - 1 if path else 0, elapsed_time)

    def get_possible_actions(self, state):
        row, col = self.find_blank(state)
        actions = []
        if row > 0:
            actions.append(0)  # up
        if row < 2:
            actions.append(1)  # down
        if col > 0:
            actions.append(2)  # left
        if col < 2:
            actions.append(3)  # right
        return actions