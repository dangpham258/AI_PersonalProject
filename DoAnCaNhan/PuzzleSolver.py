import copy
import random
from collections import defaultdict
import heapq
from collections import deque
import math
import time
from heapq import heappush, heappop
from itertools import count


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
        def dfs(state, path, depth, visited):
            if depth < 0:
                return None
            if self.is_goal(state):
                return path + [state]
            visited.add(str(state))
            for neighbor in self.get_neighbor(state):
                neighbor_str = str(neighbor)
                if neighbor_str not in visited:
                    result = dfs(neighbor, path + [state], depth - 1, visited)
                    if result is not None:
                        return result
            return None

        total_visited = set()
        start_time = time.time()
        for depth in range(max_depth + 1):
            visited = set()
            result = dfs(initial_state, [], depth, visited)
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
            visited.add(str(state))
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
        max_depth = 50
        state_count = 1
        start_time = time.time()
        global_visited = {str(initial_state)}

        def manhattan_distance_after_move(state, direction):
            blank_row, blank_col = self.find_blank(state)
            dr, dc = direction
            new_row, new_col = blank_row + dr, blank_col + dc
            if not (0 <= new_row < 3 and 0 <= new_col < 3):
                return float('inf')
            new_state = copy.deepcopy(state)
            new_state[blank_row][blank_col], new_state[new_row][new_col] = new_state[new_row][new_col], new_state[blank_row][blank_col]
            return self.manhattan_distance(new_state)

        def dfs_and(state, depth, path, local_visited, current_cost):
            nonlocal state_count, global_visited
            if depth > max_depth:
                return None
            if self.is_goal(state):
                return path + [state]

            blank_row, blank_col = self.find_blank(state)
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            directions.sort(key=lambda d: manhattan_distance_after_move(state, d))

            for dr, dc in directions:
                new_row, new_col = blank_row + dr, blank_col + dc
                if not (0 <= new_row < 3 and 0 <= new_col < 3):
                    continue

                possible_outcomes = []
                local_copy = set(local_visited)

                # Outcome 1: Di chuyển thành công
                outcome1 = copy.deepcopy(state)
                outcome1[blank_row][blank_col], outcome1[new_row][new_col] = outcome1[new_row][new_col], outcome1[blank_row][blank_col]
                outcome1_str = str(outcome1)
                if outcome1_str not in global_visited:
                    global_visited.add(outcome1_str)
                    local_copy.add(outcome1_str)
                    state_count += 1
                    possible_outcomes.append((outcome1, local_copy))

                # Outcome 2: Đứng im
                outcome2 = state  # Không cần copy vì trạng thái không thay đổi
                outcome2_str = str(outcome2)
                if outcome2_str not in global_visited:
                    global_visited.add(outcome2_str)
                    local_copy.add(outcome2_str)
                    state_count += 1
                    possible_outcomes.append((outcome2, local_copy))

                if possible_outcomes:
                    all_paths = []
                    valid_for_all = True
                    for outcome, new_local_visited in possible_outcomes:
                        heuristic_cost = self.manhattan_distance(outcome)
                        if current_cost + heuristic_cost > 100:
                            valid_for_all = False
                            break
                        result = dfs_and(outcome, depth + 1, path + [state], new_local_visited, current_cost + 1)
                        if result is None:
                            valid_for_all = False
                            break
                        all_paths.append(result)
                    if valid_for_all and all_paths:
                        return min(all_paths, key=len)
            return None

        result = dfs_and(initial_state, 0, [], {str(initial_state)}, 0)
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

    def partially_observable_search(self, initial_state, max_depth=30):
        def get_observation(state):
            return self.find_blank(state)

        initial_belief = {str(initial_state): initial_state}
        start_obs = get_observation(initial_state)
        g0 = 0
        h0 = self.manhattan_distance(initial_state)
        counter = count()
        open_heap = []
        heappush(open_heap, (g0 + h0, g0, next(counter), initial_belief, [], initial_state))

        visited = set()
        states_explored = 1
        start_time = time.time()

        while open_heap:
            f, g, _, belief, plan, actual = heappop(open_heap)
            key = (frozenset(belief.keys()), len(plan))
            if key in visited:
                continue
            visited.add(key)

            if all(self.is_goal(s) for s in belief.values()):
                path = [initial_state]
                curr = initial_state
                for act in plan:
                    res = self.results(curr, act)
                    curr = res[0] if res else curr
                    path.append(curr)
                elapsed = time.time() - start_time
                return path, (states_explored, len(path) - 1, elapsed)

            if g >= max_depth:
                continue

            actions = self.actions()
            sorted_actions = []
            for a in actions:
                res_actual = self.results(actual, a)
                if res_actual:
                    next_act = res_actual[0]
                    score = self.manhattan_distance(next_act)
                else:
                    score = float('inf')
                sorted_actions.append((score, a))
            sorted_actions.sort(key=lambda x: x[0])

            for _, action in sorted_actions:
                actual_res = self.results(actual, action)
                if not actual_res:
                    continue
                next_actual = actual_res[0]
                obs = get_observation(next_actual)

                next_belief = {}
                for s in belief.values():
                    for s2 in self.results(s, action):
                        states_explored += 1
                        if get_observation(s2) == obs:
                            next_belief[str(s2)] = s2
                if not next_belief:
                    continue

                new_plan = plan + [action]
                new_key = (frozenset(next_belief.keys()), len(new_plan))
                if new_key in visited:
                    continue

                g2 = g + 1
                h2 = self.manhattan_distance(next_actual)
                heappush(open_heap, (g2 + h2, g2, next(counter), next_belief, new_plan, next_actual))

        elapsed = time.time() - start_time
        return None, (states_explored, 0, elapsed)
    
    def is_solvable(self, state):
        flat = [tile for row in state for tile in row if tile != 0]
        inversions = 0
        for i in range(len(flat)):
            for j in range(i + 1, len(flat)):
                if flat[i] > flat[j]:
                    inversions += 1
        return inversions % 2 == 0

    def backtracking_search(self, initial_state, max_depth=50, ui=None):
        states_explored = 0
        
        def backtrack(state, path, depth, visited_in_path):
            nonlocal states_explored
            states_explored += 1

            if ui:
                ui.update_status(f"Backtracking: Exploring depth {depth}, States explored: {states_explored}")
                assignment = {(i, j): state[i][j] for i in range(3) for j in range(3)}
                ui.update_board_from_assignment(assignment)
                time.sleep(0.005)
                if not ui.solving:
                    return None

            if self.is_goal(state):
                if ui:
                    ui.update_status(f"Solution found at depth {depth}!")
                    time.sleep(0.5)
                return path

            if depth >= max_depth or depth + self.manhattan_distance(state) > max_depth:
                if ui:
                    ui.update_status(f"Pruning at depth {depth}: Path too long or unreachable")
                    time.sleep(0.005)
                return None
            
            visited_in_path.add(str(state))
            neighbors = self.get_neighbor(state)
            random.shuffle(neighbors)
            for neighbor in neighbors:
                if str(neighbor) not in visited_in_path:
                    result = backtrack(neighbor, path + [neighbor], depth + 1, visited_in_path)
                    if result is not None:
                        return result
                    if ui:
                        ui.update_status(f"Backtracking to depth {depth}, States explored: {states_explored}")
                        assignment = {(i, j): state[i][j] for i in range(3) for j in range(3)}
                        ui.update_board_from_assignment(assignment)
                        time.sleep(0.005)
                        if not ui.solving:
                            return None
            visited_in_path.remove(str(state))
            return None

        if not self.is_solvable(initial_state):
            if ui:
                ui.update_status("Puzzle is unsolvable!")
                time.sleep(0.5)
            return None, (0, 0, time.time() - time.time())
        
        path = [initial_state]
        visited_in_path = set()
        start_time = time.time()
        solution = backtrack(initial_state, path, 0, visited_in_path)
        elapsed = time.time() - start_time
        
        if solution is None:
            if ui:
                ui.update_status("No solution found within depth limit")
                time.sleep(0.005)
            return None, (states_explored, 0, elapsed)
        else:
            if ui:
                ui.update_status(f"Solution found in {states_explored} states!")
                for step, state in enumerate(solution):
                    assignment = {(i, j): state[i][j] for i in range(3) for j in range(3)}
                    ui.update_board_from_assignment(assignment)
                    ui.update_status(f"Solution step {step + 1}/{len(solution)}")
                    time.sleep(0.05)
            return solution, (states_explored, len(solution) - 1, elapsed)
    
    def forward_checking(self, initial_state, max_depth=50, ui=None):
        states_explored = 0
        
        def forward_check(state, depth):
            remaining_steps = max_depth - depth
            return self.manhattan_distance(state) <= remaining_steps
        
        def backtrack(state, path, depth, visited_in_path):
            nonlocal states_explored
            states_explored += 1
            
            if ui:
                ui.update_status(f"Forward Checking: Exploring depth {depth}, States explored: {states_explored}")
                assignment = {(i, j): state[i][j] for i in range(3) for j in range(3)}
                ui.update_board_from_assignment(assignment)
                time.sleep(0.005)
                if not ui.solving:
                    return None

            if self.is_goal(state):
                if ui:
                    ui.update_status(f"Solution found at depth {depth}!")
                    time.sleep(0.5)
                return path

            if depth >= max_depth or not forward_check(state, depth):
                if ui:
                    ui.update_status(f"Pruning at depth {depth}: Unreachable or too deep")
                    time.sleep(0.005)
                return None
            
            visited_in_path.add(str(state))
            neighbors = self.get_neighbor(state)
            neighbors.sort(key=self.manhattan_distance)
            for neighbor in neighbors:
                if str(neighbor) not in visited_in_path:
                    result = backtrack(neighbor, path + [neighbor], depth + 1, visited_in_path)
                    if result is not None:
                        return result
                    if ui:
                        ui.update_status(f"Backtracking to depth {depth}, States explored: {states_explored}")
                        assignment = {(i, j): state[i][j] for i in range(3) for j in range(3)}
                        ui.update_board_from_assignment(assignment)
                        time.sleep(0.005)
                        if not ui.solving:
                            return None
            visited_in_path.remove(str(state))
            return None

        if not self.is_solvable(initial_state):
            if ui:
                ui.update_status("Puzzle is unsolvable!")
                time.sleep(0.5)
            return None, (0, 0, time.time() - time.time())
        
        path = [initial_state]
        visited_in_path = set()
        start_time = time.time()
        solution = backtrack(initial_state, path, 0, visited_in_path)
        elapsed = time.time() - start_time
        
        if solution is None:
            if ui:
                ui.update_status("No solution found within depth limit")
                time.sleep(0.005)
            return None, (states_explored, 0, elapsed)
        else:
            if ui:
                ui.update_status(f"Solution found in {states_explored} states!")
                for step, state in enumerate(solution):
                    assignment = {(i, j): state[i][j] for i in range(3) for j in range(3)}
                    ui.update_board_from_assignment(assignment)
                    ui.update_status(f"Solution step {step + 1}/{len(solution)}")
                    time.sleep(0.05)
            return solution, (states_explored, len(solution) - 1, elapsed)

    def min_conflicts(self, initial_state, max_steps=1000, ui=None):
        current_state = copy.deepcopy(initial_state)
        path = [current_state]
        visited = {str(current_state)}
        steps = 0
        states_explored = 1
        start_time = time.time()

        def conflicts(state):
            return self.manhattan_distance(state)

        while steps < max_steps:
            steps += 1

            if ui:
                ui.update_status(f"Min-Conflicts: Step {steps}, Manhattan Distance: {conflicts(current_state)}")
                assignment = {(i, j): current_state[i][j] for i in range(3) for j in range(3)}
                ui.update_board_from_assignment(assignment)
                time.sleep(0.01)
                if not ui.solving:
                    elapsed = time.time() - start_time
                    return None, (states_explored, len(path) - 1, elapsed)

            if self.is_goal(current_state):
                if ui:
                    ui.update_status("Solution found!")
                elapsed = time.time() - start_time
                return path, (states_explored, len(path) - 1, elapsed)

            neighbors = self.get_neighbor(current_state)
            valid_neighbors = [(n, conflicts(n)) for n in neighbors if str(n) not in visited]
            if not valid_neighbors:
                neighbors = self.get_neighbor(current_state)
                current_state = random.choice(neighbors)
                path.append(current_state)
                visited.add(str(current_state))
                states_explored += 1
                continue

            min_conflict_state = min(valid_neighbors, key=lambda x: x[1])[0]
            current_state = min_conflict_state
            path.append(current_state)
            visited.add(str(current_state))
            states_explored += 1

        elapsed = time.time() - start_time
        if ui:
            ui.update_status("No solution found after maximum steps")
        return None, (states_explored, len(path) - 1, elapsed)

    def q_learning(self, initial_state):
        Q_table = defaultdict(lambda: [0, 0, 0, 0])
        alpha = 0.1                   # learning rate
        gamma = 0.9                   # discount factor
        epsilon_start = 1.0
        epsilon_end = 0.1
        total_episodes = 50000       
        max_steps = 100
        start_time = time.time()

        for episode in range(total_episodes):
            state = self.generate_random_state()
            state_tuple = tuple(map(tuple, state))

            epsilon = max(
                epsilon_end,
                epsilon_start - (epsilon_start - epsilon_end) * episode / total_episodes
            )

            for step in range(max_steps):
                possible_actions = self.get_possible_actions(state)

                if random.random() < epsilon:
                    action = random.choice(possible_actions)
                else:
                    q_vals = Q_table[state_tuple]
                    best_actions = sorted(
                        possible_actions,
                        key=lambda a: q_vals[a] + random.random() * 1e-5,
                        reverse=True
                    )
                    action = best_actions[0]

                next_states = self.results(state, action)
                assert len(next_states) == 1, "Expected deterministic result"
                next_state = next_states[0]
                next_tuple = tuple(map(tuple, next_state))

                if self.is_goal(next_state):
                    reward = 100
                else:
                    reward = -self.manhattan_distance(next_state)

                if self.is_goal(next_state):
                    target = reward
                else:
                    future_actions = self.get_possible_actions(next_state)
                    if future_actions:
                        max_q_next = max(Q_table[next_tuple][a] for a in future_actions)
                    else:
                        max_q_next = 0
                    target = reward + gamma * max_q_next

                Q_table[state_tuple][action] += alpha * (target - Q_table[state_tuple][action])

                state, state_tuple = next_state, next_tuple

                if self.is_goal(state):
                    break

        state = initial_state
        path = [state]
        visited = {tuple(map(tuple, state))}

        for _ in range(100):
            if self.is_goal(state):
                break

            state_tuple = tuple(map(tuple, state))
            possible_actions = self.get_possible_actions(state)
            if not possible_actions:
                break

            q_vals = Q_table[state_tuple]
            best_actions = sorted(
                possible_actions,
                key=lambda a: q_vals[a],
                reverse=True
            )
            action = best_actions[0]

            next_states = self.results(state, action)
            assert len(next_states) == 1
            next_state = next_states[0]
            next_tuple = tuple(map(tuple, next_state))

            if next_tuple in visited:
                break

            path.append(next_state)
            visited.add(next_tuple)
            state = next_state

        if not self.is_goal(state):
            path = None

        elapsed_time = time.time() - start_time
        states_explored = len(Q_table)
        solution_length = len(path) - 1 if path else 0

        return path, (states_explored, solution_length, elapsed_time)

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