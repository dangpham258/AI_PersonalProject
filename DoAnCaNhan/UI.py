import tkinter as tk
from tkinter import ttk, messagebox
import copy
import threading
import time
from PuzzleSolver import PuzzleSolver
from Graph import PuzzleVisualizer

class EightPuzzle:
    def __init__(self, root):
        self.root = root
        self.root.title("8-Puzzle")
        self.root.geometry("800x1000")
        self.root.resizable(False, False)

        self.BG_COLOR = "#f5f5dc"
        self.BOARD_COLOR = "#d2b48c"
        self.TILE_COLOR = "#8b4513"
        self.TEXT_COLOR = "#000000"
        self.BUTTON_COLOR = "#d2691e"
        self.BUTTON_TEXT_COLOR = "#ffffff"

        self.INITIAL_STATE = [[2, 6, 5], [0, 8, 7], [4, 3, 1]]
        self.GOAL_STATE = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]

        self.solver = PuzzleSolver(self.GOAL_STATE)
        self.current_state = copy.deepcopy(self.INITIAL_STATE)
        self.solution = None
        self.solution_index = 0
        self.solving = False
        self.solution_display_job = None
        self.timer_job = None
        self.start_time = 0
        self.state_space_size = 0
        self.last_random_state = None
        self.performance_data = {}
        
        # Initialize visualizer of Graph.py
        self.visualizer = PuzzleVisualizer(self.root)

        self.setup_ui()

    def setup_ui(self):
        self.root.configure(bg=self.BG_COLOR)
        main_frame = tk.Frame(self.root, bg=self.BG_COLOR)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=30)

        title_label = tk.Label(main_frame, text="8-Puzzle", font=("Arial", 24, "bold"), bg=self.BG_COLOR, fg=self.TEXT_COLOR)
        title_label.pack(pady=10)

        self.board_frame = tk.Frame(main_frame, bg=self.BOARD_COLOR, width=300, height=300, bd=2, relief=tk.RAISED)
        self.board_frame.pack(pady=15)

        self.cells = []
        for i in range(3):
            row = []
            for j in range(3):
                cell_frame = tk.Frame(self.board_frame, width=100, height=100, bg=self.BOARD_COLOR, bd=2, relief=tk.SUNKEN)
                cell_frame.grid(row=i, column=j)
                cell_frame.grid_propagate(False)
                cell_label = tk.Label(cell_frame, font=("Arial", 36, "bold"), bg=self.BOARD_COLOR, fg=self.TEXT_COLOR)
                cell_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
                row.append(cell_label)
            self.cells.append(row)

        status_frame = tk.Frame(main_frame, bg=self.BG_COLOR)
        status_frame.pack(fill=tk.X, pady=10)
        self.status_var = tk.StringVar(value="Ready")
        status_label = tk.Label(status_frame, textvariable=self.status_var, font=("Arial", 12), bg=self.BG_COLOR, fg=self.TEXT_COLOR)
        status_label.pack(side=tk.LEFT, padx=5)
        self.time_var = tk.StringVar(value="Time: 0.0000s")
        time_label = tk.Label(status_frame, textvariable=self.time_var, font=("Arial", 12), bg=self.BG_COLOR, fg=self.TEXT_COLOR)
        time_label.pack(side=tk.LEFT, padx=5)
        self.space_var = tk.StringVar(value="States: 0")
        space_label = tk.Label(status_frame, textvariable=self.space_var, font=("Arial", 12), bg=self.BG_COLOR, fg=self.TEXT_COLOR)
        space_label.pack(side=tk.LEFT, padx=5)

        style = ttk.Style()
        style.configure('TNotebook.Tab', font=('Arial', 14), padding=[20, 10])

        algorithm_notebook = ttk.Notebook(main_frame)
        algorithm_notebook.pack(fill=tk.X, pady=10)

        # Uninformed Search Tab
        uninformed_frame = tk.Frame(algorithm_notebook, bg=self.BG_COLOR)
        algorithm_notebook.add(uninformed_frame, text="Uninformed")
        uninformed_buttons_frame = tk.Frame(uninformed_frame, bg=self.BG_COLOR)
        uninformed_buttons_frame.pack(fill=tk.X, padx=10, pady=10)
        uninformed_algorithms = [
            ("BFS", self.solve_bfs),
            ("DFS", self.solve_dfs),
            ("IDS", self.solve_ids),
            ("UCS", self.solve_ucs)
        ]
        for i, (text, command) in enumerate(uninformed_algorithms):
            btn = tk.Button(uninformed_buttons_frame, text=text, command=command, font=("Arial", 11), bg=self.BUTTON_COLOR, fg=self.BUTTON_TEXT_COLOR, width=20, height=2)
            btn.grid(row=0, column=i, padx=5, pady=5)
            uninformed_buttons_frame.columnconfigure(i, weight=1)

        # Informed Search Tab
        informed_frame = tk.Frame(algorithm_notebook, bg=self.BG_COLOR)
        algorithm_notebook.add(informed_frame, text="Informed")
        informed_buttons_frame = tk.Frame(informed_frame, bg=self.BG_COLOR)
        informed_buttons_frame.pack(fill=tk.X, padx=10, pady=10)
        informed_algorithms = [
            ("Greedy", self.solve_greedy),
            ("A*", self.solve_astar),
            ("IDA*", self.solve_idastar)
        ]
        for i, (text, command) in enumerate(informed_algorithms):
            btn = tk.Button(informed_buttons_frame, text=text, command=command, font=("Arial", 11), bg=self.BUTTON_COLOR, fg=self.BUTTON_TEXT_COLOR, width=20, height=2)
            btn.grid(row=0, column=i, padx=5, pady=5)
            informed_buttons_frame.columnconfigure(i, weight=1)

        # Local Search Tab
        local_frame = tk.Frame(algorithm_notebook, bg=self.BG_COLOR)
        algorithm_notebook.add(local_frame, text="Local Search")
        local_buttons_frame = tk.Frame(local_frame, bg=self.BG_COLOR)
        local_buttons_frame.pack(fill=tk.X, padx=10, pady=10)
        local_algorithms = [
            ("Simple HC", self.solve_simple_hill_climbing),
            ("Steepest HC", self.solve_steepest_hill_climbing),
            ("Stochastic HC", self.solve_stochastic_hill_climbing),
            ("Sim.Annealing", self.solve_simulated_annealing),
            ("Beam Search", self.solve_beam_search),
            ("Genetic Algo", self.solve_genetic_algorithm)
        ]
        for i, (text, command) in enumerate(local_algorithms):
            btn = tk.Button(local_buttons_frame, text=text, command=command, font=("Arial", 11), bg=self.BUTTON_COLOR, fg=self.BUTTON_TEXT_COLOR, width=20, height=2)
            btn.grid(row=i // 3, column=i % 3, padx=5, pady=5)
        local_buttons_frame.columnconfigure(0, weight=1)
        local_buttons_frame.columnconfigure(1, weight=1)
        local_buttons_frame.columnconfigure(2, weight=1)
        local_buttons_frame.rowconfigure(0, weight=1)
        local_buttons_frame.rowconfigure(1, weight=1)

        # Complex Search Tab
        complex_frame = tk.Frame(algorithm_notebook, bg=self.BG_COLOR)
        algorithm_notebook.add(complex_frame, text="Complex")
        complex_buttons_frame = tk.Frame(complex_frame, bg=self.BG_COLOR)
        complex_buttons_frame.pack(fill=tk.X, padx=10, pady=10)
        complex_algorithms = [
            ("Nondeterministic", self.solve_nondeterministic),
            ("No Observations", self.solve_no_observations),
            ("Part. Observable", self.solve_partially_observable)
        ]
        for i, (text, command) in enumerate(complex_algorithms):
            btn = tk.Button(complex_buttons_frame, text=text, command=command, font=("Arial", 11), bg=self.BUTTON_COLOR, fg=self.BUTTON_TEXT_COLOR, width=20, height=2)
            btn.grid(row=0, column=i, padx=5, pady=5)
        complex_buttons_frame.columnconfigure(0, weight=1)
        complex_buttons_frame.columnconfigure(1, weight=1)
        complex_buttons_frame.columnconfigure(2, weight=1)

        # CSPs Tab
        csp_frame = tk.Frame(algorithm_notebook, bg=self.BG_COLOR)
        algorithm_notebook.add(csp_frame, text="CSPs")
        csp_buttons_frame = tk.Frame(csp_frame, bg=self.BG_COLOR)
        csp_buttons_frame.pack(fill=tk.X, padx=10, pady=10)
        csp_algorithms = [
            ("Min-Conflicts", self.solve_min_conflicts),
            ("Forward Checking", self.solve_forward_checking),
            ("Backtracking", self.solve_backtracking)
        ]
        for i, (text, command) in enumerate(csp_algorithms):
            btn = tk.Button(csp_buttons_frame, text=text, command=command, font=("Arial", 11), bg=self.BUTTON_COLOR, fg=self.BUTTON_TEXT_COLOR, width=20, height=2)
            btn.grid(row=0, column=i, padx=5, pady=5)
        csp_buttons_frame.columnconfigure(0, weight=1)
        csp_buttons_frame.columnconfigure(1, weight=1)
        csp_buttons_frame.columnconfigure(2, weight=1)

        # Reinforcement Learning Tab
        rl_frame = tk.Frame(algorithm_notebook, bg=self.BG_COLOR)
        algorithm_notebook.add(rl_frame, text="RL")
        rl_buttons_frame = tk.Frame(rl_frame, bg=self.BG_COLOR)
        rl_buttons_frame.pack(fill=tk.X, padx=10, pady=10)
        qlearning_btn = tk.Button(rl_buttons_frame, text="Q-Learning", command=self.solve_q_learning, font=("Arial", 11), bg=self.BUTTON_COLOR, fg=self.BUTTON_TEXT_COLOR, width=20, height=2)
        qlearning_btn.grid(row=0, column=0, padx=5, pady=5)
        rl_buttons_frame.columnconfigure(0, weight=1)

        # Control frame
        control_frame = tk.Frame(main_frame, bg=self.BG_COLOR)
        control_frame.pack(fill=tk.X, pady=10)

        control_buttons_frame = tk.Frame(control_frame, bg=self.BG_COLOR)
        control_buttons_frame.pack(fill=tk.X)

        buttons_row1 = tk.Frame(control_buttons_frame, bg=self.BG_COLOR)
        buttons_row1.pack(fill=tk.X, pady=5)

        reset_btn = tk.Button(buttons_row1, text="Reset", command=self.reset, 
                              font=("Arial", 12), bg=self.BUTTON_COLOR, fg=self.BUTTON_TEXT_COLOR, width=15, height=2)
        reset_btn.grid(row=0, column=0, padx=5, pady=5)

        self.play_btn = tk.Button(buttons_row1, text="Run", command=self.play_solution, 
                                  font=("Arial", 12), bg=self.BUTTON_COLOR, fg=self.BUTTON_TEXT_COLOR, width=15, height=2, state=tk.DISABLED)
        self.play_btn.grid(row=0, column=1, padx=5, pady=5)

        random_btn = tk.Button(buttons_row1, text="Random", command=self.randomize_state, 
                               font=("Arial", 12), bg=self.BUTTON_COLOR, fg=self.BUTTON_TEXT_COLOR, width=15, height=2)
        random_btn.grid(row=0, column=2, padx=5, pady=5)

        buttons_row1.columnconfigure(0, weight=1)
        buttons_row1.columnconfigure(1, weight=1)
        buttons_row1.columnconfigure(2, weight=1)

        buttons_row2 = tk.Frame(control_buttons_frame, bg=self.BG_COLOR)
        buttons_row2.pack(fill=tk.X, pady=5)

        reset_random_btn = tk.Button(buttons_row2, text="Reset Random", command=self.reset_to_last_random, 
                                     font=("Arial", 12), bg=self.BUTTON_COLOR, fg=self.BUTTON_TEXT_COLOR, width=20, height=2)
        reset_random_btn.grid(row=0, column=0, padx=5, pady=5)

        compare_btn = tk.Button(buttons_row2, text="Compare Algorithms", command=self.compare_algorithms, 
                                font=("Arial", 12), bg=self.BUTTON_COLOR, fg=self.BUTTON_TEXT_COLOR, width=20, height=2)
        compare_btn.grid(row=0, column=1, padx=5, pady=5)

        compare_csp_btn = tk.Button(buttons_row2, text="Compare CSPs", command=self.compare_csps, 
                                    font=("Arial", 12), bg=self.BUTTON_COLOR, fg=self.BUTTON_TEXT_COLOR, width=20, height=2)
        compare_csp_btn.grid(row=0, column=2, padx=5, pady=5)

        buttons_row2.columnconfigure(0, weight=1)
        buttons_row2.columnconfigure(1, weight=1)
        buttons_row2.columnconfigure(2, weight=1)

        self.update_board()

    def update_board(self):
        for i in range(3):
            for j in range(3):
                value = self.current_state[i][j]
                if value == 0:
                    self.cells[i][j].config(text="")
                else:
                    self.cells[i][j].config(text=str(value))

    def update_status(self, message):
        self.status_var.set(message)
        self.root.update()

    def update_board_from_assignment(self, assignment):
        for i in range(3):
            for j in range(3):
                val = assignment.get((i, j), 0)
                self.cells[i][j].config(text=str(val) if val != 0 else "")
        self.root.update()

    def randomize_state(self):
        if self.solving:
            return
        self.current_state = self.solver.generate_random_state()
        self.last_random_state = copy.deepcopy(self.current_state)
        self.update_board()
        self.status_var.set("Generated random initial state")

    def reset_to_last_random(self):
        if self.solving:
            self.solving = False
            self.update_status("Stopping algorithm...")
            self.root.after(100, self.complete_reset_to_random)
        else:
            self.complete_reset_to_random()

    def complete_reset_to_random(self):
        if self.last_random_state is None:
            self.status_var.set("No random state generated yet")
            return
        self.current_state = copy.deepcopy(self.last_random_state)
        self.update_board()
        self.reset_solution()

    def reset(self):
        if self.solving:
            self.solving = False
            self.update_status("Stopping algorithm...")
            self.root.after(100, self.complete_reset)
        else:
            self.complete_reset()

    def complete_reset(self):
        self.current_state = copy.deepcopy(self.INITIAL_STATE)
        self.update_board()
        self.reset_solution()

    def reset_solution(self):
        self.solution = None
        self.solution_index = 0
        self.play_btn.config(state=tk.DISABLED)
        self.status_var.set("Ready")
        self.time_var.set("Time: 0.0000s")
        self.space_var.set("States: 0")
        if self.solution_display_job:
            self.root.after_cancel(self.solution_display_job)
            self.solution_display_job = None

    def solve_with_algorithm(self, algorithm, algorithm_name):
        if self.solving:
            return
        self.reset_solution()
        self.status_var.set(f"Solving with {algorithm_name}...")
        self.solving = True
        self.start_time = time.time()
        self.update_timer()
        threading.Thread(target=self.run_algorithm, args=(algorithm, algorithm_name)).start()

    def run_algorithm(self, algorithm, algorithm_name):
        try:
            self.space_var.set("States: 0")
            result = algorithm(self.current_state)
            elapsed_time = time.time() - self.start_time
            self.solution = result[0] if result else None
            self.state_space_size, steps, algo_time = result[1] if result else (0, 0, elapsed_time)
            starting_state_str = str(self.current_state)
            if starting_state_str not in self.performance_data:
                self.performance_data[starting_state_str] = {}
            self.performance_data[starting_state_str][algorithm_name] = (algo_time, steps, self.state_space_size)
            if self.solution:
                self.root.after(0, lambda: self.status_var.set(f"FOUND THE SOLUTION: {steps} steps"))
                self.root.after(0, lambda: self.time_var.set(f"Time: {algo_time:.4f}s"))
                self.root.after(0, lambda: self.space_var.set(f"States: {self.state_space_size}"))
                self.root.after(0, lambda: self.play_btn.config(state=tk.NORMAL))
            else:
                self.root.after(0, lambda: self.status_var.set("NOT FOUND ANY SOLUTION"))
                self.root.after(0, lambda: self.time_var.set(f"Time: {algo_time:.4f}s"))
                self.root.after(0, lambda: self.space_var.set(f"States: {self.state_space_size}"))
        except Exception as e:
            error_msg = str(e)
            print(f"ERROR: {error_msg}")
            self.root.after(0, lambda msg=error_msg: messagebox.showerror("ERROR", f"ERROR: {msg}"))
        finally:
            self.solving = False
            if self.timer_job:
                self.root.after(0, self.cancel_timer)

    def cancel_timer(self):
        if self.timer_job:
            self.root.after_cancel(self.timer_job)
            self.timer_job = None

    def update_timer(self):
        if self.solving:
            elapsed_time = time.time() - self.start_time
            self.time_var.set(f"Time: {elapsed_time:.4f}s")
            self.timer_job = self.root.after(100, self.update_timer)

    def play_solution(self):
        if not self.solution or self.solution_index >= len(self.solution):
            self.status_var.set("Solution playback complete")
            self.play_btn.config(text="Run", state=tk.NORMAL)
            self.solution_index = 0
            return
        if self.solution_display_job:
            self.root.after_cancel(self.solution_display_job)
        self.play_btn.config(text="Pause", command=self.pause_solution)
        self.display_next_state()

    def pause_solution(self):
        if self.solution_display_job:
            self.root.after_cancel(self.solution_display_job)
            self.solution_display_job = None
        self.play_btn.config(text="Run", command=self.play_solution)

    def display_next_state(self):
        if self.solution_index < len(self.solution):
            self.current_state = copy.deepcopy(self.solution[self.solution_index])
            self.update_board()
            self.status_var.set(f"Step {self.solution_index}/{len(self.solution)-1}")
            self.solution_index += 1
            self.solution_display_job = self.root.after(200, self.display_next_state)
        else:
            self.pause_solution()
            self.status_var.set("Solution playback complete")

    def compare_algorithms(self):
        starting_state_str = str(self.current_state)
        if starting_state_str not in self.performance_data or not self.performance_data[starting_state_str]:
            messagebox.showinfo("Info", "No algorithms have been run for this start state yet.")
            return

        data = self.performance_data[starting_state_str]
        
        algorithm_categories = {
            "Uninformed": ["BFS", "DFS", "IDS", "UCS"],
            "Informed": ["Greedy", "A*", "IDA*"],
            "Local Search": ["Simple HC", "Steepest HC", "Stochastic HC", "Simulated Annealing", "Beam Search", "Genetic Algo"],
            "Complex": ["Nondeterministic", "No Observations", "Part. Observable"],
            "All": list(data.keys())
        }
        
        category_dialog = tk.Toplevel(self.root)
        category_dialog.title("Select Algorithm Category")
        category_dialog.geometry("400x300")
        category_dialog.transient(self.root)
        category_dialog.grab_set()
        
        tk.Label(category_dialog, text="Select category to compare:", font=("Arial", 12)).pack(pady=10)
        
        selected_category = tk.StringVar(value="All")
        
        for category in algorithm_categories.keys():
            rb = tk.Radiobutton(category_dialog, text=category, variable=selected_category, value=category, font=("Arial", 11))
            rb.pack(anchor=tk.W, padx=20, pady=5)
        
        def on_submit():
            category = selected_category.get()
            algorithms = algorithm_categories[category]
            fig, summary = self.visualizer.create_comparison_chart(data, algorithms, title=f"{category} Algorithm Performance")
            if fig:
                self.visualizer.display_chart(fig, summary, window_title=f"{category} Comparison")
            else:
                messagebox.showinfo("Info", summary)
            category_dialog.destroy()
        
        submit_btn = tk.Button(category_dialog, text="Compare", command=on_submit, font=("Arial", 12), bg=self.BUTTON_COLOR, fg=self.BUTTON_TEXT_COLOR)
        submit_btn.pack(pady=20)
        
        cancel_btn = tk.Button(category_dialog, text="Cancel", command=category_dialog.destroy, font=("Arial", 12), bg=self.BUTTON_COLOR, fg=self.BUTTON_TEXT_COLOR)
        cancel_btn.pack()

    def compare_csps(self):
        starting_state_str = str(self.current_state)
        if starting_state_str not in self.performance_data or not self.performance_data[starting_state_str]:
            messagebox.showinfo("Info", "No CSP algorithms have been run for this start state yet.")
            return

        data = self.performance_data[starting_state_str]
        csp_algorithms = ["Min-Conflicts", "Forward Checking", "Backtracking"]
        
        # Check which CSP algorithms have data
        valid_algorithms = [alg for alg in csp_algorithms if alg in data]
        
        if not valid_algorithms:
            messagebox.showinfo("Info", "No CSP algorithms have been run for this start state yet.")
            return
            
        fig, summary = self.visualizer.create_comparison_chart(data, valid_algorithms, title="CSP Algorithms Comparison")
        if fig:
            self.visualizer.display_chart(fig, summary, window_title="CSP Algorithms Comparison")
        else:
            messagebox.showinfo("Info", summary)

    # Solver methods
    def solve_bfs(self):
        self.solve_with_algorithm(self.solver.bfs, "BFS")

    def solve_dfs(self):
        self.solve_with_algorithm(self.solver.dfs, "DFS")

    def solve_ids(self):
        self.solve_with_algorithm(lambda state: self.solver.ids(state, max_depth=50), "IDS")

    def solve_ucs(self):
        self.solve_with_algorithm(self.solver.ucs, "UCS")

    def solve_greedy(self):
        self.solve_with_algorithm(self.solver.greedy_search, "Greedy")

    def solve_astar(self):
        self.solve_with_algorithm(self.solver.astar_search, "A*")

    def solve_idastar(self):
        self.solve_with_algorithm(self.solver.idastar_search, "IDA*")

    def solve_simple_hill_climbing(self):
        self.solve_with_algorithm(self.solver.simple_hill_climbing, "Simple HC")

    def solve_steepest_hill_climbing(self):
        self.solve_with_algorithm(self.solver.steepest_hill_climbing, "Steepest HC")

    def solve_stochastic_hill_climbing(self):
        self.solve_with_algorithm(self.solver.stochastic_hill_climbing, "Stochastic HC")

    def solve_simulated_annealing(self):
        self.solve_with_algorithm(self.solver.simulated_annealing, "Simulated Annealing")

    def solve_beam_search(self):
        self.solve_with_algorithm(self.solver.beam_search, "Beam Search")

    def solve_genetic_algorithm(self):
        self.solve_with_algorithm(self.solver.genetic_algorithm, "Genetic Algo")

    def solve_nondeterministic(self):
        self.solve_with_algorithm(self.solver.nondeterministic_search, "Nondeterministic")

    def solve_no_observations(self):
        self.solve_with_algorithm(self.solver.no_observations_search, "No Observations")

    def solve_partially_observable(self):
        self.solve_with_algorithm(self.solver.partially_observable_search, "Part. Observable")

    def solve_min_conflicts(self):
        self.solve_with_algorithm(lambda state: self.solver.min_conflicts(state, ui=self), "Min-Conflicts")

    def solve_forward_checking(self):
        self.solve_with_algorithm(lambda state: self.solver.forward_checking(state, ui=self), "Forward Checking")

    def solve_backtracking(self):
        self.solve_with_algorithm(lambda state: self.solver.backtracking_search(state, ui=self), "Backtracking")

    def solve_q_learning(self):
        self.solve_with_algorithm(self.solver.q_learning, "Q-Learning")