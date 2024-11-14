import keras
import time
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import numpy as np
from planning_sandbox.environment.environment_class import Environment
from planning_sandbox.environment.visualizer_class import Visualizer
from utils import generate_mlp_input_from_env

class MlpEvaluator():
    def __init__(self,model=None, model_path=None, autoencoder_path=None,env: Environment=None):
        assert model is not None or model_path is not None, "Either provide model or model_path"
        if model is not None:
            self.model = model
        else:
            self.model = keras.models.load_model(model_path)

        if autoencoder_path is not None:
            self.autoencoder = keras.models.load_model(autoencoder_path)
        else:
            self.autoencoder = None

        self.env = env
    
    def encode_map(self,env=None):
        if env is None:
            env = self.env
        return self.autoencoder.encoder(np.array([self.env.grid_map.downscaled_data]).astype(np.float32))[0]
    
    def perform_visual_evaluation(self):
        assert self.env is not None, "Environment for optimal solving is not provided"
        self.env.reset()
        vis = Visualizer(self.env, speed=50)
        solved = False
        map = self.encode_map(self.env)
        failed_attempts = 0
        while not solved:
            if failed_attempts > 100:
                break
            input_array = generate_mlp_input_from_env(self.env)
            input_array = np.concatenate((input_array, map))
            input_array = input_array.reshape(1, -1)
            predictions = self.model.predict(input_array, verbose=0)  
            predictions = tf.math.round(predictions[0])
            predictions = predictions.numpy().astype(int)
            full_solution = self.env.get_full_solution_from_action_vector(predictions)
            self.env.full_solution = full_solution
            vis.visualise_full_solution(soft_reset=False)
            if self.env.scheduler.all_goals_claimed():
                solved = True
            else:
                failed_attempts += 1
                random.shuffle(self.env.goals)
    
    def perform_comparison_to_optimal_solver(self, runs):
        assert self.env is not None, "Environment for optimal solving is not provided"
        runs = runs
        successes = []
        optimal_costs = []
        predicted_costs = []
        optimal_times = []
        predicted_times = []

        pred_time = 0
        pred_cost = 0
        opt_cost = 0


        for i in range(runs):
            print(f"Run {i+1}/{runs}")
            self.env.reset()
            map = self.encode_map(self.env)
            solved = False
            optmial_found = False
            while not solved:
                # print("Input Array:", input_array)

                if not optmial_found:
                    optmial_found = True
                    opt_start = time.perf_counter_ns()
                    optimal_solution, new_opt_cost = self.env.find_numerical_solution(solve_type='optimal')
                    opt_end = time.perf_counter_ns()
                    self.env.solve_full_solution(fast=False, soft_reset=True)
                    opt_cost = self.env.get_agent_benchmarks()[2]
                    self.env.soft_reset()

                input_array = generate_mlp_input_from_env(self.env)
                input_array = np.concatenate((input_array, map))
                input_array = input_array.reshape(1, -1)
                pred_start = time.perf_counter_ns()
                predictions = self.model.predict(input_array, verbose=0)  
                pred_end = time.perf_counter_ns()
                pred_time += (pred_end - pred_start) / 1e6
                predictions = tf.math.round(predictions[0])
                predictions = predictions.numpy().astype(int)
                full_solution = self.env.get_full_solution_from_action_vector(predictions)
                self.env.full_solution = full_solution
                self.env.solve_full_solution(fast=False, soft_reset=False)
                pred_cost = self.env.get_agent_benchmarks()[2]
                
                if self.env.scheduler.all_goals_claimed():
                    solved = True
                    successes.append(1)
                    optimal_costs.append(opt_cost)
                    predicted_costs.append(pred_cost)
                    optimal_times.append((opt_end - opt_start) / 1e6)
                    predicted_times.append(pred_time)
                    print(" ====== RESULTS ====== ")
                    print(f"Optimal Cost: {opt_cost}")
                    print(f"Predicted Cost: {pred_cost}")
                    print(f"Cost Difference: {pred_cost-opt_cost}")
                    print(f"Optimal Time: {(opt_end - opt_start) / 1e6} ms")
                    print(f"Predicted Time: {pred_time} ms")
                    print(" ===================== ")
                    opt_cost = 0
                    pred_time = 0
                    pred_cost = 0
                else:
                    random.shuffle(self.env.goals)

        print(f"Success Rate: {sum(successes)/len(successes)}")

        # Compare costs and times of optimal and predicted solutions
        optimal_costs = np.array(optimal_costs)
        predicted_costs = np.array(predicted_costs)
        cost_diff = predicted_costs - optimal_costs

        optimal_times = np.array(optimal_times)
        predicted_times = np.array(predicted_times)
        time_diff = predicted_times - optimal_times

        print(f"Mean Optimal Cost: {np.mean(optimal_costs)}")
        print(f"Mean Predicted Cost: {np.mean(predicted_costs)}")
        print(f"Mean Optimal Time: {np.mean(optimal_times)} ms")
        print(f"Mean Predicted Time: {np.mean(predicted_times)} ms")

        print(f"Mean Cost Difference: {np.mean(cost_diff)}")
        print(f"Max Cost Difference: {np.max(cost_diff)}")
        print(f"Min Cost Difference: {np.min(cost_diff)}")
        print(f"Cost Difference Standard Deviation: {np.std(cost_diff)}")
        print(f"Cost Difference Variance: {np.var(cost_diff)}")
        print(f"Cost Difference Median: {np.median(cost_diff)}")

        # ===========================
        # Plotting code starts here
        # ===========================

        import matplotlib.pyplot as plt  # Already imported at the top


        # Plot settings for better visuals in a publication
        plt.style.use('seaborn-v0_8-darkgrid')  # Use an available style
        plt.rcParams.update({
            'figure.figsize': (10, 6),
            'figure.dpi': 200,
            'axes.titlesize': 24,   # Title font size
            'axes.labelsize': 20,   # x and y labels font size
            'xtick.labelsize': 20,  # x-axis tick font size
            'ytick.labelsize': 20,  # y-axis tick font size
            'legend.fontsize': 20,  # Legend font size
        })

        env_params = f'Agents: {self.env.num_agents}, Goals: {self.env.num_goals}, Skills: {self.env.num_skills}, Size: {self.env.size}, Runs: {runs}'

        # 1. Bar Chart of Mean Computation Times
        plt.figure()
        times = [np.mean(optimal_times), np.mean(predicted_times)]
        labels = ['Expert Solution', 'NN Model']
        plt.bar(labels, times, color=['blue', 'green'])
        plt.title('Mean Computation Times')
        plt.ylabel('Time (ms)')
        plt.text(0.5, max(times)*0.9, env_params, ha='center', fontsize=12)
        plt.savefig('mean_computation_times.png')
        plt.close()
        print("Saved 'mean_computation_times.png'.")

        # 2. Boxplot of Cost Differences
        plt.figure()
        plt.boxplot(cost_diff, vert=False)
        plt.title('Cost Difference Between NN Model and Expert Solution')
        plt.xlabel('Cost Difference (NN Model Cost - Expert Cost)')
        plt.text(np.mean(cost_diff), 1.05, env_params, ha='center', fontsize=12)
        plt.savefig('cost_difference_boxplot.png')
        plt.close()
        print("Saved 'cost_difference_boxplot.png'.")

        # 3. Bar Chart of Mean Costs
        plt.figure()
        costs = [np.mean(optimal_costs), np.mean(predicted_costs)]
        labels = ['Expert Solution', 'NN Model']
        plt.bar(labels, costs, color=['blue', 'green'])
        plt.title('Mean Costs')
        plt.ylabel('Total Cost (units)')
        plt.text(0.5, max(costs)*0.9, env_params, ha='center', fontsize=12)
        plt.savefig('mean_costs.png')
        plt.close()
        print("Saved 'mean_costs.png'.")

        # 4. Scatter Plot of Computation Time Difference vs. Cost Difference
        plt.figure()
        plt.scatter(cost_diff, time_diff, color='purple', alpha=0.7)
        plt.title('Computation Time Difference vs. Cost Difference (NN Model Time - Expert Time)')
        plt.xlabel('Cost Difference (NN Model Cost - Expert Cost)')
        plt.ylabel('Computation Time Difference (ms)')
        plt.axhline(0, color='grey', linestyle='--', linewidth=1)
        plt.axvline(0, color='grey', linestyle='--', linewidth=1)
        plt.text(0.5, max(time_diff)*0.9, env_params, ha='center', fontsize=12)
        plt.savefig('time_vs_cost_difference.png')
        plt.close()
        print("Saved 'time_vs_cost_difference.png'.")