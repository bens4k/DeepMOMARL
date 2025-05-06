import torch
import numpy as np

class TPBenchmark:
    def __init__(self, problem_id):
        """
        Initialize the TP benchmark problem.
        
        Args:
            problem_id: Integer from 1 to 10 indicating which problem to use
        """
        self.problem_id = problem_id
        
        # Set dimensions based on the problem
        self.dimensions = self._get_dimensions()
        self.n = self.dimensions['n']  # Upper-level dimension
        self.m = self.dimensions['m']  # Lower-level dimension
        
        # Set variable bounds and constraints
        self.bounds = self._get_bounds()
        
    def _get_dimensions(self):
        """Define dimensions for each problem"""
        dimensions = {}
        
        if self.problem_id in [1, 2, 3, 5]:
            dimensions['n'] = 2
            dimensions['m'] = 2
        elif self.problem_id == 4:
            dimensions['n'] = 2
            dimensions['m'] = 3
        elif self.problem_id == 6:
            dimensions['n'] = 1
            dimensions['m'] = 2
        elif self.problem_id == 7:
            dimensions['n'] = 2
            dimensions['m'] = 2
        elif self.problem_id == 8:
            dimensions['n'] = 2
            dimensions['m'] = 2
        elif self.problem_id in [9, 10]:
            dimensions['n'] = 10
            dimensions['m'] = 10
            
        return dimensions
    
    def _get_bounds(self):
        """Define variable bounds for each problem"""
        bounds = {}
        
        if self.problem_id == 1:
            bounds['x_u'] = {'lower': torch.zeros(self.n), 'upper': torch.tensor([30.0, 20.0])}
            bounds['x_l'] = {'lower': torch.zeros(self.m), 'upper': torch.tensor([10.0, 10.0])}
            
        elif self.problem_id == 2:
            bounds['x_u'] = {'lower': torch.zeros(self.n), 'upper': torch.tensor([50.0, 50.0])}
            bounds['x_l'] = {'lower': torch.tensor([-10.0, -10.0]), 'upper': torch.tensor([50.0, 50.0])}
            
        elif self.problem_id == 3:
            bounds['x_u'] = {'lower': torch.zeros(self.n), 'upper': torch.tensor([15.0, 15.0])}
            bounds['x_l'] = {'lower': torch.zeros(self.m), 'upper': torch.tensor([15.0, 15.0])}
            
        elif self.problem_id == 4:
            bounds['x_u'] = {'lower': torch.zeros(self.n), 'upper': torch.tensor([1.0, 1.0])}
            bounds['x_l'] = {'lower': torch.zeros(self.m), 'upper': torch.ones(self.m)}
            
        elif self.problem_id == 5:
            bounds['x_u'] = {'lower': torch.zeros(self.n), 'upper': torch.tensor([1.0, 1.0])}
            bounds['x_l'] = {'lower': torch.zeros(self.m), 'upper': torch.tensor([1.0, 1.0])}
            
        elif self.problem_id == 6:
            bounds['x_u'] = {'lower': torch.zeros(self.n), 'upper': torch.tensor([1.0])}
            bounds['x_l'] = {'lower': torch.zeros(self.m), 'upper': torch.ones(self.m)}
            
        elif self.problem_id == 7:
            bounds['x_u'] = {'lower': torch.zeros(self.n), 'upper': torch.tensor([100.0, 100.0])}
            bounds['x_l'] = {'lower': torch.zeros(self.m), 'upper': torch.tensor([100.0, 100.0])}
            
        elif self.problem_id == 8:
            bounds['x_u'] = {'lower': torch.zeros(self.n), 'upper': torch.tensor([50.0, 50.0])}
            bounds['x_l'] = {'lower': torch.tensor([-10.0, -10.0]), 'upper': torch.tensor([20.0, 20.0])}
            
        elif self.problem_id in [9, 10]:
            bounds['x_u'] = {'lower': -torch.pi * torch.ones(self.n), 'upper': torch.pi * torch.ones(self.n)}
            bounds['x_l'] = {'lower': -torch.pi * torch.ones(self.m), 'upper': torch.pi * torch.ones(self.m)}
            
        return bounds
    
    def upper_level_objective(self, x_u, x_l):
        """
        Compute the upper-level objective function F based on the problem_id.
        
        Args:
            x_u: Upper-level decision variables
            x_l: Lower-level decision variables
            
        Returns:
            Upper-level objective value
        """
        if self.problem_id == 1:
            return (x_u[0] - 30)**2 + (x_u[1] - 20)**2 - 20*x_l[0] + 20*x_l[1]
            
        elif self.problem_id == 2:
            return 2*x_u[0] + 2*x_u[1] - 3*x_l[0] - 3*x_l[1] - 60
            
        elif self.problem_id == 3:
            return -(x_u[0])**2 - 3*(x_u[1])**2 - 4*x_l[0] + (x_l[1])**2
            
        elif self.problem_id == 4:
            return -8*x_u[0] - 4*x_u[1] + 4*x_l[0] - 40*x_l[1] - 4*x_l[2]
            
        elif self.problem_id == 5:
            #b = torch.tensor([[1.0, 3.0], [3.0, 10.0]])
            #c = torch.tensor([[-1.0, 2.0], [-1.0/3.0, -3.0]]) 
            #c = torch.tensor([[-1.0, 2.0], [3.0, -3.0]])  ## PAPER FORMULA

            r = 0.1
            x = torch.tensor(x_u)
            y = torch.tensor(x_l)
            return r * torch.matmul(x.T, x) - 3*y[0] -4*y[1] + 0.5 * torch.matmul(y.T, y)
            #return torch.matmul(torch.matmul(x_u.T, b), x_u) - 3*x_l[0] - 4*x_l[1] + 0.5*torch.matmul(torch.matmul(x_l.T, c), x_l)
            
        elif self.problem_id == 6:
            return (x_u[0] - 1)**2 + 2*x_l[0] - 2*x_u[0]
            #return (x_u[0] - 1)**2 + 2*x_l[0] - 2*x_l[1]
            
        elif self.problem_id == 7:
            numerator = (x_u[0] + x_l[0]) * (x_u[1] + x_l[1])
            #denominator = 1 + x_u[0] + x_u[1] + x_l[0] + x_l[1]
            denominator = 1 + (x_u[0]*x_l[0]) + (x_u[1] * x_l[1])
            return -numerator / denominator
            
        elif self.problem_id == 8:
            #return 2*x_u[0] + 2*x_u[1] - 3*x_l[0] - 3*x_l[1] - 60
            return torch.abs(torch.tensor(2*x_u[0] + 2*x_u[1] - 3*x_l[0] - 3*x_l[1] - 60))
            
        elif self.problem_id == 9:
            x = torch.tensor(x_u)
            y = torch.tensor(x_l)
            return torch.sum(torch.abs(x - 1) + torch.abs(y))
            
        elif self.problem_id == 10:
            return torch.sum(torch.abs(x_u - 1) + torch.abs(x_l))
    
    def lower_level_objective(self, x_u, x_l):
        """
        Compute the lower-level objective function f based on the problem_id.
        
        Args:
            x_u: Upper-level decision variables
            x_l: Lower-level decision variables
            
        Returns:
            Lower-level objective value
        """
        if self.problem_id == 1:
            return (x_u[0] - x_l[0])**2 + (x_u[1] - x_l[1])**2
            
        elif self.problem_id == 2:
            return (x_l[0] - x_u[0] + 20)**2 + (x_l[1] - x_u[1] + 20)**2
            
        elif self.problem_id == 3:
            return 2*(x_u[0])**2 + (x_l[0])**2 - 5*x_l[1]
            
        elif self.problem_id == 4:
            #return x_u[0] + 2*x_u[1] + x_l[0] + x_l[1] + x_l[2] + 2*x_l[0]
            return x_u[0] + 2*x_u[1] + x_l[0] + x_l[1] + 2*x_l[2]
        
        elif self.problem_id == 5:
            b = torch.tensor([[1.0, 3.0], [3.0, 10.0]])
            #c = torch.tensor([[-1.0, 2.0], [-1.0/3.0, -3.0]])
            c = torch.tensor([[-1.0, 2.0], [3.0, -3.0]])
            r = 0.1
            x = torch.tensor(x_u)
            y = torch.tensor(x_l)
            b_x = torch.matmul(c, x)
            return 0.5 * torch.matmul(torch.matmul(y.T, b), y) - torch.matmul(b_x.T, y)
            #return 0.5*torch.matmul(torch.matmul(x_u.T, b), x_l) - 0.5*torch.matmul(torch.matmul(x_l.T, c), x_u)
            
        elif self.problem_id == 6:
            return (2*x_l[0] - 4)**2 + (2*x_l[1] - 1)**2 + x_u[0]*x_l[0]
            
        elif self.problem_id == 7:
            numerator = (x_u[0] + x_l[0]) * (x_u[1] + x_l[1])
            #denominator = 1 + x_u[0] + x_u[1] + x_l[0] + x_l[1]
            denominator = 1 + (x_u[0] * x_l[0]) + (x_u[1] * x_l[1])
            return numerator / denominator

        elif self.problem_id == 8:
            return (x_l[0] - x_u[0] + 20)**2 + (x_l[1] - x_u[1] + 20)**2
            
        elif self.problem_id == 9:
            y = torch.tensor(x_l)
            x = torch.tensor(x_u)
            sum_y_term = torch.sum(y**2)
            sum_x_term = torch.sum(x**2)
            #prod_term = torch.prod(torch.cos(4*x_l))
            prod_term = torch.prod(torch.cos(y)/torch.sqrt(torch.tensor([i for i in range(1, len(y)+1)])))
            #return torch.exp(1 + (1/4000)*sum_y_term - prod_term)
            return torch.exp((1 + (1/4000)*sum_y_term - prod_term) * sum_x_term)

            
        elif self.problem_id == 10:
            #sum_term = torch.sum(x_l**2)
            sum_term = torch.sum((x_l*x_u)**2)
            #prod_term = torch.prod(torch.cos(4*x_l))
            prod_term = torch.prod(torch.cos(x_l*x_u)/torch.sqrt(torch.tensor([i for i in range(1, len(x_l)+1)])))
            
            return torch.exp(1 + (1/4000)*sum_term - prod_term)
    
    def lower_level_constraint_violations(self, x_u, x_l):
        """
        Check if the lower-level constraints are violated for the given problem.
        
        Args:
            x_u: Upper-level decision variables
            x_l: Lower-level decision variables
            
        Returns:
            Number of lower-level constraint violations
        """
        if self.problem_id == 1:
            c4 = x_l[0] >= 0
            c5 = x_l[1] >= 0
            c6 = x_l[0] <= 10
            c7 = x_l[1] <= 10
            return np.invert(np.array([c4, c5, c6, c7])).sum()
            
        elif self.problem_id == 2:
            c1 = x_u[0] - 2*x_l[0] >= 10
            c2 = x_u[1] - 2*x_l[1] >= 10
            #c2 = x_l[0] - 2*x_l[1] >= 10
            c3 = x_l[0] <= -10 or x_l[0] >= 20
            c4 = x_l[1] <= -10 or x_l[1] >= 20
            #c4 = x_l[0] >= -10
            #c5 = x_l[1] >= -10
            #c6 = x_l[0] <= 50
            #c7 = x_l[1] <= 50
            return np.invert(np.array([c1, c2, c3, c4])).sum()
            
        elif self.problem_id == 3:
            c2 = (x_u[0])**2 - 2*x_u[0] + x_u[1]**2 -2*x_l[0] + x_l[1] >= -3
            #c2 = (x_u[1])**2 - 2*x_u[0] + x_l[0] + x_l[1] >= -3
            c3 = x_u[1] + 3*x_l[0] - 4*x_l[1] >= 4
            c4 = x_l[0] >= 0
            c5 = x_l[1] >= 0
            #c6 = x_l[0] <= 15
            #c7 = x_l[1] <= 15
            return np.invert(np.array([c2, c3, c4, c5])).sum()
            
        elif self.problem_id == 4:
            c1 = x_l[1] + x_l[2] - x_l[0] <= 1
            #c1 = x_l[0] + x_l[1] + x_l[2] <= 1
            c2 = 2*x_u[0] - x_l[0] + 2*x_l[1] - 0.5*x_l[2] <= 1
            c3 = 2*x_u[1] + 2*x_l[0] - x_l[1] - 0.5*x_l[2] <= 1
            c4 = x_l[0] >= 0
            c5 = x_l[1] >= 0
            c6 = x_l[2] >= 0
            #c7 = x_l[0] <= 1
            #c8 = x_l[1] <= 1
            #c9 = x_l[2] <= 1
            return np.invert(np.array([c1, c2, c3, c4, c5, c6])).sum()
            
        elif self.problem_id == 5:
            c1 = -0.333*x_l[0] + x_l[1] - 2 <= 0
            c2 = x_l[0] - 0.333*x_l[1] - 2 <= 0
            c3 = x_l[0] >= 0
            c4 = x_l[1] >= 0
            #c5 = x_l[0] <= 1
            #c6 = x_l[1] <= 1
            return np.invert(np.array([c1, c2, c3, c4])).sum()
            
        elif self.problem_id == 6:
            c1 = 4*x_u[0] + 5*x_l[0] + 4*x_l[1] <= 12
            c2 = 4*x_l[1] - 4*x_u[0] - 5 * x_l[0] <= -4
            c3 = 4*x_u[0] - 4*x_l[0] + 5 * x_l[1] <= 4
            c4 = 4*x_l[0] - 4*x_u[0] + 5 * x_l[1] <= 4
            #c1 = 4*x_l[0] + 5*x_l[1] + 4*x_l[0] <= 12
            #c2 = 4*x_l[0] - 4*x_l[1] + 5*x_l[1] >= -4
            #c3 = 4*x_l[0] - 4*x_l[1] + 5*x_l[0] <= 4
            #c4 = 4*x_l[0] - 4*x_l[1] + 5*x_l[1] <= 4
            c5 = x_l[0] >= 0
            c6 = x_l[1] >= 0
            #c7 = x_l[0] <= 1
            #c8 = x_l[1] <= 1
            return np.invert(np.array([c1, c2, c3, c4, c5, c6])).sum()
            
        elif self.problem_id == 7:
            c3 = x_l[0] >= 0
            c4 = x_l[1] >= 0
            c5 = x_l[0] <= x_u[0]
            c6 = x_l[1] <= x_u[1]
            return np.invert(np.array([c3, c4, c5, c6])).sum()
            
        elif self.problem_id == 8:
            c2 = 2*x_l[0] - x_u[0] + 10 <= 0
            c3 = 2*x_l[1] - x_u[1] + 10 <= 0
            c4 = x_l[0] >= -10
            c5 = x_l[1] >= -10
            c6 = x_l[0] <= 20
            c7 = x_l[1] <= 20
            return np.invert(np.array([c2, c3, c4, c5, c6, c7])).sum()
            
        elif self.problem_id in [9, 10]:
            y = torch.tensor(x_l)
            c3 = torch.all(y >= -torch.pi)
            c4 = torch.all(y <= torch.pi)
            return np.invert(np.array([c3, c4])).sum()
            
        return 0

    def upper_level_constraint_violations(self, x_u, x_l):
        """
        Check if the upper-level constraints are violated for the given problem.
        
        Args:
            x_u: Upper-level decision variables
            x_l: Lower-level decision variables
            
        Returns:
            Number of upper-level constraint violations
        """
        if self.problem_id == 1:
            c1 = x_u[0] + 2*x_u[1] >= 30
            c2 = x_u[0] + x_u[1] <= 25
            c3 = x_u[1] <= 15
            return np.invert(np.array([c1, c2, c3])).sum()
            
        elif self.problem_id == 2:
            c1 = x_u[0] + x_u[1] + x_l[0] - 2*x_l[1] <= 40
            return np.invert(np.array([c1])).sum()
            
        elif self.problem_id == 3:
            c1 = (x_u[0])**2 + 2*x_u[1] <= 4
            return np.invert(np.array([c1])).sum()
            
        elif self.problem_id == 7:
            c1 = (x_u[0])**2 + (x_u[1])**2 <= 100
            c2 = x_u[0] - x_u[1] <= 0
            return np.invert(np.array([c1, c2])).sum()
            
        elif self.problem_id == 8:
            c1 = x_u[0] + x_u[1] + x_l[0] - 2*x_l[1] <= 40
            return np.invert(np.array([c1])).sum()
            
        #elif self.problem_id in [9, 10]:
        #    c1 = torch.all(x_u >= -torch.pi)
        #    c2 = torch.all(x_u <= torch.pi)
        #    return np.invert(np.array([c1, c2])).sum()
            
        return 0
    
    def initialize_variables(self, device='cpu'):
        """
        Initialize variables within their bounds.
        
        Args:
            device: PyTorch device to use
            
        Returns:
            Initialized variables as PyTorch tensors
        """
        # Initialize variables randomly within bounds
        x_u = torch.rand(self.n, device=device) * (self.bounds['x_u']['upper'] - self.bounds['x_u']['lower']) + self.bounds['x_u']['lower']
        x_l = torch.rand(self.m, device=device) * (self.bounds['x_l']['upper'] - self.bounds['x_l']['lower']) + self.bounds['x_l']['lower']
        
        # Make variables require gradients for optimization
        x_u.requires_grad_(True)
        x_l.requires_grad_(True)
        
        return x_u, x_l


class GeneticAlgorithmSolver:
    def __init__(self, benchmark, population_size=50, generations=100, mutation_rate=0.1, crossover_rate=0.8, device='cpu'):
        """
        Initialize the genetic algorithm solver.

        Args:
            benchmark: An instance of TPBenchmark.
            population_size: Number of individuals in the population.
            generations: Number of generations to evolve.
            mutation_rate: Probability of mutation.
            crossover_rate: Probability of crossover.
            device: PyTorch device to use.
        """
        self.benchmark = benchmark
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.device = device

    def initialize_population(self):
        """Initialize the population with random individuals."""
        population = []
        for _ in range(self.population_size):
            x_u, x_l = self.benchmark.initialize_variables(device=self.device)
            population.append((x_u, x_l))
        return population

    def evaluate_fitness(self, x_u, x_l):
        """Evaluate the fitness of an individual."""
        upper_obj = self.benchmark.upper_level_objective(x_u, x_l).item()
        lower_obj = self.benchmark.lower_level_objective(x_u, x_l).item()
        upper_violations = self.benchmark.upper_level_constraint_violations(x_u, x_l)
        lower_violations = self.benchmark.lower_level_constraint_violations(x_u, x_l)
        penalty = upper_violations + lower_violations
        return upper_obj + lower_obj + penalty * 1e6  # Penalize constraint violations heavily

    def select_parents(self, population, fitness):
        """Select parents using tournament selection."""
        selected = []
        for _ in range(2):
            candidates = np.random.choice(len(population), size=3, replace=False)
            best_candidate = min(candidates, key=lambda idx: fitness[idx])
            selected.append(population[best_candidate])
        return selected

    def crossover(self, parent1, parent2):
        """Perform crossover between two parents."""
        if np.random.rand() < self.crossover_rate:
            alpha = torch.rand_like(parent1[0])
            child1_x_u = alpha * parent1[0] + (1 - alpha) * parent2[0]
            child1_x_l = alpha * parent1[1] + (1 - alpha) * parent2[1]
            child2_x_u = (1 - alpha) * parent1[0] + alpha * parent2[0]
            child2_x_l = (1 - alpha) * parent1[1] + alpha * parent2[1]
            return (child1_x_u, child1_x_l), (child2_x_u, child2_x_l)
        return parent1, parent2

    def mutate(self, individual):
        """Mutate an individual."""
        if np.random.rand() < self.mutation_rate:
            noise_u = torch.randn_like(individual[0]) * 0.1
            noise_l = torch.randn_like(individual[1]) * 0.1
            individual_0 = torch.clamp(individual[0] + noise_u, self.benchmark.bounds['x_u']['lower'], self.benchmark.bounds['x_u']['upper'])
            individual_1 = torch.clamp(individual[1] + noise_l, self.benchmark.bounds['x_l']['lower'], self.benchmark.bounds['x_l']['upper'])
            return (individual_0, individual_1)
        return individual

    def evolve(self):
        """Run the genetic algorithm."""
        population = self.initialize_population()
        for generation in range(self.generations):
            fitness = [self.evaluate_fitness(x_u, x_l) for x_u, x_l in population]
            new_population = []
            for _ in range(self.population_size // 2):
                parent1, parent2 = self.select_parents(population, fitness)
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                new_population.extend([child1, child2])
            population = new_population

            # Print best fitness in the current generation
            best_fitness = min(fitness)
            print(f"Generation {generation + 1}/{self.generations}, Best Fitness: {best_fitness}")

        # Return the best solution
        best_idx = np.argmin([self.evaluate_fitness(x_u, x_l) for x_u, x_l in population])
        return population[best_idx]


# Solve TPBenchmark problem using GeneticAlgorithmSolver
# if __name__ == "__main__":
#     # Choose problem ID (1-10)
#     problem_id = 1

#     # Create benchmark problem
#     benchmark = TPBenchmark(problem_id=problem_id)

#     # Create genetic algorithm solver
#     ga_solver = GeneticAlgorithmSolver(benchmark, population_size=50, generations=200, mutation_rate=0.1, crossover_rate=0.8, device='cpu')

#     # Run the genetic algorithm
#     best_solution = ga_solver.evolve()

#     # Extract the best solution
#     best_x_u, best_x_l = best_solution
#     best_upper_obj = benchmark.upper_level_objective(best_x_u, best_x_l).item()
#     best_lower_obj = benchmark.lower_level_objective(best_x_u, best_x_l).item()

#     print(f"Best upper-level decision variables: {best_x_u}")
#     print(f"Best lower-level decision variables: {best_x_l}")
#     print(f"Best upper-level objective value: {best_upper_obj}")
#     print(f"Best lower-level objective value: {best_lower_obj}")

# Example usage
if __name__ == "__main__":
   # Choose problem ID (1-10)
   problem_id = 9
   
   # Create benchmark problem
   benchmark = TPBenchmark(problem_id=problem_id)
   
   # Initialize variables
   x_u, x_l = benchmark.initialize_variables()
   
   # Compute objective values
   upper_obj = benchmark.upper_level_objective(x_u, x_l)
   lower_obj = benchmark.lower_level_objective(x_u, x_l)
   
   # Check constraints
   #constraints_satisfied = benchmark.check_constraints(x_u, x_l)
   
   print(f"Problem TP{problem_id} initialized with dimensions: n={benchmark.n}, m={benchmark.m}")
   print(f"Initial upper-level objective: {upper_obj.item()}")
   print(f"Initial lower-level objective: {lower_obj.item()}")
   #print(f"Constraints satisfied: {constraints_satisfied}")
   print(f"Variable bounds: {benchmark.bounds}") 