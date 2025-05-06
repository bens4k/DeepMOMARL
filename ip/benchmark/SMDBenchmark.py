import torch
import numpy as np

class SMDBenchmark:
    def __init__(self, problem_id, p=10, q=10, r=10, s=10):
        """
        Initialize the SMD benchmark problem.
        
        Args:
            problem_id: Integer from 1 to 6 indicating which problem to use
            p: Dimension of x_u1
            q: Dimension of x_l1
            r: Dimension of x_u2/x_l2
            s: Additional dimension parameter for problem 6
        """
        self.problem_id = problem_id
        self.p = p
        self.q = q
        self.r = r
        self.s = s
        
        # Set variable bounds based on the problem
        self.bounds = self._get_bounds()
        
    def _get_bounds(self):
        """Define variable bounds for each problem"""
        bounds = {}
        
        # Set bounds based on the problem ID
        if self.problem_id == 1:
            bounds['x_u1'] = {'lower': -5.0, 'upper': 10.0}
            bounds['x_u2'] = {'lower': -5.0, 'upper': 10.0}
            bounds['x_l1'] = {'lower': -5.0, 'upper': 10.0}
            bounds['x_l2'] = {'lower': -np.pi/2, 'upper': np.pi/2}
            
        elif self.problem_id == 2:
            bounds['x_u1'] = {'lower': -5.0, 'upper': 10.0}
            bounds['x_u2'] = {'lower': -5.0, 'upper': 1.0}
            bounds['x_l1'] = {'lower': -5.0, 'upper': 10.0}
            bounds['x_l2'] = {'lower': 0.0, 'upper': np.e}
            
        elif self.problem_id == 3:
            bounds['x_u1'] = {'lower': -5.0, 'upper': 10.0}
            bounds['x_u2'] = {'lower': -5.0, 'upper': 10.0}
            bounds['x_l1'] = {'lower': -5.0, 'upper': 10.0}
            bounds['x_l2'] = {'lower': -np.pi/2, 'upper': np.pi/2}
            
        elif self.problem_id == 4:
            bounds['x_u1'] = {'lower': -5.0, 'upper': 10.0}
            bounds['x_u2'] = {'lower': -1.0, 'upper': 1.0}
            bounds['x_l1'] = {'lower': -5.0, 'upper': 10.0}
            bounds['x_l2'] = {'lower': 0.0, 'upper': np.e}
            
        elif self.problem_id == 5:
            bounds['x_u1'] = {'lower': -5.0, 'upper': 10.0}
            bounds['x_u2'] = {'lower': -5.0, 'upper': 10.0}
            bounds['x_l1'] = {'lower': -5.0, 'upper': 10.0}
            bounds['x_l2'] = {'lower': -5.0, 'upper': 10.0}
            
        elif self.problem_id == 6:
            bounds['x_u1'] = {'lower': -5.0, 'upper': 10.0}
            bounds['x_u2'] = {'lower': -5.0, 'upper': 10.0}
            bounds['x_l1'] = {'lower': -5.0, 'upper': 10.0}
            bounds['x_l2'] = {'lower': -5.0, 'upper': 10.0}
        
        return bounds
    
    def upper_level_objective(self, x_u, x_l):
        """
        Compute the upper-level objective function F based on the problem_id.
        
        Args:
            x_u1: Upper-level decision variables (first group)
            x_u2: Upper-level decision variables (second group)
            x_l1: Lower-level decision variables (first group)
            x_l2: Lower-level decision variables (second group)
            
        Returns:
            Upper-level objective value
        """
        x_u1, x_u2 = x_u
        x_l1, x_l2 = x_l
        
        x_u1 = torch.tensor(x_u1)
        x_u2 = torch.tensor(x_u2)
        x_l1 = torch.tensor(x_l1)
        x_l2 = torch.tensor(x_l2)

        if self.problem_id == 1:
            F1 = torch.sum(x_u1**2)
            F2 = torch.sum(x_l1**2)
            F3 = torch.sum(x_u2**2) + torch.sum((x_l2 - torch.tan(x_l2))**2)
            return F1 + F2 + F3
            
        elif self.problem_id == 2:
            F1 = torch.sum(x_u1**2)
            F2 = torch.sum(x_l1**2)
            F3 = torch.sum(x_u2**2) - torch.sum((x_l2 - torch.log(x_l2))**2)
            return F1 + F2 + F3
            
        elif self.problem_id == 3:
            F1 = torch.sum(x_u1**2)
            F2 = torch.sum(x_l1**2)
            F3 = torch.sum(x_u2**2) + torch.sum(((x_u2)**2 - torch.tan(x_l2))**2)
            return F1 + F2 + F3
            
        elif self.problem_id == 4:
            F1 = torch.sum(x_u1**2)
            F2 = torch.sum(x_l1**2)
            F3 = torch.sum(x_u2**2) - torch.sum((torch.abs(x_l2) - torch.log(1 + x_l2))**2)
            return F1 + F2 + F3
            
        elif self.problem_id == 5:
            F1 = torch.sum(x_u1**2)
            F2 = -torch.sum(((x_l1 - x_l1)**2) + (x_l1 - 1)**2)
            F3 = torch.sum(x_u2**2) - torch.sum((torch.abs(x_l2) - (x_l2)**2)**2)
            return F1 + F2 + F3
            
        elif self.problem_id == 6:
            F1 = torch.sum(x_u1**2)
            F2 = torch.sum(x_l1**2) + torch.sum(x_l1**2)
            F3 = torch.sum(x_u2**2) - torch.sum((x_l2 - x_l2**2)**2)
            return F1 + F2 + F3
    
    def lower_level_objective(self, x_u, x_l):
        """
        Compute the lower-level objective function f based on the problem_id.
        
        Args:
            x_u1: Upper-level decision variables (first group)
            x_u2: Upper-level decision variables (second group)
            x_l1: Lower-level decision variables (first group)
            x_l2: Lower-level decision variables (second group)
            
        Returns:
            Lower-level objective value
        """
        x_u1, x_u2 = x_u
        x_l1, x_l2 = x_l

        x_u1 = torch.tensor(x_u1)
        x_u2 = torch.tensor(x_u2)
        x_l1 = torch.tensor(x_l1)
        x_l2 = torch.tensor(x_l2)

        if self.problem_id == 1:
            f1 = torch.sum(x_u1**2)
            f2 = torch.sum(x_l1**2)
            f3 = torch.sum((x_l2 - torch.tan(x_l2))**2)
            return f1 + f2 + f3
            
        elif self.problem_id == 2:
            f1 = torch.sum(x_u1**2)
            f2 = torch.sum(x_l1**2)
            f3 = torch.sum((x_l2 - torch.log(x_l2))**2)
            return f1 + f2 + f3
            
        elif self.problem_id == 3:
            q = self.q  # Using q as a parameter in the formula
            f1 = torch.sum(x_u1**2)
            f2 = q + torch.sum((x_l1**2 - torch.cos(2*torch.pi*x_l1)))
            f3 = torch.sum(((x_u2)**2 - torch.tan(x_l2))**2)
            return f1 + f2 + f3
            
        elif self.problem_id == 4:
            q = self.q  # Using q as a parameter in the formula
            f1 = torch.sum(x_u1**2)
            f2 = q + torch.sum((x_l1**2 - torch.cos(2*torch.pi*x_l1)))
            f3 = torch.sum((torch.abs(x_l2) - torch.log(1 + x_l2))**2)
            return f1 + f2 + f3
            
        elif self.problem_id == 5:
            f1 = torch.sum(x_u1**2)
            f2 = torch.sum(((x_l1 - x_l1)**2) + (x_l1 - 1)**2)
            f3 = torch.sum((torch.abs(x_l2) - (x_l2)**2)**2)
            return f1 + f2 + f3
            
        elif self.problem_id == 6:
            q = self.q
            s = self.s
            f1 = torch.sum(x_u1**2)
            f2 = torch.sum(x_l1**2) + torch.sum((x_l1 - x_l1)**2)
            f3 = torch.sum((x_l2 - x_l2**2)**2)
            return f1 + f2 + f3
    
    def initialize_variables(self, device='cpu'):
        """
        Initialize variables within their bounds.
        
        Args:
            device: PyTorch device to use
            
        Returns:
            Initialized variables as PyTorch tensors
        """
        # Initialize variables randomly within bounds
        x_u1 = torch.rand(self.p, device=device) * (self.bounds['x_u1']['upper'] - self.bounds['x_u1']['lower']) + self.bounds['x_u1']['lower']
        x_u2 = torch.rand(self.r, device=device) * (self.bounds['x_u2']['upper'] - self.bounds['x_u2']['lower']) + self.bounds['x_u2']['lower']
        
        x_l1_size = self.q if self.problem_id != 6 else self.q + self.s
        x_l1 = torch.rand(x_l1_size, device=device) * (self.bounds['x_l1']['upper'] - self.bounds['x_l1']['lower']) + self.bounds['x_l1']['lower']
        x_l2 = torch.rand(self.r, device=device) * (self.bounds['x_l2']['upper'] - self.bounds['x_l2']['lower']) + self.bounds['x_l2']['lower']
        
        # Make variables require gradients for optimization
        x_u1.requires_grad_(True)
        x_u2.requires_grad_(True)
        x_l1.requires_grad_(True)
        x_l2.requires_grad_(True)
        
        return [x_u1, x_u2], [x_l1, x_l2]


## Example usage
if __name__ == "__main__":
   # Choose problem ID (1-6)
   problem_id = 1
   
   # Set dimensions
   p = 5  # Dimension of x_u1
   q = 5  # Dimension of x_l1
   r = 3  # Dimension of x_u2/x_l2
   s = 2  # Additional dimension for problem 6
   
   # Create benchmark problem
   benchmark = SMDBenchmark(problem_id=problem_id, p=p, q=q, r=r, s=s)
   
   # Initialize variables
   x_u, x_l = benchmark.initialize_variables()
   
   # Compute objective values
   #upper_obj = benchmark.upper_level_objective(x_u, x_l)
   #lower_obj = benchmark.lower_level_objective(x_u, x_l)
   
   print(f"Problem {problem_id} initialized with dimensions: p={p}, q={q}, r={r}, s={s}")
   #print(f"Initial upper-level objective: {upper_obj.item()}")
   #print(f"Initial lower-level objective: {lower_obj.item()}")
   print(f"Variable bounds: {benchmark.bounds}")