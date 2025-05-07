import torch
import torch.nn as nn
import torch.autograd as autograd

from .base import   SpatialSampler,\
                    DatasetGeneratorBase,\
                    NormalizerBase,\
                    DataLoaderBase,\
                    TrainerBase,\
                    general_call,\
                    scatter_error2d,\
                    to_device,\
                    set_seed,\
                    EquationLookUp
from config import EQUATION_KEY

class PINNDatasetGenerator(DatasetGeneratorBase):
    def __init__(self, T, Equation, seed=1234, **kwargs):
        self.T = T
        self.kwargs = kwargs
        self.Equation = Equation
        self.seed = seed
        set_seed(seed)
        
    def __call__(self, n_sample, n_points, sampler="mesh"):
        """
        Generate dataset for PINN
        
        Parameters:
        - n_sample: number of equation instances (parameter sets)
        - n_points: number of spatial points for each instance
        - sampler: spatial sampling method
        
        Returns:
        - inputs: [n_samples * n_points, 3+d] (t, x, y, mu)
        - outputs: [n_samples * n_points, 1] u(t, x, y, mu)
        """
        inputs = []
        outputs = []
        u0s = []
        points_sampler = SpatialSampler(self.Equation, sampler=sampler)
        
        x_dim = self.Equation.x_domain.shape[0]
        
        for _ in range(n_sample):
            points = points_sampler(n_points, flatten=True)
            equation = self.Equation(**self.kwargs)
            
            # Create random time points (0, T)
            time_points = torch.rand(points.shape[0], 1) * self.T
            
            # Input includes time, spatial coordinates, and equation parameters
            input_data = torch.cat([
                time_points, 
                points, 
                equation.variable.flatten()[None,:].tile(points.shape[0], 1)
            ], -1)  # [n_points, 1+2+d]
            
            # Output is the solution at given time and points
            output = equation(
                time_points.squeeze(), 
                *[points[:,i] for i in range(x_dim)]
            )[:, None]  # [n_points, 1]
            
            # Also record initial condition
            u0 = equation(0, *[points[:,i] for i in range(x_dim)])  # [n_points]
            
            u0s.append(u0)
            inputs.append(input_data)
            outputs.append(output)
     
        self.u0s = torch.stack(u0s, dim=0)  # [n_samples, n_points]
        inputs = torch.cat(inputs, dim=0)   # [n_samples * n_points, 1+2+d]
        outputs = torch.cat(outputs, dim=0) # [n_samples * n_points, 1]
       
        return inputs, outputs

class PINNNormalizer(NormalizerBase):
    def __init__(self, inputs_min, inputs_max, outputs_min, outputs_max):
        self.inputs_min = inputs_min
        self.inputs_max = inputs_max
        self.outputs_min = outputs_min
        self.outputs_max = outputs_max

    @classmethod
    def init(cls, inputs, outputs):
        inputs_min = inputs.min(0).values[None, ...]
        inputs_max = inputs.max(0).values[None, ...]
        outputs_min = outputs.min(0).values[None, ...]
        outputs_max = outputs.max(0).values[None, ...]
        return cls(inputs_min, inputs_max, outputs_min, outputs_max)

    def __call__(self, inputs, outputs):
        inputs_min, inputs_max, outputs_min, outputs_max = to_device(
            [self.inputs_min, self.inputs_max, self.outputs_min, self.outputs_max],
            inputs.device
        )
        # normalize input
        inputs = (inputs-inputs_min)/(inputs_max-inputs_min)
        # normalize output
        outputs = (outputs-outputs_min)/(outputs_max-outputs_min)
        return inputs, outputs
    
    def norm_input(self, input):
        inputs_min, inputs_max = to_device([self.inputs_min, self.inputs_max], input.device)
        return (input - inputs_min) / (inputs_max-inputs_min)
    
    def unorm_output(self, output):
        outputs_min, outputs_max = to_device([self.outputs_min, self.outputs_max], output.device)
        return output*(outputs_max-outputs_min)+outputs_min
    
    def save(self, path):
        torch.save({
            'inputs_min': self.inputs_min,
            'inputs_max': self.inputs_max,
            'outputs_min': self.outputs_min,
            'outputs_max': self.outputs_max
        }, path)
    
    @classmethod
    def load(cls, path):
        data = torch.load(path)
        return cls(data['inputs_min'], data['inputs_max'], data['outputs_min'], data['outputs_max'])

class PINNDataLoader(DataLoaderBase):
    pass

class PINNTrainer(TrainerBase):
    """
    PINN Trainer implements Physics-Informed Neural Networks
    """
    DataLoader = PINNDataLoader
    Normalizer = PINNNormalizer
    
    def __init__(self, config):
        from models import PINN
        self.config = config
        Equation = EquationLookUp[config.equation]
        equation_kwargs = {EQUATION_KEY[config.equation]: config[EQUATION_KEY[config.equation]]}
        self.xlims = Equation.x_domain
        
        self.dataset_generator = PINNDatasetGenerator(
            config.T, Equation, seed=self.config.seed, **equation_kwargs
        )
        
        # Input size: time + spatial dimensions + equation parameters
        input_size = 1 + Equation.x_domain.shape[0] + Equation.degree_of_freedom(**equation_kwargs)
        
        self.model = PINN(
            input_size=input_size, 
            output_size=1, 
            hidden_size=config.num_hidden, 
            num_layers=config.num_layers, 
            activation=config.activation
        )
        
        self.equation = Equation
        self.equation_kwargs = equation_kwargs
        
        self.weight_path = f"weights/{config.equation}_{'_'.join([f'{k}={v}' for k,v in equation_kwargs.items()])}/pinn"
        self.image_path = f"images/{config.equation}_{'_'.join([f'{k}={v}' for k,v in equation_kwargs.items()])}/pinn"
    
    def compute_pde_loss(self, inputs, model):
        """
        Compute physics-informed loss based on PDE constraints
        """
        # Create inputs that require gradients
        inputs_tensor = inputs.clone().detach().requires_grad_(True)
        
        # Get normalized outputs from model
        u_pred = model(inputs_tensor)

        # Get dimensions first
        batch_size = inputs_tensor.shape[0]
        x_dim = self.equation.x_domain.shape[0]
            
        # Compute different loss terms based on the equation type
        equation_type = self.config.equation
        pde_loss = 0
        
        # Note: The original t and x slicings are removed as derivatives are now
        # calculated w.r.t. the whole inputs_tensor and then components are extracted.
        # t = inputs_tensor[:, 0:1] # No longer used this way
        # x = inputs_tensor[:, 1:1+x_dim] # No longer used this way

        if equation_type == "heat":
            # Calculate all first-order derivatives of u_pred w.r.t. inputs_tensor
            # inputs_tensor is a leaf and requires_grad=True
            all_first_order_grads = autograd.grad(
                u_pred, inputs_tensor,
                grad_outputs=torch.ones_like(u_pred),
                create_graph=True, # Must be true to compute higher-order derivatives
                retain_graph=True  # Keep graph for subsequent grad calls if any, or if u_pred is used elsewhere
            )[0]

            u_t = all_first_order_grads[:, 0:1]
            # u_x_components contains [du/dx_spatial_0, du/dx_spatial_1, ...]
            u_x_components = all_first_order_grads[:, 1:1+x_dim]

            u_xx_laplacian = 0
            for i in range(x_dim):
                # Current component of first spatial derivative (e.g., du/dx_i)
                du_dxi = u_x_components[:, i:i+1]
                
                # Compute its derivative w.r.t. the entire inputs_tensor
                # This is needed because du_dxi's graph depends on inputs_tensor
                grad_du_dxi_wrt_inputs = autograd.grad(
                    du_dxi, inputs_tensor,
                    grad_outputs=torch.ones_like(du_dxi),
                    create_graph=True,
                    retain_graph=True
                )[0]
                
                # Extract the specific second derivative d/dx_i (du/dx_i)
                # This corresponds to the (1+i)-th column of grad_du_dxi_wrt_inputs
                u_xxi = grad_du_dxi_wrt_inputs[:, 1+i:1+i+1]
                u_xx_laplacian += u_xxi
            
            # Heat equation: u_t = laplacian(u)
            pde_loss = ((u_t - u_xx_laplacian)**2).mean()
            
        elif equation_type == "wave":
            c = 0.1
            if hasattr(self, 'equation_kwargs') and 'c' in self.equation_kwargs:
                c = self.equation_kwargs['c']

            # Calculate all first-order derivatives of u_pred w.r.t. inputs_tensor
            all_first_order_grads = autograd.grad(
                u_pred, inputs_tensor,
                grad_outputs=torch.ones_like(u_pred),
                create_graph=True, retain_graph=True
            )[0]

            u_t_from_all = all_first_order_grads[:, 0:1]
            u_x_components = all_first_order_grads[:, 1:1+x_dim]

            # Calculate u_tt (second time derivative)
            grad_ut_wrt_inputs = autograd.grad(
                u_t_from_all, inputs_tensor,
                grad_outputs=torch.ones_like(u_t_from_all),
                create_graph=True,
                retain_graph=True
            )[0]
            u_tt = grad_ut_wrt_inputs[:, 0:1]

            # Calculate u_xx_laplacian (sum of second spatial derivatives)
            u_xx_laplacian = 0
            for i in range(x_dim):
                du_dxi = u_x_components[:, i:i+1]
                grad_du_dxi_wrt_inputs = autograd.grad(
                    du_dxi, inputs_tensor,
                    grad_outputs=torch.ones_like(du_dxi),
                    create_graph=True,
                    retain_graph=True
                )[0]
                u_xxi = grad_du_dxi_wrt_inputs[:, 1+i:1+i+1]
                u_xx_laplacian += u_xxi
            
            pde_loss = ((u_tt - (c**2) * u_xx_laplacian)**2).mean()
            
        elif equation_type == "poisson":
            # Poisson equation: -laplacian(u) = f (here simplified to -laplacian(u) = 0)
            all_first_order_grads = autograd.grad(
                u_pred, inputs_tensor,
                grad_outputs=torch.ones_like(u_pred),
                create_graph=True, retain_graph=True
            )[0]

            u_x_components = all_first_order_grads[:, 1:1+x_dim]

            u_xx_laplacian = 0
            for i in range(x_dim):
                du_dxi = u_x_components[:, i:i+1]
                grad_du_dxi_wrt_inputs = autograd.grad(
                    du_dxi, inputs_tensor,
                    grad_outputs=torch.ones_like(du_dxi),
                    create_graph=True,
                    retain_graph=True
                )[0]
                u_xxi = grad_du_dxi_wrt_inputs[:, 1+i:1+i+1]
                u_xx_laplacian += u_xxi
            
            pde_loss = ((u_xx_laplacian)**2).mean()
        
        return pde_loss
    
    def fit(self):
        """
        Train the PINN model with both data loss and physics loss
        """
        os.makedirs(self.image_path, exist_ok=True)
        os.makedirs(self.weight_path, exist_ok=True)

        config = self.config

        if config.pin_memory:
            kwargs = {"pin_memory": True}
        else:
            kwargs = {}
    
        train_dataset = self.dataset_generator(config.n_train_sample, config.n_train_spatial)
        valid_dataset = self.dataset_generator(config.n_valid_sample, config.n_valid_spatial)
        normalizer = self.Normalizer.init(*train_dataset)
        self.normalizer = normalizer
        train_dataset = normalizer(*train_dataset)
        valid_dataset = normalizer(*valid_dataset)
        train_dataloader = self.DataLoader(
            *train_dataset,
            batch_size=config.batch_size, 
            shuffle=True,
            device=config.device,
            **kwargs
        )
        valid_dataloader = self.DataLoader(
            *valid_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            device=config.device,
            **kwargs
        )
        
        model = self.model.to(config.device)
        losses = {"train": [], "valid": []}
        best_weight, best_loss, best_epoch = None, float('inf'), None
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        criterion = nn.MSELoss()
        
        p = tqdm(range(config.epoch))
    
        for ep in p:
            # training step
            model.train()
            
            iteration_losses = []
            for input_batch, output_batch in train_dataloader:
                input_batch, output_batch = to_device([input_batch, output_batch], config.device)

                optimizer.zero_grad()
        
                prediction = general_call(model, input_batch)
                
                # Data loss
                data_loss = criterion(prediction, output_batch)
                
                # Physics loss
                physics_loss = self.compute_pde_loss(input_batch, model)
                
                # Combined loss (with potential weighting)
                physics_weight = 0.1  # Can be added to config if needed
                loss = data_loss + physics_weight * physics_loss
                
                loss.backward()
                optimizer.step()

                iteration_losses.append(loss.item())

            iteration_loss = np.mean(iteration_losses)

            # record for display
            losses['train'].append((ep, iteration_loss))
            p.set_postfix({'loss': iteration_loss})

            if (ep+1) % config.eval_every_eps == 0:
                # validation every eval_every_eps epoch
                model.eval()

                with torch.no_grad():
                    iteration_losses = []
                    for input_batch, output_batch in valid_dataloader:
                        input_batch, output_batch = to_device([input_batch, output_batch], config.device)
                        
                        prediction = general_call(model, input_batch)
                        
                        # Only data loss for validation
                        iteration_losses.append(criterion(prediction, output_batch).item())

                    valid_loss = np.mean(iteration_losses)
                    losses['valid'].append((ep, valid_loss))

                    # save best valid loss weight
                    if valid_loss < best_loss:
                        best_weight = model.state_dict()
                        best_loss = valid_loss
                        best_epoch = ep

        # load the best recorded weight
        if best_weight is not None:
            model.load_state_dict(best_weight)

        model.eval()
        model = model.cpu()

        # plot loss
        self.plot_loss(losses['train'], losses['valid'], best_epoch, best_loss)

        self.model = model
    
    def eval(self):
        """
        Evaluate PINN model on test data
        
        Returns:
        - position: torch.Tensor, shape=(n_eval_sample, n_eval_spatial, 2)
        - predictions: torch.Tensor, shape=(n_eval_sample, n_eval_spatial)
        - outputs: torch.Tensor, shape=(n_eval_sample, n_eval_spatial)
        """
        self.to(self.config.device)
        config = self.config
        x_dim = self.xlims.shape[0]
        
        # Generate dataset at time T for evaluation
        inputs, outputs = self.dataset_generator(
            config.n_eval_sample, config.n_eval_spatial, sampler=config.sampler
        )
        
        # Extract spatial points for visualization
        # We evaluate at final time T for all points
        points = inputs[:, 1:1+x_dim].reshape(
            config.n_eval_sample, config.n_eval_spatial, x_dim
        )  # [n_eval_sample, n_eval_spatial, 2]
        
        # Create fixed-time evaluation points (all at time T)
        T_tensor = torch.ones((inputs.shape[0], 1)) * self.T
        eval_inputs = torch.cat([T_tensor, inputs[:, 1:]], dim=1)
        
        # Normalize inputs and outputs
        eval_inputs = self.normalizer.norm_input(eval_inputs)
        dataloader = self.DataLoader(
            eval_inputs, outputs, 
            batch_size=config.batch_size, 
            device=config.device, 
            shuffle=True
        )

        predictions = []
        outputs_list = []

        with torch.no_grad():
            for input_batch, output_batch in dataloader:  
                prediction = general_call(self.model, input_batch)  #[batch_size*n_eval_spatial, 1] 
                prediction = self.normalizer.unorm_output(prediction).reshape([-1, config.n_eval_spatial])
                output_batch = self.normalizer.unorm_output(output_batch).reshape([-1, config.n_eval_spatial])
                predictions.append(prediction.cpu())
                outputs_list.append(output_batch.cpu())

        predictions = torch.cat(predictions, dim=0)  # [n_eval_sample, n_eval_spatial]
        outputs = torch.cat(outputs_list, dim=0)     # [n_eval_sample, n_eval_spatial]

        return points, predictions, outputs

    def predict(self, n_eval_spatial):
        """
        Predict solution at time T using the trained model
        
        Parameters:
        - n_eval_spatial: number of spatial points for prediction
        
        Returns:
        - points: torch.Tensor, shape=(n_eval_spatial, 2) 
        - u0: torch.Tensor, shape=(n_eval_spatial)
        - prediction: torch.Tensor, shape=(n_eval_spatial)
        - uT: torch.Tensor, shape=(n_eval_spatial)
        """
        self.to(self.config.device)
        set_seed(self.config.seed)
        
        # Generate dataset at time T for prediction
        inputs, outputs = self.dataset_generator(1, n_eval_spatial, sampler="mesh")
        
        # Extract spatial points
        x_dim = self.xlims.shape[0]
        points = inputs[:, 1:1+x_dim]
        
        # Create fixed-time evaluation points (all at time T)
        T_tensor = torch.ones((inputs.shape[0], 1)) * self.T
        eval_inputs = torch.cat([T_tensor, inputs[:, 1:]], dim=1)
        
        # Normalize inputs for prediction
        eval_inputs = self.normalizer.norm_input(eval_inputs)
        eval_inputs = to_device(eval_inputs, self.config.device)
        
        with torch.no_grad():
            prediction = self.model(eval_inputs)
            
        prediction = self.normalizer.unorm_output(prediction).cpu()

        return points.cpu(), self.dataset_generator.u0s.flatten(), prediction.flatten(), outputs.flatten().cpu()

    def plot_prediction(self, n_eval_spatial):
        """
        Plot prediction vs ground truth
        """
        points, u0, prediction, uT = self.predict(n_eval_spatial)
        scatter_error2d(
            points[:,0], points[:,1], prediction, uT, 
            self.image_path, self.xlims, u0=u0
        )

# Import for numpy - for trainer functionality
import numpy as np
import os
from tqdm import tqdm