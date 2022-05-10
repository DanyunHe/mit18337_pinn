# MIT 18.337 final project 

## Exploration of a discrete-time physices informed neural network

This project explores the physics-informed neural networks. In particular, we use the implicit Runge-Kutta time stepping schemes with unlimited number of stages to solve one-dimensional Burger’s Equation. The neural network is implemented in Julia and we vary different parameters including number of time-step size, number of stages in Runge-Kutta and different neural network structures and discuss their performance in final prediction. We also investigate different sampling strategies for more efficient training. 

- Section 4.1. Varying number of stage q and time-step sizes dt: `train_code_vary_stage_dt.jl`.

- Section 4.2. Varying number of samples: `train_code_sample_size.jl`.

- Section 4.3. Varying Neural network structure: `train_code_NN_tructure.jl`.

- Section 4.4. Importance sampling: `train_importance_sampling.jl`.

- Section 4.5. Exploration of multi-time-stepping procedure:
`train_code_multi_step_uniform_sampling_once.jl`, 
`train_code_multi_step_uniform_sampling_repeat.jl`, 
`train_code_multi_step_loss_weighted_sampling_repeat.jl`.
