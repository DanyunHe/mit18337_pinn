#1D Burger's equation, Eqn (A.1.)
#t: 0 to 1; x: -1 to 1
#initial and boundary conditions

using CUDA, MAT, Flux, DelimitedFiles, ForwardDiff, BenchmarkTools

CUDA.allowscalar(true)
# ENV["JULIA_CUDA_VERBOSE"]=true

q=100 |> gpu; # number of stages

# define layers
layers=cu([1,50,50,50,q+1]) |> gpu;
lb=cu([-1.]) |> gpu;
ub=cu([1.]) |> gpu;

# read data
data=matread("Data/burgers_shock.mat");

t=cu(data["t"]) |> gpu; #length 100
x=cu(data["x"]) |> gpu;  #length 256
exact=cu(data["usol"]) |> gpu; #length 256x100, exact[:,1] is 256 data at time t[1]

#solution data at time n
idx_t0=10 |> gpu;
idx_t1=20 |> gpu;
data_tn_u=exact[:,idx_t0] |> gpu;
#the corresponding locations of the data
data_tn_x=x |> gpu;
#number of data point Ndata
Ndata=256 |> gpu;
#timestep size
dt=t[idx_t1].-t[idx_t0] |> gpu;
#boundary data
data_tn_x1 = reshape(hcat(lb, ub), 2) |> gpu;

# load IRK weights
temp=readdlm("IRK_weights/Butcher_IRK100.txt") |> gpu;
IRK_weights=cu(reshape(temp[1:q^2+q],(q+1,q))) |> gpu;
IRK_times=cu(temp[q^2+q:size(temp)[1]])|> gpu;

#Nueral Net of solutions at q stages and at time n+1, LHS of eqn 7
#input: x vector of length Ndata, output: solutions at locations x.
NN = Chain(Dense(layers[1],layers[2],tanh),
           Dense(layers[2],layers[3],tanh),
           Dense(layers[3],layers[4],tanh),
           Dense(layers[4],layers[5])) |> gpu;
# NN = fmap(cu, NN)

function NN_U1(x)
    N = size(x)[1]
    x_reshaped = reshape(x, 1, 1, N)
    U1_pred = NN(x_reshaped |> gpu)
    return reshape(U1_pred, q+1, N) # (q+1)*N
end |> gpu;

function NN_U0(x)
    N = size(x)[1]
    x_reshaped = reshape(x, 1, 1, N)
    x_nested = [[x[i]] for i in 1:N]
    nu = 0.01/pi
    U1 = reshape(NN(x_reshaped |> gpu), q+1, N) # (q+1)*N
    U = U1[1:q, :] # q*N
    # U_x_unshaped = ForwardDiff.jacobian(NN |> gpu, x_reshaped |> gpu) 
    # U_x = reshape(filter(!iszero |> gpu, U_x_unshaped |> gpu), q+1, N)[1:q, :] # q*N
    U_x_unshaped = ForwardDiff.jacobian.(NN_U1|>gpu , x_nested |> gpu)  
    U_x = reduce(vcat,transpose.(U_x_unshaped))
    # U_x = cu(reshape(collect(Iterators.flatten(U_x_unshaped)), (length(U_x_unshaped[1]),length(U_x_unshaped)))[1:q, :]); # q*N
    # U_xx = CUDA.zeros(q, N)
    # U_x = CUDA.zeros(q, N)
    # U_x  = ForwardDiff.jacobian(NN_U1 |> gpu, [x[1]] |> gpu)[1:q]
    # U_xx  = ForwardDiff.jacobian(y -> ForwardDiff.jacobian(NN_U1, y), [x[1]] )[1:q]
    # for i in 1:N
    #     # U_x = hcat(U_x , ForwardDiff.jacobian(NN_U1 |> gpu, [x[i]] |> gpu)[1:q])
    #     # U_xx = hcat(U_xx , ForwardDiff.jacobian(y -> ForwardDiff.jacobian(NN_U1, y), [x[i]] )[1:q])
    #     # U_x[:, i] = ForwardDiff.jacobian(NN_U1 |> gpu, [x[i]] |> gpu)[1:q]
    #     U_xx[:, i] = ForwardDiff.jacobian(y -> ForwardDiff.jacobian(NN_U1, y), [x[i]] )[1:q]# q*1
    # end
    hessian = ForwardDiff.jacobian.(y -> ForwardDiff.jacobian(NN_U1, y), x_nested);
    U_xx = reduce(vcat,transpose.(hessian))
    # U_xx = cu(reshape(collect(Iterators.flatten(hessian)), (length(hessian[1]),length(hessian)))[1:q, :]) # q*N

    F = -U.*U_x + nu*U_xx # q*N
    U0 = U1 - dt*(IRK_weights * F) #(q+1)*N
    return U0
end |> gpu;

#Eqn 8, Eqn A.9.
function loss()
        total_loss=0 
        #loop through N data points at time tn, and calculate and add up losses
        for i in 1:Ndata  #time n is t[idx_t0]
            #fill an array of length q+1 with value of exact soln at time n
            total_loss=total_loss+sum(abs2,transpose(NN_U0([x[i]] |> gpu)).-exact[i,idx_t0])
        end

        #add in boundary condition losses: u(x=-1)=0, u(x=1)=0
        total_loss = total_loss +sum(abs2,NN_U0([-1.]|> gpu))+sum(abs2,NN_U0([1.] |> gpu))
        return total_loss
end |> gpu;

function loss_tzy()
    nn_pred =  transpose(NN_U0(x |> gpu)) |> gpu;
    total_loss = sum((nn_pred .-exact[:,idx_t0]).^2)+ sum(NN_U0([-1.]|> gpu).^2)+sum(NN_U0([1.] |> gpu).^2)
    return total_loss
end |> gpu;

p=Flux.params(NN);

#train parameters in NN_U1 based on loss function, repeat the training iteration on the data points for iterN times
iterN=1000
benchmark_record = @benchmark Flux.train!(loss_tzy, p, Iterators.repeated((), iterN), ADAM())

# @epochs 2 Flux.train!(loss,p,Iterators.repeated((), iterN), ADAM())
open("time_records.txt", "a") do io
    println(io,benchmark_record ,"\n")
end;

#prediciton of solution at time n+1 at location x=[x0,x1,x2,x3...]
U1_star=cu(Array{Float64}(undef, Ndata))
for i in 1:Ndata
        U1_star[i]=NN_U1([x[i]])[q+1]
end

#Error calculation
using LinearAlgebra
error=norm(U1_star.-exact[:,idx_t1])/norm(exact[:,idx_t1])
open("error_records.txt", "a") do io
    println(io,error, "\n")
end;

#plot
using Plots
p = plot(x, [U1_star,exact[:,idx_t1]])
savefig(p, "prediction_plot")
