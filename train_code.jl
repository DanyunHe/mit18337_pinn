#1D Burger's equation, Eqn (A.1.)
#t: 0 to 1; x: -1 to 1
#initial and boundary conditions

# define layers 
q=500 # number of stages 
layers=[1,50,50,50,q+1]
lb=[-1.]
ub=[1.]

# pkg add MAT
# read data 
using MAT
data=matread("../Data/burgers_shock.mat")
t=data['tt']
x=data['x']
exact=data['uu']

#solution data at time n
idx_t0=10
idx_t1=90
data_tn_u=exact[idx_t0:idx_t0+1,:].T
#the corresponding locations of the data
data_tn_x=x
#number of data point Ndata
Ndata=250
#timestep size
dt=t[idx_t1]-t[idx_t0]
#boundary data
data_tn_x1=[lb,ub]

# load IRK weights 
# import Pkg
# Pkg.add("DelimitedFiles")
using DelimitedFiles
temp=readdlm("./IRK_weights/Butcher_IRK100.txt");
IRK_weights=reshape(temp[1:q^2+q],(q+1,q))
IRK_times=temp[q^2+q:size(temp)[1]]

#Nueral Net of solutions at q stages and at time n+1, LHS of eqn 7
#input: x vector of length Ndata, output: solutions at locations x.
NN = Chain(Dense(1,50,tanh),
           Dense(50,50,tanh),
           Dense(50,q+1))

function NN_U1(x)
    U1_pred = NN(x)  
    return U1_pred
end

function NN_U0(x)
    U1 = NN(x) # q*1
    U = U1[:, 1:q]
    U_x = ForwardDiff.gradient(NN, x)
    U_xx = ForwardDiff.hessian(NN, x)
    F = -U*U_x + nu*U_xx
    U0 = U1 - dt*(F* IRK_weights.T)
    return U0
end
#Eqn 8, Eqn A.9.
function loss()
        total_loss=0
        #loop through N data points at time tn, and calculate and add up losses
        for i in 1:N  #time n is t[idx_t0]
                #fill an array of length p+1 with value of exact soln at time n
                exact_tn_array=fill(exact[idx_t0,i], q+1)
                total_loss=total_loss+sum(abs2,NN_U0(x[i]).-exact_tn_array)
        end
        #add in boundary condition losses: u(x=-1)=0, u(x=1)=0
        total_loss = total_loss+sum(abs2,NN_U0(-1))+sum(abs2,NN_U0(1))
end

p=params(NN_U1)

#train parameters in NN_U1 based on loss function, repeat the training iteration on the data points for iterN times
iterN=1000
Flux.train!(loss,p,Iterators.repeated(data_tn, iterN), ADAM(0.1))


#prediciton of solution at time n+1 at location x=[x0,x1,x2,x3...]
U1_star=Array{Float64}(undef, N)
for i in 1:N
        U1_star[i]=NN_U1(x[i])[p+1]
end

#Error calculation
error=norm(U1_star.-exact[idx_t1,:])/norm(exact[idx_t1,:])
