#1D Burger's equation, Eqn (A.1.)
#t: 0 to 1; x: -1 to 1
#initial and boundary conditions

#solution data at time n
data_tn_u=
#the corresponding locations of the data
data_tn_x=
#number of data point Ndata
Ndata=
#timestep size
dt=


# define layers 
q=100
layers=[1,200,200,200,200,q+1]
lb=[-1.]
ub=[1.]

N=200

# pkg add MAT
# read data 
using MAT
data=matread("../Data/AC.nat")
t=data['tt']
x=data['x']
exact=data['uu']
idx_t0=20
idx_t1=180
dt=t[idx_t1]-t[idx_t0]

# inital data
idx_x=rand(1:N)

#solution data at time n
data_tn=t

# differential operator

Dx = Differential(x)
Dxx = Differential(x)^2

#Nueral Net of solutions at q stages and at time n+1, LHS of eqn 7
#input: x vector of length Ndata, output: solutions at locations x.
NN = Chain(Dense(1,200,tanh),
           Dense(200,200,tanh),
           Dense(200,q+1))

function NNU1(x)
    U1_pred = NN(x)  
    U1_x_pred = Dx(NN(x))
    return U1_pred, U1_x_pred
end

function NNU0(x)
    U1 = NN(x) 
    U = U1[:, :-1]
    U_x = Dx(U)
    U_xx = Dxx(U)
    F = 5.0*U - 5.0*U^3 + 0.0001*U_xx
    U0 = U1 - dt*(F* IRK_weights.T)
    return U0
end

# N_U0 =  #RHS of Eqn 9:RK scheme of NN_U1, and nonlinear operation from specific PDE (above Eqn.(A.9))

loss()= sum(NN_U0(data_tn_x)-data_tn_u)^2) # (NN_U0- data at un (Eqn 8)) (Eqn. A.9.)
        + sum(NN_U0(-1)^2) +  sum(NN_U0(1)^2)        # + bdry conditions (Eqn A.9.)

p=params(NN_U1)

#train parameters in NN_U1 based on loss function, repeat the training iteration on the data points for iterN times
iterN=1000
Flux.train!(loss,p,Iterators.repeated(data_tn, iterN), ADAM(0.1))


#prediciton of solution at time n+1 at location x=[x0,x1,x2,x3...]
U1_star=NN_U1(x)[p+1]
