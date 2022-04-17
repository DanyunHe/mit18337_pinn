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

#Nueral Net of solutions at q stages and at time n+1, LHS of eqn 7
#input: x vector of length Ndata, output: solutions at locations x.
NN_U1 = Chain(Dense(Ndata,200,tanh),
           Dense(200,200,tanh),
           Dense(200,Ndata))

NN_U0 =  #RHS of Eqn 9:RK scheme of NN_U1, and nonlinear operation from specific PDE (above Eqn.(A.9))

loss()= sum(NN_U0(data_tn_x)-data_tn_u)^2) # (NN_U0- data at un (Eqn 8)) (Eqn. A.9.)
        + sum(NN_U0(-1)^2) +  sum(NN_U0(1)^2)        # + bdry conditions (Eqn A.9.)

p=params(NN_U1)

#train parameters in NN_U1 based on loss function, repeat the training iteration on the data points for iterN times
iterN=1000
Flux.train!(loss,p,Iterators.repeated(data_tn, iterN), ADAM(0.1))


#prediciton of solution at time n+1 at location x=[x0,x1,x2,x3...]
U1_star=NN_U1(x)[p+1]
