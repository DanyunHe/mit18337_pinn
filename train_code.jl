
#solution data at time n
data_tn=

#Nueral Net of solutions at q stages and at time n+1, LHS of eqn 7
NN_U1 = Chain(Dense(10,32,tanh),
           Dense(32,32,tanh),
           Dense(32,5))

NN_U0=  #RHS of Eqn 9:RK scheme of NN_U1, and nonlinear operation from specific PDE

loss()= # (NN_U0- data at un (Eqn 8))+ bdry conditions

p=params(NN_U1)

#train parameters in NN_U1 based on loss function, repeat the training iteration on the data points for iterN times
iterN=1000
Flux.train!(loss,p,Iterators.repeated(data_tn, iterN), ADAM(0.1))


#prediciton of solution at time n+1 at location x=[x0,x1,x2,x3...]
U1_star=NN_U1(x)[p+1]
