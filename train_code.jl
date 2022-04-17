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

#Nueral Net of solutions at q stages and at time n+1, LHS of eqn 7
#input: x vector of length Ndata, output: solutions at locations x.
NN_U1 = Chain(Dense(1,200,tanh),
           Dense(200,200,tanh),
           Dense(200,q+1))


NN_U0 =  #RHS of Eqn 9:RK scheme of NN_U1, and nonlinear operation from specific PDE (above Eqn.(A.9))

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
U1_star=NN_U1(x)[p+1]

#Error calculation
error=norm(U1_star.-exact[idx_t1,:])/norm(exact[idx_t1,:])
