#1D Burger's equation, Eqn (A.1.)
#t: 0 to 1; x: -1 to 1
#initial and boundary conditions
using MAT
using StatsBase
using DelimitedFiles
using Flux
using ForwardDiff
using LinearAlgebra
using Plots
using BSON: @save
using BSON: @load


using DiffEqFlux
using Optim

using Zygote
using FluxOptTools
using Statistics

# read data
data=matread("./Data/burgers_shock.mat")
t=data["t"]  #length 100
x=data["x"]  #length 256
exact=data["usol"] #length 256x100, exact[:,1] is 256 data at time t[1]
total_data_size=256
#list of stages
q_list=[100,32,4,1] #4 stages
#list of dt's
dt_list=[0.1,0.2] #3 dt's

for dti in 1:4
        for qi in 1:2

                # number of stages
                q=q_list[qi]
                #timestep size
                dt=dt_list[dti] #t[idx_t1]-t[idx_t0]

                #define layers
                layers=[1,50,50,50,q+1]   #4 layers, 50 neuron/layer
                #boundaries in x
                lb=[-1.]
                ub=[1.]
                #t0 and tfinal
                idx_t0=11
                #idx_tf=41
                #total number of steps
                #dt_steps=Int32((t[idx_tf]-t[idx_t0])/dt)
                #one step final time
                idx_t1=Int32(idx_t0+(dt/0.1)*10)

                # load IRK weights
                temp=readdlm("./IRK_weights/Butcher_IRK$q.txt");
                IRK_weights=reshape(temp[1:q^2+q],(q,q+1))
                IRK_weights=IRK_weights'
                IRK_times=temp[q^2+q:size(temp)[1]]

                #read in full data at time t0
                read_data_tn_u=exact[:,idx_t0]
                #the corresponding locations of the read in full data
                read_data_tn_x=x
                #total number of read in full data:
                total_data_size=size(read_data_tn_u)[1] #value is 256
                #number of data point Ndata that we will sample
                Ndata=250
                #the corresponding location x index of random uniform sampled data at time t0
                sample_data_location_index = sample(1:total_data_size, Ndata, replace = false)
                #obtain sampled data at time t0: location x and data values
                data_tn_x=zeros(Ndata)
                data_tn_u=zeros(Ndata)
                for i in 1:Ndata
                        data_tn_x[i]=read_data_tn_x[sample_data_location_index[i]]
                        data_tn_u[i]=read_data_tn_u[sample_data_location_index[i]]
                end

                #Nueral Net of solutions at q stages and at time n+1, LHS of eqn 7
                #input: x vector of length Ndata, output: solutions at locations x.
                NN = Chain(Dense(layers[1],layers[2],tanh),  #1x50
                           Dense(layers[2],layers[3],tanh),  #50x50
                           Dense(layers[3],layers[4],tanh),  #50x50
                           Dense(layers[4],layers[5]))       #50x(q+1)=50x101

                function NN_U1(x)
                    U1_pred = NN(x)
                    return U1_pred # q*1
                end

                function NN_U0(x)
                    nu = 0.01/pi
                    U1 = NN(x) # (q+1)*1
                    U = U1[1:q] # q*1
                    U_x = ForwardDiff.jacobian(NN, x)[1:q]# q*1
                    U_xx = ForwardDiff.jacobian(y -> ForwardDiff.jacobian(NN_U1, y), x)[1:q]# q*1
                    F = -U.*U_x + nu*U_xx # q*1
                    U0 = U1 - dt*(IRK_weights * F)
                    return U0
                end
                #Eqn 8, Eqn A.9.
                function loss()
                        total_loss=0
                        #loop through N data points at time tn, and calculate and add up losses
                        for i in 1:Ndata  #time n is t[idx_t0]
                                #fill an array of length p+1 with value of exact soln at time n
                                total_loss=total_loss+sum(abs2,NN_U0([data_tn_x[i]]).-data_tn_u[i])
                        end

                        #add in boundary condition losses: u(x=-1)=0, u(x=1)=0
                        total_loss = total_loss+sum(abs2,NN_U1([-1.]))+sum(abs2,NN_U1([1.]))
                        return total_loss
                end

                #train parameters in NN_U1 based on loss function, repeat the training iteration on the data points
                #total number of iterations of training: 10000
                #Save model parameters and loss value and predicted solution error every 100 iterations
                total_iteration=10000   #10000
                iterN=100
                number_big_step=total_iteration/iterN

                loss_array = Vector{Float64}()
                MSE_array = Vector{Float64}()
                iteration_array = Vector{Int32}()

                current_loss_0=loss()
                MSE_0=current_loss_0/(Ndata+2)
                append!(loss_array,current_loss_0)
                append!(MSE_array,MSE_0)
                append!(iteration_array,0)

                @save "./vary_stage_dt/PINN_NN_model_$(q)_$(dt)_0" NN
                open("./vary_stage_dt/PINN_iter_loss_MSE_$(q)_$(dt).txt", "a") do file
                    println(file, "0 $current_loss_0 $MSE_0 ")
                    flush(file)
                end


                for iteri in 1:number_big_step #1:number_big_step
                        Zygote.refresh()
                        p=Flux.params(NN)

                        #Flux.train!(loss,p,Iterators.repeated((), iterN), ADAM())


                        if iteri<=50 #5000 ADAM
                                Flux.train!(loss,p,Iterators.repeated((), iterN), ADAM()) #train iterN=100 times
                        else #then, use BFGS() to train the model 5000 iters
                                lossfun, gradfun, fg!, p0 = optfuns(loss, p)
                                res = Optim.optimize(Optim.only_fg!(fg!), p0, BFGS(), Optim.Options(iterations=iterN))
                        end

                        #save model parameters
                        total_iteration_i=iteri*iterN
                        @save "./vary_stage_dt/PINN_NN_model_$(q)_$(dt)_$(total_iteration_i)" NN

                        #compute and save loss function value, MSE value
                        current_loss_i=loss()
                        MSE_i=current_loss_i/(Ndata+2)
                        append!(loss_array,current_loss_i)
                        append!(MSE_array,MSE_i)
                        append!(iteration_array,total_iteration_i)

                        open("./vary_stage_dt/PINN_iter_loss_MSE_$(q)_$(dt).txt", "a") do file
                            println(file, "$total_iteration_i $current_loss_i $MSE_i ")
                            flush(file)
                        end
                end

                #prediciton of solution at time n+1 at location x=[x0,x1,x2,x3...]
                U1_star=Array{Float64}(undef, total_data_size)
                for i in 1:total_data_size
                        U1_star[i]=NN_U1([x[i]])[q+1]
                end

                #Error calculation of predicted solution at time t(n+1)
                finaltime=t[idx_t1]
                error=norm(U1_star.-exact[:,idx_t1])/norm(exact[:,idx_t1])
                println("Final time $finaltime relative L2 error $error")
                #save error to file
                open("./vary_stage_dt/PINN_error_vary_q_dt.txt", "a") do file
                    println(file, "$q $dt $error ")
                    flush(file)
                end

                #plot
                #plot(iteration_array, loss_array, xlabel="iteration", ylabel="PINN SSE loss")
                #plot(iteration_array, loss_array./(Ndata+2), xlabel="iteration", ylabel="PINN MSE loss")
                img=plot(x, [U1_star,exact[:,idx_t1]], labels=["predicted soln at final time" "exact solution"],  xlabel="x",ylabel="solution u",title="t= $finaltime")
                savefig(img,"./vary_stage_dt/PINN_soln_plot_$(q)_$(dt).png")

        end
end
