using DelimitedFiles

sample_size=[10,50,100,150,200,250]
N=length(sample_size)
fn="./result"

total_time=zeros(N)
total_err=zeros(N)
total_loss=zeros(101,N)
total_iter=zeros(101,N)
for i in 1:N
    size=sample_size[i]
    data=readdlm("$fn/out_$size/PINN_error.txt");
    total_err[i]=data[1]
    total_time[i]=data[2]
    temp=readdlm("$fn/out_$size/PINN_iter_loss_MSE.txt");
    total_loss[:,i]=temp[1:101,3]
    total_iter[:,i]=temp[1:101,1]
end

using Plots

set_theme!(fontsize_theme)
plt=plot(sample_size,total_err,title="Relative L2 error",marker=:dot,yaxis=:log,legend=false,xtickfont = font(10),ytickfont = font(10))
xlabel!("number of samples")
ylabel!("relative L2 error")
savefig(plt,"samplesize_err.pdf")

plt=plot(sample_size,total_time,title="Computation time",marker=:dot,c=:red,legend=false,xtickfont = font(10),ytickfont = font(10))
xlabel!("number of samples")
ylabel!("computation time (s)")
savefig(plt,"samplesize_time.pdf")

plt=plot(total_iter[:,1],total_loss[:,1],title="Loss",yaxis=:log,xtickfont = font(15),ytickfont = font(10),label="N=$(sample_size[1])")
for i in 2:N
    plot!(total_iter[:,i],total_loss[:,i],title="Loss",yaxis=:log,xtickfont = font(15),ytickfont = font(10),label="N=$(sample_size[i])")
end
xlabel!("number of iterations")
ylabel!("loss")
savefig(plt,"samplesize_loss.pdf")

