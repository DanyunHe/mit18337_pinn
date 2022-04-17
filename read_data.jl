using MAT
data=matread("./Data/burgers_shock.mat")
t=data['tt']
x=data['x']
exact=data['uu']