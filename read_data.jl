using MAT
data=matread("./Data/burgers_shock.mat")
t=data["t"]
x=data["x"]
exact=data["usol"]