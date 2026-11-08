import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import plotly.graph_objects as go
import sympy as smp
from sympy.vector import cross

I = float(input("Enter the current value (A): "))
phi = np.linspace(0, 2*np.pi, 200)
mu_0 = 4 * np.pi * 1e-7
R = float(input("Enter the radius value (M): "))

def circle():
    return R * np.array([np.cos(phi), np.sin(phi), np.zeros(len(phi))])

def cardioid():
    return np.array([(2*np.cos(phi)-np.cos(2*phi)), (2*np.sin(phi) - np.sin(2*phi)), np.zeros(len(phi))])

def spring():
    return np.array([np.cos(phi), np.sin(phi), (phi-np.pi)/np.pi])

# plot the current shape in 2d
lx, ly, lz = circle()
plt.figure(figsize=(6,6))
plt.plot(lx, ly)
plt.xlabel=("x")
plt.ylabel=("y")
plt.show()

# solve the biot savart integral using symbolic computation library
t, x, y, z = smp.symbols("t, x, y, z")
l = R * smp.Matrix([smp.cos(t), smp.sin(t), 0]) # for circle
# l = smp.Matrix([(2*smp.cos(t)-smp.cos(2*t)), (2*smp.sin(t) - smp.sin(2*t)), 0]) # for cardioid
# l = smp.Matrix([smp.cos(t), smp.sin(t), (t-smp.pi)/np.pi]) # for spring
r = smp.Matrix([x, y, z])

displacement = r - l
integrand = smp.diff(l, t).cross(displacement) / displacement.norm()**3

# compute the integrand in 3d
dBxdt = smp.lambdify([t, x, y, z], integrand[0])
dBydt = smp.lambdify([t, x, y, z], integrand[1])
dBzdt = smp.lambdify([t, x, y, z], integrand[2])

# computing the integral in 3d
def B(x, y, z):
    return np.array([
        quad(dBxdt, 0, 2*np.pi, args=(x, y, z))[0],
        quad(dBydt, 0, 2*np.pi, args=(x, y, z))[0],
        quad(dBzdt, 0, 2*np.pi, args=(x, y, z))[0]
    ])

# creating 3d plot
x = np.linspace(-2*R, 2*R, 20)
xv, yv, zv = np.meshgrid(x, x, x)

B_field = np.vectorize(B, signature='(), (), ()->(n)')(xv, yv, zv)
B_field = (mu_0 * I) / (4 * np.pi) * B_field

Bx = B_field[:,:,:,0]
By = B_field[:,:,:,1]
Bz = B_field[:,:,:,2]

# drawing 3d plot
data = go.Cone(x=xv.ravel(), y=yv.ravel(), z=zv.ravel(),
            u=Bx.ravel(), v=By.ravel(), w=Bz.ravel(),
            colorscale="Inferno", colorbar=dict(title="B: "),
            sizemode="scaled", sizeref=1
)

layout = go.Layout(title="Plot title", scene=dict(
  xaxis_title="x", yaxis_title="y", zaxis_title="z" ,
  aspectratio=dict(x=1, y=1, z=1), camera_eye=dict(x=0.6, y=0.6, z=0.6) 
))

figure = go.Figure(data=data, layout=layout)
figure.add_scatter3d(x=lx, y=ly, z=lz, mode="lines", line=dict(color="green", width=10))

figure.show()