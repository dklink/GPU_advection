import generate_field
import numpy as np
import matplotlib.pyplot as plt
import cv2

field = generate_field.hycom_surface(months=list(range(1, 13)))

t = 0
lat_min = 20

#X, Y = np.meshgrid(field.x, field.y[-lat_min:])

#plt.quiver(X, Y, field.U[t, :, -lat_min:], field.V[t, :, -lat_min:])

# plot magnitude of currents
plt.figure()
plt.contourf(field.x, field.y, np.sqrt(field.U[t]**2 + field.V[t]**2).T)
plt.colorbar()
plt.title('surface current speed (2015-01-01)')
# ^ this plot answers our question, the guttering is occuring outside the bounds of our currents, i.e. north of 80N

# plot divergence
div = np.gradient(np.mean(field.U, axis=0), axis=0) + np.gradient(np.mean(field.V, axis=0), axis=1)
# smooth the divergence before plotting
filter = np.ones([50, 50])
div[np.isnan(div)] = 0
div = cv2.blur(div, (15, 15))
div[np.isnan(field.U[t])] = np.nan

plt.figure()
plt.contourf(field.x, field.y, div.T, cmap=plt.get_cmap('seismic'))
plt.colorbar()
plt.title('surface current divergence (2015 mean)')