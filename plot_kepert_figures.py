from kepert import Kepert_linear_assym
import numpy as np

f = open('rmw.xy', 'w')
f1 = open('us.xyz', 'w')
f2 = open('vs.xyz', 'w')
f3 = open('usf.xyz', 'w')
f4 = open('vsf.xyz', 'w')
f5 = open('swf1.xyz', 'w')
f6 = open('swf2.xyz', 'w')
f7 = open('swf3.xyz', 'w')
f8 = open('swf4.xyz', 'w')
f9 = open('u1.xyz', 'w')
f10 = open('u2.xyz', 'w')
f11 = open('v1.xyz', 'w')
f12 = open('v2.xyz', 'w')

nz = 400

x = np.arange(-150e3, 151e3, 2.5e3)
y = np.arange(-150e3, 151e3, 2.5e3)
z = np.arange(5, nz*5+1, 5)
xgrid, ygrid = np.meshgrid(x, y) #np.resize(np.arange(-150e3,151e3,5e3), (61, 61))

xkm = xgrid*1e-3
ykm = ygrid*1e-3
xx = xkm[0,]

rm = 4e4
lp = 60

cx = rm*1e-3*np.sin(np.arange(0.00,6.29,0.01))  # Circle for rmw plots.
cy = rm*1e-3*np.cos(np.arange(0.00,6.29,0.01))

p = Kepert_linear_assym(xgrid, ygrid, lat=-15, rm=4e4, vm=65.0, b=1.4, \
                        dp=9680.13292308, pe=1e5, rho=1.15, Ut=-5)

#output circle for rmw plots

for i in range(len(cx)):
    f.write(repr(cx[i]) + ',' + repr(cy[i]) + '\n')

#output horizontal wind fields

(a,b) = xkm.shape

for i in range(a):
    for j in range(b):
        f1.write(repr(xkm[i,j]) + ',' + repr(ykm[i,j]) + ',' + \
                 repr(p.us[i,j]) + '\n')
        f2.write(repr(xkm[i,j]) + ',' + repr(ykm[i,j]) + ',' + \
                 repr(p.vs[i,j]) + '\n')
        f3.write(repr(xkm[i,j]) + ',' + repr(ykm[i,j]) + ',' + \
                 repr(p.usf[i,j]) + '\n')
        f4.write(repr(xkm[i,j]) + ',' + repr(ykm[i,j]) + ',' + \
                 repr(p.vsf[i,j]) + '\n')
        f5.write(repr(xkm[i,j]) + ',' + repr(ykm[i,j]) + ',' + \
                 repr(p.swf1[i,j]) + '\n')
        f6.write(repr(xkm[i,j]) + ',' + repr(ykm[i,j]) + ',' + \
                 repr(p.swf2[i,j]) + '\n')
        f7.write(repr(xkm[i,j]) + ',' + repr(ykm[i,j]) + ',' + \
                 repr(p.swf3[i,j]) + '\n')
        f8.write(repr(xkm[i,j]) + ',' + repr(ykm[i,j]) + ',' + \
                 repr(p.swf4[i,j]) + '\n')


#output vertical wind fields

(a,b,c) = p.u.shape

for i in range(a):
    for j in range(b):
        f9.write(repr(xx[j]) + ',' + repr(z[i]) + ',' + \
                 repr(p.u[i,lp+1,j]) + '\n')
        f10.write(repr(xx[j]) + ',' + repr(z[i]) + ',' + \
                 repr(p.u[i,j,lp+1]) + '\n')
        f11.write(repr(xx[j]) + ',' + repr(z[i]) + ',' + \
                 repr(p.v[i,lp+1,j]) + '\n')
        f12.write(repr(xx[j]) + ',' + repr(z[i]) + ',' + \
                 repr(p.v[i,j,lp+1]) + '\n')

f.close()
f1.close()
f2.close()
f3.close()
f4.close()
f5.close()
f6.close()
f7.close()
f8.close()
f9.close()
f10.close()
f11.close()
f12.close()
