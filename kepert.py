from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cmath

import pdb

sns.set_style('ticks', {'image.cmap':'coolwarm'})
sns.set_context('talk')

class Kepert_linear_assym:

    """Analytical linear model of the tropical cyclone boundary layer
       as in Kepert (JAS, 2001).

       Written Jeff Kepert, Bureau of Meteorology, 1998-2000.
       Copyright the Bureau of Meteorology.
       Please do not distribute without my knowledge.

       Conventions: All quantities in SI units for calculation, sometimes
       modified for plotting (eg m -> km).
       The solution is calculated on a cartesian grid, though wind variables
       (u,v,w) are in cylindrical polar coordinates (lam,r,z). Winds in
       cartesian coords are (ux,vx) etc. An a on appended to
       variables often refer to the aymmetric parts, s refers to surface
       conditions, 0, p and m refer to k = 0, +1, -1.

       Two possibilities for the core are given - quadratic or cubic. The
       former has a discontinuity in the second radial derivative of the
       vorticity at the rmw, which makes w discontinuous there. The latter
       is ok there.
    """

    def __init__(self, x, y, lat, rm, vm, b, dp, pe, rho, Ut):

        f = 2*7.292e-5*np.sin(lat*np.pi/180.0)  # Coriolis parameter
        thetaFm = 0
        tx = 1.0 * x # to create a new copy of x, not a pointer to x
        np.putmask(tx, x==0, 1e-30) # to avoid divide by zero errors
        lam = np.arctan2(y,tx)
        r = np.sqrt(x**2 + y**2)
        np.putmask(r, r==0, 1e-30) # to avoid divide by zero errors

        xkm = x*1e-3
        ykm = y*1e-3
        xx = xkm[0,]
        yy = ykm[:,0]
        cx = rm*1e-3*np.sin(np.arange(0,6.29,0.01))  # Circle for rmw plots.
        cy = rm*1e-3*np.cos(np.arange(0,6.29,0.01))
        # Define parametric gradient wind (V) and vorticity (Z) profiles
        # using Holland (1980) bogus with core modified to quadratic.

        rmrb = (rm/r)**b
        rmrbs = rmrb.shape
        emrmrb = np.zeros(rmrbs, float)
        for i in range(rmrbs[0]):  # needed to cope with Python OverflowError
            for j in range(rmrbs[1]):
                try:
                    emrmrb[i,j] = np.exp(-rmrb[i,j])
                except OverflowError:
                    pass
        V = np.sqrt((b*dp/rho) * emrmrb * rmrb + (0.5*f*r)**2) - 0.5*abs(f)*r

        if dp < 4000.:
            Umod = Ut* (1. - (4000. - dp)/4000.)
        else:
            Umod = Ut

        if (np.abs(V).max()/np.abs(Ut) < 5):
            Umod = Umod * (1. - (5. - np.abs(V).max()/np.abs(Ut))/5.)
        else:
            Umod = Umod


        fig, axes = plt.subplots(2, 2,subplot_kw={'aspect':'equal'},
                                 figsize=(18,18))
        ax = axes.flatten()
        #ax[0].hold(True)
        ax[0].set_xlim([-50,50])
        ax[0].set_ylim([-50,50])
        levels = np.arange(-20, 21, 2)
        cm = ax[0].contourf(xkm, ykm, V, np.arange(-100., 101, 5))
        cs = ax[0].contour(xkm, ykm, V, np.arange(-100, 101, 5), colors='k')
        ax[0].clabel(cs, fontsize='x-small', fmt='%1.2f') #, range(-20,11,4))
        plt.colorbar(cm, ax=ax[0])
        ax[0].plot(cx,cy,'w')
        ax[0].plot(2*cx,2*cy,'w')

        Vt = np.ones(rmrbs, float) * Umod
        core = np.where(r >= 2.*rm)
        #Vt[core] = Umod*np.exp(-((r[core]/(2.*rm)) - 1.)**2.)

        Z = abs(f) + (b**2*dp*(rmrb**2)*emrmrb/(2*rho*r) -\
                      b**2*dp*rmrb*emrmrb/(2*rho*r) + r*f**2/4) \
            / np.sqrt(b*dp*rmrb*emrmrb/rho + (r*f/2)**2) + \
            (np.sqrt(b*dp*rmrb*emrmrb/rho + (r*f/2)**2))/r

        #ax[1].hold(True)
        ax[1].set_xlim([-50, 50])
        ax[1].set_ylim([-50, 50])
        levels = np.arange(-0.02, 0.021, .002)
        cm = ax[1].contourf(xkm, ykm, Z, np.arange(-0.02, 0.021, .002))
        cs = ax[1].contour(xkm, ykm, Z, np.arange(-0.02, 0.021, .002),
                           colors='k')
        ax[1].clabel(cs, fontsize='x-small', fmt='%1.2f') #, range(-20,11,4))
        plt.colorbar(cm, ax=ax[1])
        ax[1].plot(cx,cy,'w')

        if 0:
            # Quadratic core (NB."if" part of tree not yet Python-ised)
            Vm = V.max()
            rs = r[np.where(V==Vm)]
            rs = rs[0]
            rmrsb = (rm/rs)**b
            Vs = np.sqrt(b*dp/rho * np.exp(-rmrsb) * rmrsb + (0.5*f*rs)**2)
            - 0.5*abs(f)*rs
            icore = np.where(r<rs)
            V[icore] = Vs * (r[icore]/rs) * (2 - (r[icore]/rs))
            Z[icore] = Vs/rs * (4 - 3*r[icore]/rs)
        else:
            # Fit cubic at rm, matching derivatives
            E = np.exp(1)
            Vm = (np.sqrt(b*dp/(rho*E) + (0.5*f*rm)**2) - 0.5*abs(f)*rm)
            dVm = (-np.abs(f)/2 + (E*(f**2)*rm*np.sqrt((4*b*dp/rho)/E + \
                                                       (f*rm)**2))/ \
                          (2*(4*b*dp/rho + E*(f*rm)**2)))
            d2Vm = (b*dp*(-4*b**3*dp/rho - (-2 + b**2)*E*(f*rm)**2)) / \
                    (E*rho*np.sqrt((4*b*dp)/(E*rho) + (f*rm)**2) * \
                      (4*b*dp*rm**2/rho + E*(f*rm**2)**2))
            aa = (d2Vm/2 - (dVm - Vm/rm)/rm) / rm
            bb = (d2Vm - 6*aa*rm) / 2
            cc = dVm - 3*aa*rm**2 - 2*bb*rm
            icore = np.nonzero(np.ravel(r<rm))
            for ind in icore:
                V.flat[ind] = r.flat[ind] * \
                                 (r.flat[ind] * (r.flat[ind]*aa + bb) + cc)
                Z.flat[ind] = r.flat[ind] * (r.flat[ind] * 4*aa + 3*bb) + 2*cc
                pm = pe - dp*(1 - np.exp(-1))

        V = V*np.sign(f)
        Z = Z*np.sign(f)

        #ax[2].hold(True)
        #set(plt.gca(),'DataAspectRatio',[1,1,1])
        ax[2].set_xlim([-50,50])
        ax[2].set_ylim([-50,50])
        levels = np.arange(-20, 21, 2)
        cm = ax[2].contourf(xkm, ykm, V, np.arange(-100., 101, 5))
        cs = ax[2].contour(xkm, ykm, V, np.arange(-100., 101, 5), colors='k')
        ax[2].clabel(cs, fontsize='x-small', fmt='%1.2f') #, range(-20,11,4))
        plt.colorbar(cm, ax=ax[2])
        ax[2].plot(cx,cy,'w')

        #ax[3].hold(True)
        #set(plt.gca(),'DataAspectRatio',[1,1,1])
        ax[3].set_xlim([-50,50])
        ax[3].set_ylim([-50,50])
        levels = np.arange(-0.02, 0.021, .002)
        cm = ax[3].contourf(xkm, ykm, Z, np.arange(-0.02, 0.021, .002))
        cs = ax[3].contour(xkm, ykm, Z, np.arange(-0.02, 0.021, .002),
                           colors='k')
        ax[3].clabel(cs, fontsize='x-small', fmt='%1.2f') #, range(-20,11,4))
        plt.colorbar(cm, ax=ax[3])
        ax[3].plot(cx,cy,'w')
        fig.tight_layout()
        plt.savefig('radial_wind_vorticity.png')

        Vx = -V*y/r  # Cartesian winds (??? divide by y ???)
        Vy =  V*x/r

        I2 = (f + 2*V/r)*(f + Z)  # Inertial stability squared

        K = 50    # Diffusivity
        C = 0.002 # Drag coefficient

        # Calculate alpha, beta, gamma, chi, eta and psi in Kepert (2001).
        # The III's are to get the solution in the case where V/r > I.
        al = (f + (2 * V / r)) / (2*K)
        be = (f + Z) / (2*K)
        gam = V / (2*K*r)

        III = np.nonzero(gam > np.sqrt(al*be))
#        fixup = nonzero(ravel(isNaN(al) | isnan(be) | isnan(gam)))
        chi = np.abs((C/K)*V / np.sqrt(np.sqrt(al*be)))
#        chi[fixup] = np.nan
        eta = np.abs((C/K)*V / np.sqrt(np.sqrt(al*be) + np.abs(gam)))
#        eta[fixup] = np.nan
        psi = np.abs((C/K)*V / np.sqrt(np.abs(np.sqrt(al*be) - np.abs(gam))))
#        psi[fixup] = nan
        albe = np.sqrt(al/be)

        # Calculate A_k's ... p and m refer to k = +1 and -1.
        i = cmath.sqrt(-1)
        A0 =  -chi*V*(1 + i*(1 + chi)) / (2*chi**2 + 3*chi + 2)
#        A0[fixup] = np.nan
        u0s = np.sign(f)* albe * A0.real   # Symmetric surface wind component
        v0s =                    A0.imag

        Am = -(psi * (1 + 2 * albe + (1 + i) * (1 + albe) * eta) * Umod) \
             / (albe * ((2 + 2 * i) * (1 + eta * psi) + 3 * psi + 3* i * eta))
        AmIII = -(psi * (1 + 2 * albe + (1 + i) * (1 + albe) * eta) * Umod) \
                / (albe * ((2 - 2 * i + 3 * (eta + psi) + (2 + 2 * i)*eta*psi)))

        Am[III] = AmIII[III]

        # First asymmetric surface component
        ums =       albe * (Am * np.exp(-i*(thetaFm - lam)*np.sign(f))).real
        vms = np.sign(f) * (Am * np.exp(-i*(thetaFm - lam)*np.sign(f))).imag


        Ap = -(eta * (1 - 2 * albe + (1 + i)*(1 - albe) * psi) * Umod) \
             / (albe * ((2 + 2*i)*(1 + eta * psi) + 3*eta + 3*i*psi))
        ApIII = -(eta * (1 - 2 * albe + (1 - i)*(1 - albe)*psi)*Umod) \
                / (albe * (2 + 2 * i + 3 * (eta + psi) + (2 -2 * i)*eta*psi))
        Ap[III] = ApIII[III]

        # Second asymmetric surface component
        ups =       albe * (Ap * np.exp(i*(thetaFm - lam)*np.sign(f))).real
        vps = np.sign(f) * (Ap * np.exp(i*(thetaFm - lam)*np.sign(f))).imag


        fig, axes = plt.subplots(2, 2, figsize=(18, 18),
                                 subplot_kw={'aspect':'equal'})
        ax = axes.flatten()
        #ax[0].hold(True)
        #set(plt.gca(),'DataAspectRatio',[1,1,1])
        ax[0].set_xlim([-100,100])
        ax[0].set_ylim([-100,100])
        levels = np.arange(-20, 21, 2)
        cm = ax[0].contourf(xkm, ykm, u0s, np.arange(-50, 51, 5))
        cs = ax[0].contour(xkm, ykm, u0s, np.arange(-50, 51, 5), colors='k')
        ax[0].clabel(cs, fontsize='x-small', fmt='%1.2f') #, range(-20,11,4))
        plt.colorbar(cm, ax=ax[0])
        ax[0].plot(cx,cy,'w')
        ax[0].plot(2*cx,2*cy,'w')
        ax[0].set_xlabel("u0s")

        #ax[1].hold(True)
        #set(plt.gca(),'DataAspectRatio',[1,1,1])
        ax[1].set_xlim([-100,100])
        ax[1].set_ylim([-100,100])
        levels = np.arange(-20, 21, 2)
        cm = ax[1].contourf(xkm, ykm, ups, np.arange(-50, 51, 5))
        cs = ax[1].contour(xkm, ykm, ups, np.arange(-50, 51, 5), colors='k')
        ax[1].clabel(cs, fontsize='x-small', fmt='%1.2f') #, range(-20,11,4))
        plt.colorbar(cm, ax=ax[1])
        ax[1].plot(cx,cy,'w')
        ax[1].plot(2*cx,2*cy,'w')
        ax[1].set_xlabel("ups")

        #ax[2].hold(True)
        #set(plt.gca(),'DataAspectRatio',[1,1,1])
        ax[2].set_xlim([-100,100])
        ax[2].set_ylim([-100,100])
        levels = np.arange(-20, 21, 2)
        cm = ax[2].contourf(xkm, ykm, ums, np.arange(-50, 51, 5))
        cs = ax[2].contour(xkm, ykm, ums, np.arange(-50, 51, 5), colors='k')
        ax[2].clabel(cs, fontsize='x-small', fmt='%1.2f') #, range(-20,11,4))
        plt.colorbar(cm, ax=ax[2])
        ax[2].plot(cx,cy,'w')
        ax[2].plot(2*cx,2*cy,'w')
        ax[2].set_xlabel("ums")



        # Total surface wind in (moving coordinate system)
        us =     (u0s + ups + ums)
        vs = V + (v0s + vps + vms)

        #ax[3].hold(True)
        #set(plt.gca(),'DataAspectRatio',[1,1,1])
        ax[3].set_xlim([-100,100])
        ax[3].set_ylim([-100,100])
        levels = np.arange(-20, 21, 2)
        cm = ax[3].contourf(xkm, ykm, us, np.arange(-50, 51, 5))
        cs = ax[3].contour(xkm, ykm, us, np.arange(-50, 51, 5), colors='k')
        ax[3].clabel(cs, fontsize='x-small', fmt='%1.2f') #, range(-20,11,4))
        plt.colorbar(cm, ax=ax[3])
        ax[3].plot(cx,cy,'w')
        ax[3].plot(2*cx,2*cy,'w')
        ax[3].set_xlabel("us")
        fig.tight_layout()
        plt.savefig("radial_components.png")

        # Total surface wind in stationary coordinate system (f for fixed)
        usf = us + Umod*np.cos(lam - thetaFm)
        vsf = vs - Umod*np.sin(lam - thetaFm)
        Uf =   + Umod*np.cos(lam)
        Vf = V - Umod*np.sin(lam)
        phi = np.arctan2(usf, vsf)
        Ux = np.sqrt(usf ** 2. + vsf ** 2.) * np.sin(phi - lam)
        Vy = np.sqrt(usf ** 2. + vsf ** 2.) * np.cos(phi - lam)
        #Figure 1 here
        fig, axes = plt.subplots(2, 2,subplot_kw={'aspect':'equal'},figsize=(18,18))
        ax = axes.flatten()
        #plt.clf()
        #plt.subplot(221, aspect='equal')
        #plt.cla()
        #ax[0].hold(True)
        #set(plt.gca(),'DataAspectRatio',[1,1,1])
        ax[0].set_xlim([-150,150])
        ax[0].set_ylim([-150,150])
        levels = np.arange(-20, 21, 2)
        cm = ax[0].contourf(xkm, ykm, usf, np.arange(-50., 51, 2))
        cs = ax[0].contour(xkm, ykm, usf, np.arange(-50, 51, 2), colors='k')
        ax[0].clabel(cs, fontsize='x-small', fmt='%1.2f') #, range(-20,11,4))
        plt.colorbar(cm, ax=ax[0])
        ax[0].plot(cx,cy,'w')
        ax[0].plot(2*cx,2*cy,'w')
        ax[0].set_xlabel('Storm-relative surface u')

        #plt.subplot(222,  aspect='equal')
        #plt.cla()
        #plt.hold(True)
        ax[1].set_xlim([-150,150])
        ax[1].set_ylim([-150,150])
        cm = ax[1].contourf(xkm,ykm,vsf,range(-100,101,5))
        cs = ax[1].contour(xkm,ykm,vsf,range(-100,101,5),colors='k')
        ax[1].clabel(cs, fontsize='x-small', fmt='%1.2f')
        plt.colorbar(cm, ax=ax[1])
        ax[1].plot(cx,cy,'w')
        ax[1].plot(2*cx,2*cy,'w')
        ax[1].set_xlabel('Storm-relative surface v')

        #plt.subplot(223, aspect='equal')
        #plt.cla()
        #plt.hold(True)
        ax[2].set_xlim([-150,150])
        ax[2].set_ylim([-150,150])
        cm = ax[2].contourf(xkm,ykm,Ux,range(-100, 101, 5))
        cs = ax[2].contour(xkm,ykm,Ux,range(-100, 101, 5),colors='k')
        ax[2].clabel(cs, fontsize='x-small', fmt='%1.2f') #,range(-20,11,4))
        plt.colorbar(cm, ax=ax[2])
        ax[2].plot(cx,cy,'w')
        ax[2].plot(2*cx,2*cy,'w')
        ax[2].set_xlabel('Earth-relative surface u (cartesian)')

        #plt.subplot(224, aspect='equal')
        #plt.cla()
        #plt.hold(True)
        ax[3].set_xlim([-150,150])
        ax[3].set_ylim([-150,150])
        cm = ax[3].contourf(xkm,ykm,Vy,range(-100, 101, 5))
        cs = ax[3].contour(xkm,ykm,Vy,range(-100, 101, 5),colors='k')
        ax[3].clabel(cs, fontsize='x-small', fmt='%1.2f') #,h,range(0,51,10)
        plt.colorbar(cm, ax=ax[3])
        ax[3].plot(cx,cy,'w')
        ax[3].plot(2*cx,2*cy,'w')
        ax[3].set_xlabel('Earth-relative surface v (cartesian)')

        fig.suptitle('V_m=' + repr(vm) + ' r_m=' + repr(rm*1e-3) + ' b=' + repr(b) + \
                  ' lat=' + repr(lat) + ' K=' + repr(K) + ' C=' + repr(C))
        fig.tight_layout()
        plt.savefig('sfc_wind_components.png')


        # Four possible surface wind factors \ depending on whether you
        # use the total or azimuthal wind, and in moving or fixed coordinates.
        swf1 = np.abs(vs / V)
        swf2 = np.abs(np.sqrt(us**2 + vs**2) / V)
        swf3 = np.abs(vsf / Vf)
        swf4 = np.sqrt(usf**2 + vsf**2) / np.sqrt(Uf**2 + Vf**2)

        plt.clf()
        plt.subplot(111, aspect='equal')
        #plt.hold(True)
        plt.xlim([-150, 150])
        plt.ylim([-150, 150])
        mag = np.sqrt(Ux**2 + Vy**2) # Magnitude of the total surface wind
        cm = plt.contourf(xkm, ykm, mag, levels=np.arange(0, 101, 5))
        cs = plt.contour(xkm, ykm, mag, levels=np.arange(0, 101, 5), colors='k')
        plt.barbs(xkm[::5, ::5], ykm[::5, ::5], Ux[::5, ::5], Vy[::5, ::5],flip_barb=True)
        plt.clabel(cs, fontsize='x-small', fmt='%1.1f')
        plt.colorbar(cm)
        plt.plot(cx, cy, 'w')
        plt.plot(2*cx,2*cy,'w')
        plt.xlabel('Earth-relative total surface wind speed')
        plt.tight_layout()
        plt.savefig('sfc_total_wind.png')

        #Figure 2 here
        #figure(2)
        plt.clf()
        plt.subplot(221, aspect='equal')
        plt.cla()
        #plt.hold(True)
        plt.xlim([-150,150])
        plt.ylim([-150,150])
        cm = plt.contourf(xkm,ykm,swf1,levels=np.arange(0.5,1.55,0.05))
        cs = plt.contour(xkm,ykm,swf1,np.arange(0.5,1.55,0.05),colors='k')
        plt.clabel(cs)
        plt.colorbar(cm)
        plt.plot(cx,cy,'w')
        plt.plot(2*cx,2*cy,'w')

        plt.xlabel('Storm-relative azimuthal swrf')
        plt.title('V_m=' + repr(vm) + ' r_m=' + repr(rm*1e-3) + ' b=' + repr(b) + \
              ' lat=' + repr(lat) + ' K=' + repr(K) + ' C=' + repr(C))
        plt.subplot(222, aspect='equal')
        plt.cla()
        #plt.hold(True)

        plt.xlim([-150,150])
        plt.ylim([-150,150])
        cm = plt.contourf(xkm,ykm,swf2,levels=np.arange(0.5,1.55,0.05))
        cs = plt.contour(xkm,ykm,swf2,np.arange(0.5,1.55,0.05), colors='k')
        plt.clabel(cs)
        plt.colorbar(cm)
        plt.plot(cx,cy,'w')
        plt.plot(2*cx,2*cy,'w')
        plt.xlabel('Storm-relative total swrf')
        plt.subplot(223, aspect='equal')
        plt.cla()
        #plt.hold(True)
        plt.xlim([-150,150])
        plt.ylim([-150,150])
        cm = plt.contourf(xkm,ykm,swf3,levels=np.arange(0.5,1.55,0.05))
        cs = plt.contour(xkm,ykm,swf3,np.arange(0.5,1.55,0.05), colors='k')
        plt.clabel(cs)
        plt.colorbar(cm)
        plt.plot(cx,cy,'w')
        plt.plot(2*cx,2*cy,'w')
        plt.xlabel('Earth-relative azimuthal swrf')
        plt.subplot(224, aspect='equal')
        plt.cla()
        #plt.hold(True)
        plt.xlim([-150,150])
        plt.ylim([-150,150])
        cm = plt.contourf(xkm,ykm,swf4,levels=np.arange(0.5,1.55,0.05))
        cs = plt.contour(xkm,ykm,swf4,levels=np.arange(0.5,1.55,0.05), colors='k')
        plt.clabel(cs)
        plt.colorbar(cm)
        plt.plot(cx,cy,'w')
        plt.plot(2*cx,2*cy,'w')
        plt.xlabel('Earth-relative total swrf')
        plt.tight_layout()
        plt.savefig('swrf.png')


        # Coefficients p_k for k = 0, 1, -1.
        p0 = -(1 + i) * (al * be)**(1./4.)
        pp = -(1 + i) * np.sqrt(np.sqrt(al * be) + gam)
        pm = -(1 + i) * np.sqrt(np.abs(np.sqrt(al * be) - gam))
        pm[III] = -i * pm[III]

        # Set up a 3-d grid and calculate winds
        nz = 400
        z = np.arange(5, nz*5+1, 5)
        [ny,nx] = x.shape
        u = np.zeros((nz,ny,nx),float)
        v = np.zeros((nz,ny,nx),float)
        ua = np.zeros((nz,ny,nx),float)
        va = np.zeros((nz,ny,nx),float)
        ux = np.zeros((nz,ny,nx),float)  # x for cartesian wind
        vx = np.zeros((nz,ny,nx),float)
        uxa = np.zeros((nz,ny,nx),float)  # Asymmetric parts have trailing a
        vxa = np.zeros((nz,ny,nx),float)

#        for kk in range(1,nz+1):
        for kk in range(nz):
            # These w's are really omega's!
            wm = Am * np.exp(z[kk]*pm - i*lam*np.sign(f))
#            wm[fixup] = np.nan
            w0 = np.sign(f) * A0 * np.exp(z[kk]*p0)
#            w0[fixup] = np.nan
            wp = Ap * np.exp(z[kk]*pp + i*lam*np.sign(f))
#            wp[fixup] = np.nan

            u[kk] = albe*(np.sign(f) * wm + w0 + np.sign(f) * wp).real
            v[kk] = V + (np.sign(f) * wm + w0 + np.sign(f) * wp).imag
            ua[kk] = np.sign(f) * albe*(np.sign(f) * wm + np.sign(f) * wp).real
            va[kk] = (np.sign(f) * wm + np.sign(f) * wp).imag

#            ux[kk] = (squeeze(u[kk])*x - squeeze(v[kk])*y) / r
#            vx[kk] = (squeeze(v[kk])*x + squeeze(u[kk])*y) / r
#            uxa[kk] = (squeeze(ua[kk])*x - squeeze(va[kk])*y) / r
#            vxa[kk] = (squeeze(va[kk])*x + squeeze(ua[kk])*y) / r

            ux[kk] = (u[kk]*x - v[kk]*y) / r
            vx[kk] = (v[kk]*x + u[kk]*y) / r
            uxa[kk] = (ua[kk]*x - va[kk]*y) / r
            vxa[kk] = (va[kk]*x + ua[kk]*y) / r

        ucon = np.arange(-20, 21, 2)
        vcon = np.arange(-100,101,5)

        # Figure 3 here
        plt.figure(6)
        plt.clf()
        plt.subplot(221)
        plt.cla()
        lp = 60
        cm = plt.contourf(xx,z,np.squeeze(u[:,lp+1,:]),ucon)
        #plt.hold(True)
        cs = plt.contour(xx,z,np.squeeze(u[:,lp+1,:]),levels=[0],colors='k',linewidths=2)
        plt.colorbar(cm)
        plt.xlabel('Storm-relative u WE section')
        plt.title(r'$V_m=$' + repr(vm) + r' $r_m=$' + repr(rm*1e-3) + r' $b=$' + repr(b) + \
                  ' lat=' + repr(lat) + ' K=' + repr(K) + ' C=' + repr(C) + \
                  r' $V_t=$' + repr(Ut))
        plt.subplot(222)
        plt.cla()
        cm = plt.contourf(xx,z,np.squeeze(u[:,:,lp+1]),levels=ucon)
        #plt.hold(True)
        cs = plt.contour(xx,z,np.squeeze(u[:,:,lp+1]),levels=[0],colors='k', linewidths=2)
        plt.colorbar(cm)
        plt.xlabel('Storm-relative u SN section')
        plt.subplot(223)
        plt.cla()
        cm = plt.contourf(xx,z,np.squeeze(v[:,lp+1,:]),vcon)
        #plt.hold(True)
        cs = plt.contour(xx,z,np.squeeze(v[:,lp+1,:]),levels=[0],colors='k')
        plt.colorbar(cm)
        plt.xlabel('Storm-relative v WE section')
        plt.subplot(224)
        plt.cla()
        cm = plt.contourf(xx,z,np.squeeze(v[:,:,lp+1]),vcon)
        #plt.hold(True)
        cs = plt.contour(xx,z,np.squeeze(v[:,:,lp+1]),levels=[0],colors='k', linewidths=2)
        plt.colorbar(cm)
        plt.xlabel('Storm-relative v SN section')
        plt.tight_layout()
        plt.savefig('xsection.png')

        ucon = np.arange(-100, 101, 5)
        vcon = np.arange(-100, 101, 5)
        plt.figure(7)
        plt.clf()
        plt.subplot(221)
        plt.cla()
        lp = 60
        cm = plt.contourf(xx,z,np.squeeze(ux[:,lp+1,:]),ucon)
        #plt.hold(True)
        cs = plt.contour(xx,z,np.squeeze(ux[:,lp+1,:]),levels=[0],colors='k',linewidths=2)
        plt.colorbar(cm)
        plt.xlabel('Earth-relative u WE section')
        plt.title(r'$V_m=$' + repr(vm) + r' $r_m=$' + repr(rm*1e-3) + r' $b=$' + repr(b) + \
                  ' lat=' + repr(lat) + ' K=' + repr(K) + ' C=' + repr(C) + \
                  r' $V_t=$' + repr(Ut))
        plt.subplot(222)
        plt.cla()
        cm = plt.contourf(xx,z,np.squeeze(ux[:,:,lp+1]),levels=ucon)
        #plt.hold(True)
        cs = plt.contour(xx,z,np.squeeze(ux[:,:,lp+1]),levels=[0],colors='k', linewidths=2)
        plt.colorbar(cm)
        plt.xlabel('Earth-relative u SN section')
        plt.subplot(223)
        plt.cla()
        cm = plt.contourf(xx,z,np.squeeze(vx[:,lp+1,:]),vcon)
        #plt.hold(True)
        cs = plt.contour(xx,z,np.squeeze(vx[:,lp+1,:]),levels=[0],colors='k')
        plt.colorbar(cm)
        plt.xlabel('Earth-relative v WE section')
        plt.subplot(224)
        plt.cla()
        cm = plt.contourf(xx,z,np.squeeze(vx[:,:,lp+1]),vcon)
        #plt.hold(True)
        cs = plt.contour(xx,z,np.squeeze(vx[:,:,lp+1]),levels=[0],colors='k', linewidths=2)
        plt.colorbar(cm)
        plt.xlabel('Earth-relative v SN section')
        plt.tight_layout()
        plt.savefig('xsection_fixed.png')

        self.rm = rm
        self.vm = vm
        self.b = b
        self.lat = lat
        self.rho = rho
        self.dp = dp
        self.pe = pe
        self.f = f
        self.Vt = Vt
        self.V = V
        self.Z = Z
        self.us = us
        self.vs = vs
        self.usf = usf
        self.vsf = vsf
        self.Uf = Uf
        self.Vf = Vf
        self.swf1 = swf1
        self.swf2 = swf2
        self.swf3 = swf3
        self.swf4 = swf4
        self.u = u
        self.v = v


