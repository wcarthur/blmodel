from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cmath
sns.set_style('ticks', {'image.cmap':'coolwarm'})
sns.set_context('poster')

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
        thetaFm = 0#3*np.pi/4.0
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
        if dp < 4000.:
            Umod = Ut* (1. - (4000. - dp)/4000.)
        else:
            Umod = Ut
        

        for i in range(rmrbs[0]):  # needed to cope with Python OverflowError
            for j in range(rmrbs[1]):
                try:
                    emrmrb[i,j] = np.exp(-rmrb[i,j])
                except OverflowError:
                    pass

        V = np.sqrt(b*dp/rho * emrmrb * rmrb + (0.5*f*r)**2) - 0.5*abs(f)*r

        if (Ut > 0) and (np.abs(V).max()/Ut < 5):
            Umod = Ut * (1. - (5. - np.abs(V).max()/Ut)/5.)
        else:
            Umod = Ut
        Vt = np.ones(rmrbs, float) * Umod
        #Vt = Umod
        core = np.where(r >= 2.*rm)
        Vt[core] = Umod*np.exp(-((r[core]/(2.*rm)) - 1.)**2.)
    
        Z = -abs(f)  + np.sqrt(4*b*dp/rho * emrmrb * rmrb + (f*r)**2) \
            * (b*dp/rho * rmrb * (2 + b*(rmrb - 1)) * emrmrb + (f*r)**2) \
            / (r * (4*b*dp/rho * emrmrb * rmrb + (f*r)**2))

        if 0:
            # Quadratic core (NB."if" part of tree not yet Python-ised)
            Vm = V.max()
            rs = r[np.where(V==Vm)]
            rs = rs[0]
            rmrsb = (rm/rs)**b
            Vs = np.sqrt(b*dp/rho * np.exp(-rmrsb) * rmrsb + (0.5*f*rs)**2) - 0.5*abs(f)*rs
            icore = np.where(r<rs)
            V[icore] = Vs * (r[icore]/rs) * (2 - (r[icore]/rs))
            Z[icore] = Vs/rs * (4 - 3*r[icore]/rs)
        else:
            # Fit cubic at rm, matching derivatives
            E = np.exp(1)
            Vm = (np.sqrt(b*dp/(rho*E) + (0.5*f*rm)**2) - 0.5*abs(f)*rm)
            dVm = (-f/2 + (E*(f**2)*rm*np.sqrt((4*b*dp/rho)/E + (f*rm)**2))/ \
                          (2*(4*b*dp/rho + E*(f*rm)**2)))
            d2Vm = (b*dp*(-4*b**3*dp/rho - (-2 + b**2)*E*(f*rm)**2)) / \
                    (E*rho*np.sqrt((4*b*dp)/(E*rho) + (f*rm)**2) * \
                      (4*b*dp*rm**2/rho + E*(f*rm**2)**2))
            aa = (d2Vm/2 - (dVm - Vm/rm)/rm) / rm
            bb = (d2Vm - 6*aa*rm) / 2
            cc = dVm - 3*aa*rm**2 - 2*bb*rm
            icore = np.nonzero(np.ravel(r<rm))
            ## xx = r * (r * (r*aa + bb) + cc)
            for ind in icore:
                V.flat[ind] = r.flat[ind] * \
                                 (r.flat[ind] * (r.flat[ind]*aa + bb) + cc)
                Z.flat[ind] = r.flat[ind] * (r.flat[ind] * 4*aa + 3*bb) + 2*cc
                pm = pe - dp*(1 - np.exp(-1))
#                p.flat[ind] = pm + (r.flat[ind]**6 - rm**6) * (aa**2)/6 \
#                                 + (r.flat[ind]**5 - rm**5) * (2*aa*bb)/5 \
#                                 + (r.flat[ind]**4 - rm**4) * (aa*(2*cc+f)+bb**2)/4 \
#                                 + (r.flat[ind]**3 - rm**3) * (bb*(2*cc+f))/3 \
#                                 + (r.flat[ind]**2 - rm**2) * (cc*(cc+f))/2
        V = V*np.sign(f)
        Z = Z*np.sign(f)

        print V.min(), V.max()

        Vx = -V*y/r  # Cartesian winds (??? divide by y ???)
        Vy =  V*x/r

        I2 = (f + 2*V/r)*(f + Z)  # Inertial stability squared

        K = 70    # Diffusivity
        C = 0.002 # Drag coefficient

        # Calculate alpha, beta, gamma, chi, eta and psi in Kepert (2001). 
        # The III's are to get the solution in the case where V/r > I.
        al = (2 * V / r + f) / (2*K)
        be = (f + Z) / (2*K)
        gam = np.abs(V / (2*K*r))

#        III = np.nonzero(np.ravel(gam > np.sqrt(al*be)))
#        fixup = nonzero(ravel(isNaN(al) | isnan(be) | isnan(gam)))
        chi = np.abs((C/K)*V / np.sqrt(np.sqrt(al*be)))
#        chi[fixup] = np.nan
        eta = np.abs((C/K)*V / np.sqrt(np.sqrt(al*be) + gam))
#        eta[fixup] = np.nan
        psi = np.abs((C/K)*V / np.sqrt(np.abs(np.sqrt(al*be) - gam)))
#        psi[fixup] = nan
        albe = np.sqrt(al/be)

        # Calculate A_k's ... p and m refer to k = +1 and -1.
        i = cmath.sqrt(-1)
        A0 =  -chi*V*(1 + i*(1 + chi)) / (2*chi**2 + 3*chi + 2)
#        A0[fixup] = np.nan
        u0s = albe * A0.real   # Symmetric surface wind component
        v0s =        A0.imag

        Am = -((1 + (1+i)*eta)/albe \
             + (2 + (1+i)*eta))*psi*Vt \
             / ((2 + 2*i)*(1 + eta*psi) + 3*psi + 3*i*eta) 
        AmIII = -((1 + (1+i)*eta)/albe \
                + (2 + (1+i)*eta))*psi*Vt \
                / (2 - 2*i + 3*(psi + eta) + (2 + 2*i)*eta*psi)
#        Am[III] = AmIII[III]
#        Am[fixup] = np.nan
        ums = albe * (Am * np.exp(-i*lam)).real  # First asymmetric surface component
        vms = (Am * np.exp(-i*lam)).imag
        Ap = -((1 + (1+i)*psi)/albe \
             - (2 + (1+i)*psi))*eta*Vt \
             / ((2 + 2*i)*(1 + eta*psi) + 3*eta + 3*i*psi) 
        ApIII = -( (1 + (1-i)*psi)/albe \
                - (2 + (1-i)*psi))*eta*Vt \
                / (2 + 2*i + 3*(eta + psi) + (2 - 2*i)*eta*psi)
#        Ap[III] = ApIII[III]
#        Ap[fixup] = np.nan
        ups = albe * (Ap * np.exp(i*lam)).real # Second asymmetric surface component
        vps = (Ap * np.exp(i*lam)).imag
        
        # Total surface wind in (moving coordinate system)
        us =     np.sign(f) * (u0s + ups + ums)
        vs = V + v0s + vps + vms

        # Total surface wind in stationary coordinate system (f for fixed)
        usf = us + Vt*np.cos(lam - thetaFm)
        vsf = vs - Vt*np.sin(lam - thetaFm)
        Uf =   + Vt*np.cos(lam)
        Vf = V - Vt*np.sin(lam)
        phi = np.arctan2(usf, vsf)
        Ux = np.sqrt(usf ** 2. + vsf ** 2.) * np.sin(phi - lam)
        Vy = np.sqrt(usf ** 2. + vsf ** 2.) * np.cos(phi - lam)
        #Figure 1 here
        fig, axes = plt.subplots(2, 2,subplot_kw={'aspect':'equal'},figsize=(18,18))
        ax = axes.flatten()
        #plt.clf()
        #plt.subplot(221, aspect='equal')
        #plt.cla()
        ax[0].hold(True)
        #set(plt.gca(),'DataAspectRatio',[1,1,1])
        ax[0].set_xlim([-150,150])
        ax[0].set_ylim([-150,150])
        levels = np.arange(-20, 21, 2)
        cm = ax[0].contourf(xkm, ykm, usf, np.arange(-50., 51, 2))
        cs = ax[0].contour(xkm, ykm, usf, np.arange(-50, 51, 2), colors='k')
        ax[0].clabel(cs, fontsize='x-small', fmt='%1.2f') #, range(-20,11,4))
        plt.colorbar(cm, ax=ax[0])
        ax[0].plot(cx,cy,'w')
        ax[0].set_xlabel('Storm-relative surface u')

        #plt.subplot(222,  aspect='equal')
        #plt.cla()
        #plt.hold(True)
        ax[1].set_xlim([-150,150])
        ax[1].set_ylim([-150,150])
        cm = ax[1].contourf(xkm,ykm,vsf,range(-50,51,5))
        cs = ax[1].contour(xkm,ykm,vsf,range(-50,51,5),colors='k')
        ax[1].clabel(cs, fontsize='x0small', fmt='%1.2f') 
        plt.colorbar(cm, ax=ax[1])
        ax[1].plot(cx,cy,'w')
        ax[1].set_xlabel('Storm-relative surface v')

        #plt.subplot(223, aspect='equal')
        #plt.cla()
        #plt.hold(True)
        ax[2].set_xlim([-150,150])
        ax[2].set_ylim([-150,150])
        cm = ax[2].contourf(xkm,ykm,Ux,range(-50, 51, 5))
        cs = ax[2].contour(xkm,ykm,Ux,range(-50, 51, 5),colors='k')
        ax[2].clabel(cs, fontsize='x-small', fmt='%1.2f') #,range(-20,11,4))
        plt.colorbar(cm, ax=ax[2])
        ax[2].plot(cx,cy,'w')
        ax[2].set_xlabel('Earth-relative surface u (cartesian)')
        
        #plt.subplot(224, aspect='equal')
        #plt.cla()
        #plt.hold(True)
        ax[3].set_xlim([-150,150])
        ax[3].set_ylim([-150,150])
        cm = ax[3].contourf(xkm,ykm,Vy,range(-50, 51, 5))
        cs = ax[3].contour(xkm,ykm,Vy,range(-50, 51, 5),colors='k')
        ax[3].clabel(cs, fontsize='x-small', fmt='%1.2f') #,h,range(0,51,10)
        plt.colorbar(cm, ax=ax[3])
        ax[3].plot(cx,cy,'w')
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
        plt.hold(True)
        plt.xlim([-150, 150])
        plt.ylim([-150, 150])
        mag = np.sqrt(Ux**2 + Vy**2) # Magnitude of the total surface wind
        cm = plt.contourf(xkm, ykm, mag, levels=np.arange(0, 101, 5))
        cs = plt.contour(xkm, ykm, mag, levels=np.arange(0, 101, 5), colors='k')
        plt.barbs(xkm[::5, ::5], ykm[::5, ::5], Ux[::5, ::5], Vy[::5, ::5],flip_barb=True)
        plt.clabel(cs)
        plt.colorbar(cm)
        plt.plot(cx, cy, 'w')
        plt.xlabel('Earth-relative total surface wind speed')
        plt.tight_layout()
        plt.savefig('sfc_total_wind.png')

        #Figure 2 here
        #figure(2)
        plt.clf()
        plt.subplot(221, aspect='equal')
        plt.cla()
        plt.hold(True)
        plt.xlim([-150,150])
        plt.ylim([-150,150])
        cm = plt.contourf(xkm,ykm,swf1,levels=np.arange(0.5,1.25,0.05))
        cs = plt.contour(xkm,ykm,swf1,np.arange(0.5,1.25,0.05),colors='k')
        plt.clabel(cs)
        plt.colorbar(cm)
        plt.plot(cx,cy,'w')
        plt.xlabel('Storm-relative azimuthal swrf')
        plt.title('V_m=' + repr(vm) + ' r_m=' + repr(rm*1e-3) + ' b=' + repr(b) + \
              ' lat=' + repr(lat) + ' K=' + repr(K) + ' C=' + repr(C))
        plt.subplot(222, aspect='equal')
        plt.cla()
        plt.hold(True)

        plt.xlim([-150,150])
        plt.ylim([-150,150])
        cm = plt.contourf(xkm,ykm,swf2,levels=np.arange(0.5,1.25,0.05))
        cs = plt.contour(xkm,ykm,swf2,np.arange(0.5,1.25,0.05), colors='k')
        plt.clabel(cs)
        plt.colorbar(cm)
        plt.plot(cx,cy,'w')
        plt.xlabel('Storm-relative total swrf')
        plt.subplot(223, aspect='equal')
        plt.cla()
        plt.hold(True)
        plt.xlim([-150,150])
        plt.ylim([-150,150])
        cm = plt.contourf(xkm,ykm,swf3,levels=np.arange(0.5,1.25,0.05))
        cs = plt.contour(xkm,ykm,swf3,np.arange(0.5,1.25,0.05), colors='k')
        plt.clabel(cs)
        plt.colorbar(cm) 
        plt.plot(cx,cy,'w')
        plt.xlabel('Earth-relative azimuthal swrf')
        plt.subplot(224, aspect='equal')
        plt.cla()
        plt.hold(True)
        plt.xlim([-150,150])
        plt.ylim([-150,150])
        cm = plt.contourf(xkm,ykm,swf4,levels=np.arange(0.5,1.25,0.05))
        cs = plt.contour(xkm,ykm,swf4,levels=np.arange(0.5,1.25,0.05), colors='k')
        plt.clabel(cs)
        plt.colorbar(cm) 
        plt.plot(cx,cy,'w')
        plt.xlabel('Earth-relative total swrf')
        plt.tight_layout()
        plt.savefig('swrf.png')


        # Coefficients p_k for k = 0, 1, -1.
        p0 = -(1 + i) * (al * be)**(1./4.)
        pp = -(1 + i) * np.sqrt(np.sqrt(al * be) + gam)
        pm = -(1 + i) * np.sqrt(np.abs(np.sqrt(al * be) - gam))
#        pm[III] = -i * pm[III]

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
            wm = Am * np.exp(z[kk]*pm - i*lam)
#            wm[fixup] = np.nan
            w0 = A0 * np.exp(z[kk]*p0)
#            w0[fixup] = np.nan
            wp = Ap * np.exp(z[kk]*pp + i*lam)
#            wp[fixup] = np.nan

            u[kk] = albe*(wm + w0 + wp).real
            v[kk] = V + (wm + w0 + wp).imag
            ua[kk] = albe*(wm + wp).real
            va[kk] = (wm + wp).imag

#            ux[kk] = (squeeze(u[kk])*x - squeeze(v[kk])*y) / r
#            vx[kk] = (squeeze(v[kk])*x + squeeze(u[kk])*y) / r
#            uxa[kk] = (squeeze(ua[kk])*x - squeeze(va[kk])*y) / r
#            vxa[kk] = (squeeze(va[kk])*x + squeeze(ua[kk])*y) / r

            ux[kk] = (u[kk]*x - v[kk]*y) / r
            vx[kk] = (v[kk]*x + u[kk]*y) / r
            uxa[kk] = (ua[kk]*x - va[kk]*y) / r
            vxa[kk] = (va[kk]*x + ua[kk]*y) / r

        ucon = np.arange(-20, 21, 2)
        vcon = np.arange(0,51,5)
        # Figure 3 here
        plt.figure(3)
        plt.clf()
        plt.subplot(221)
        plt.cla()
        lp = 60
        cm = plt.contourf(xx,z,np.squeeze(u[:,lp+1,:]),ucon)
        plt.hold(True)
        cs = plt.contour(xx,z,np.squeeze(u[:,lp+1,:]),levels=[0,0],colors='k',linewidths=2)
        plt.colorbar(cm)
        plt.xlabel('Storm-relative u WE section')
        plt.title(r'$V_m=$' + repr(vm) + r' $r_m=$' + repr(rm*1e-3) + r' $b=$' + repr(b) + \
                  ' lat=' + repr(lat) + ' K=' + repr(K) + ' C=' + repr(C) + \
                  r' $V_t=$' + repr(Ut))
        plt.subplot(222)
        plt.cla()
        cm = plt.contourf(xx,z,np.squeeze(u[:,:,lp+1]),levels=ucon)
        plt.hold(True)
        cs = plt.contour(xx,z,np.squeeze(u[:,:,lp+1]),levels=[0,0],colors='k', linewidths=2)
        plt.colorbar(cm)
        plt.xlabel('Storm-relative u SN section')
        plt.subplot(223)
        plt.cla()
        cm = plt.contourf(xx,z,np.squeeze(v[:,lp+1,:]),vcon)
        plt.hold(True)
        cs = plt.contour(xx,z,np.squeeze(v[:,lp+1,:]),levels=[0,0],colors='k')
        plt.colorbar(cm)
        plt.xlabel('Storm-relative v WE section')
        plt.subplot(224)
        plt.cla()
        cm = plt.contourf(xx,z,np.squeeze(v[:,:,lp+1]),vcon)
        plt.hold(True)
        cs = plt.contour(xx,z,np.squeeze(v[:,:,lp+1]),levels=[0,0],colors='k', linewidths=2)
        plt.colorbar(cm)
        plt.xlabel('Storm-relative v SN section')        
        plt.tight_layout()
        plt.savefig('xsection.png')

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


