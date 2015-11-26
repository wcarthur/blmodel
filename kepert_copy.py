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

    def __init__(x, y, lat, rm, vm, b, dp, pe, rho, Ut):

        from math import sqrt, sin, cos, atan2, pi, exp
        from pylab import figure

        f = 2*7.292e-5*sin(lat*pi/180)  # Coriolis parameter

        lam = atan2(y,x)
        r = sqrt(x**2 + y**2)
        r[where(r==0)] = 1e-30  # to avoid divide by zero errors

        xkm = x*1e-3
        ykm = y*1e-3
        xx = xkm[0,]
        yy = ykm[:,0]
        cx = rm*1e-3*sin(range(0,6.29,0.01))  # Circle for rmw plots.
        cy = rm*1e-3*cos(range(0,6.29,0.01))

        # Define parametric gradient wind (V) and vorticity (Z) profiles
        # using Holland (1980) bogus with core modified to quadratic.
        rmrb = (rm/r)**b
        V = sqrt(b*dp/rho * exp(-rmrb) * rmrb + (0.5*f*r)**2) - 0.5*f*r
        Z = -f + sqrt(4*b*dp/rho * exp(-rmrb) * rmrb + (f*r)**2) \
            * (b*dp/rho * rmrb * (2 + b*(rmrb - 1)) * exp(-rmrb) + (f*r)**2) \
            / (r * (4*b*dp/rho * exp(-rmrb) * rmrb + (f*r)**2))

        if 0:
            # Quadratic core
            Vm = V.max()
            rs = r[where(V==Vm)]
            rs = rs[0]
            rmrsb = (rm/rs)**b
            Vs = sqrt(b*dp/rho * exp(-rmrsb) * rmrsb + (0.5*f*rs)**2) - 0.5*f*rs
            icore = where(r<rs)
            V[icore] = Vs * (r[icore]/rs) * (2 - (r[icore]/rs))
            Z[icore] = Vs/rs * (4 - 3*r[icore]/rs)
        else:
            # Fit cubic at rm, matching derivatives
            E = exp(1)
            Vm = sqrt(b*dp/(rho*E) + (0.5*f*rm)**2) - 0.5*f*rm
            dVm = -f/2 + (E*(f**2)*rm*sqrt((4*b*dp/rho)/E + (f*rm)**2))/ \
                          (2*(4*b*dp/rho + E*(f*rm)**2))
            d2Vm = (b*dp*(-4*b**3*dp/rho - (-2 + b**2)*E*(f*rm)**2)) / \
                    (E*rho*sqrt((4*b*dp)/(E*rho) + (f*rm)**2) * \
                      (4*b*dp*rm**2/rho + E*(f*rm**2)**2))
            aa = (d2Vm/2 - (dVm - Vm/rm)/rm) / rm
            bb = (d2Vm - 6*aa*rm) / 2
            cc = dVm - 3*aa*rm**2 - 2*bb*rm
            icore = where(r<rm)
            # xx = r * (r * (r*aa + bb) + cc)
            V[icore] = r[icore] * (r[icore] * (r[icore]*aa + bb) + cc)
            Z[icore] = r[icore] * (r[icore] * 4*aa + 3*bb) + 2*cc
            pm = pe - dp*(1 - exp(-1))
            p[icore] = pm + (r[icore]**6 - rm**6) * (aa**2)/6 \
                          + (r[icore]**5 - rm**5) * (2*aa*bb)/5 \
                          + (r[icore]**4 - rm**4) * (aa*(2*cc+f)+bb**2)/4 \
                          + (r[icore]**3 - rm**3) * (bb*(2*cc+f))/3 \
                          + (r[icore]**2 - rm**2) * (cc*(cc+f))/2

        Vx = -V*y/r  # Cartesian winds
        Vy =  V*x/r

        I2 = (f + 2*V/r)*(f + Z)  # Inertial stability squared

        K = 50    # Diffusivity
        C = 0.002 # Drag coefficient

        # Calculate alpha, beta, gamma, chi, eta and psi in Kepert (2001). 
        # The III's are to get the solution in the case where V/r > I.
        al = (2 * V / r + f) / (2*K)
        be = (f + Z) / (2*K)
        gam = V / (2*K*r)
        III = where(gam > sqrt(al*be))
        fixup = where(isnan(al) | isnan(be) | isnan(gam))
        chi = (C/K)*V / sqrt(sqrt(al*be))
        chi[fixup] = nan
        eta = (C/K)*V / sqrt(sqrt(al*be) + gam)
        eta[fixup] = nan
        psi = (C/K)*V / sqrt(abs(sqrt(al*be) - gam))
        psi[fixup] = nan
        albe = sqrt(al/be)

        # Calculate A_k's ... p and m refer to k = +1 and -1.
        i = sqrt(-1)
        A0 =  -chi*V*(1 + i*(1 + chi)) / (2*chi**2 + 3*chi + 2)
        A0[fixup] = nan
        u0s = albe * A0.real   # Symmetric surface wind component
        v0s =        A0.imag
        Am = -((1 + (1+i)*eta)/albe \
             + (2 + (1+i)*eta))*psi*Ut \
             / ((2 + 2*i)*(1 + eta*psi) + 3*psi + 3*i*eta) 
        AmIII = -((1 + (1+i)*eta)/albe \
                + (2 + (1+i)*eta))*psi*Ut \
                / (2 - 2*i + 3*(psi + eta) + (2 + 2*i)*eta*psi)
        Am[III] = AmIII[III]
        Am[fixup] = nan
        ums = albe * (Am * exp(-i*lam)).real  # First asymmetric surface component
        vms = (Am * exp(-i*lam)).imag
        Ap = -((1 + (1+i)*psi)/albe \
             - (2 + (1+i)*psi))*eta*Ut \
             / ((2 + 2*i)*(1 + eta*psi) + 3*eta + 3*i*psi) 
        ApIII = -( (1 + (1-i)*psi)/albe \
                - (2 + (1-i)*psi))*eta*Ut \
                / (2 + 2*i + 3*(eta + psi) + (2 - 2*i)*eta*psi)
        Ap[III] = ApIII[III]
        Ap[fixup] = nan
        ups = albe * (Ap * exp(i*lam)).real # Second asymmetric surface component
        vps = (Ap * exp(i*lam)).imag

        # Total surface wind in (moving coordinate system)
        us =     u0s + ups + ums
        vs = V + v0s + vps + vms

        # Total surface wind in stationary coordinate system (f for fixed)
        usf = us + Ut*cos(lam)
        vsf = vs - Ut*sin(lam)
        Uf =   + Ut*cos(lam)
        Vf = V - Ut*sin(lam)

        figure(1)
        clf()
        subplot(221)
        cla()
        hold(True)
        set(gca(),'DataAspectRatio',[1,1,1])
        xlim([-150,150])
        ylim([-150,150])
        [cs,h] = contourf(xkm,ykm,us,range(-20,11,2),'k-')
        set(h,'LineStyle','none')
        [cs,h] = contour(xkm,ykm,us,range(-20,11,2),'k-')
        clabel(cs,h,range(-20,11,4))
        plot(cx,cy,'w')
        xlabel('Storm-relative surface u')
        title('V_m=' + repr(vm) + ' r_m=' + repr(rm*1e-3) + ' b=' + repr(b) + \
              ' lat=' + repr(lat) + ' K=' + repr(K) + ' C=' + repr(C))
        subplot(222)
        cla()
        hold(True)
        set(gca(),'DataAspectRatio',[1,1,1])
        xlim([-150,150])
        ylim([-150,150])
        [cs,h] = contourf(xkm,ykm,vs,range(0,51,5),'k-')
        set(h,'LineStyle','none')
        [cs,h] = contour(xkm,ykm,vs,range(0,51,5),'k-')
        clabel(cs,h,range(0,51,10))
        plot(cx,cy,'w')
        xlabel('Storm-relative surface v')
        subplot(223)
        cla()
        hold(True)
        set(gca(),'DataAspectRatio',[1,1,1])
        xlim([-150,150])
        ylim([-150,150])
        [cs,h] = contourf(xkm,ykm,usf,range(-20,11,2),'k-')
        set(h,'LineStyle','none')
        [cs,h] = contour(xkm,ykm,usf,range(-20,11,2),'k-')
        clabel(cs,h,range(-20,11,4))
        plot(cx,cy,'w')
        xlabel('Earth-relative surface u')
        subplot(224)
        cla()
        hold(True)
        set(gca(),'DataAspectRatio',[1,1,1])
        xlim([-150,150])
        ylim([-150,150])
        [cs,h] = contourf(xkm,ykm,vsf,range(0,51,5),'k-')
        set(h,'LineStyle','none')
        [cs,h] = contour(xkm,ykm,vsf,range(0,51,5),'k-')
        clabel(cs,h,range(0,51,10))
        plot(cx,cy,'w')
        xlabel('Earth-relative surface v')

        # Four possible surface wind factors \ depending on whether you 
        # use the total or azimuthal wind, and in moving or fixed coordinates.
        swf1 = vs / V
        swf2 = sqrt(us**2 + vs**2) / V
        swf3 = vsf / Vf
        swf4 = sqrt(usf**2 + vsf**2) / sqrt(Uf**2 + Vf**2)

        #figure(2)
        clf()
        subplot(221)
        cla()
        hold(True)
        set(gca(),'DataAspectRatio',[1,1,1])
        xlim([-150,150])
        ylim([-150,150])
        [cs,h] = contourf(xkm,ykm,swf1,range(0.5,1.25,0.05))
        set(h,'LineStyle','none')
        [cs,h] = contourf(xkm,ykm,swf1,range(0.5,1.25,0.05))
        clabel(cs,h,range(0.5,1.25,0.1))
        plot(cx,cy,'w')
        xlabel('Storm-relative azimuthal swrf')
        title('V_m=' + repr(vm) + ' r_m=' + repr(rm*1e-3) + ' b=' + repr(b) + \
              ' lat=' + repr(lat) + ' K=' + repr(K) + ' C=' + repr(C))
        subplot(222)
        cla()
        hold(True)
        set(gca(),'DataAspectRatio',[1,1,1])
        xlim([-150,150])
        ylim([-150,150])
        [cs,h] = contourf(xkm,ykm,swf2,range(0.5,1.25,0.05))
        set(h,'LineStyle','none')
        [cs,h] = contourf(xkm,ykm,swf2,range(0.5,1.25,0.05))
        clabel(cs,h,range(0.5,1.25,0.1))
        plot(cx,cy,'w')
        xlabel('Storm-relative total swrf')
        subplot(223)
        cla()
        hold(True)
        set(gca(),'DataAspectRatio',[1,1,1])
        xlim([-150,150])
        ylim([-150,150])
        [cs,h] = contourf(xkm,ykm,swf3,range(0.5,1.25,0.05))
        set(h,'LineStyle','none')
        [cs,h] = contourf(xkm,ykm,swf3,range(0.5,1.25,0.05))
        clabel(cs,h,range(0.5,1.25,0.1))
        plot(cx,cy,'w')
        xlabel('Earth-relative azimuthal swrf')
        subplot(224)
        cla()
        hold(True)
        set(gca(),'DataAspectRatio',[1,1,1])
        xlim([-150,150])
        ylim([-150,150])
        [cs,h] = contourf(xkm,ykm,swf4,range(0.5,1.25,0.05))
        set(h,'LineStyle','none')
        [cs,h] = contourf(xkm,ykm,swf4,range(0.5,1.25,0.05))
        clabel(cs,h,range(0.5,1.25,0.1))
        plot(cx,cy,'w')
        xlabel('Earth-relative total swrf')

        # Coefficients p_k for k = 0, 1, -1.
        p0 = -(1 + i) * (al * be)**(1/4)
        pp = -(1 + i) * sqrt(sqrt(al * be) + gam)
        pm = -(1 + i) * sqrt(abs(sqrt(al * be) - gam))
        pm[III] = -i * pm[III]

        # Set up a 3-d grid and calculate winds 
        nz = 20
        z = range(1,nz+1)*100

        [ny,nx] = x.shape
        nz = max(z.size)
        u = zeros((nz,ny,nx),Float)
        v = zeros((nz,ny,nx),Float)
        ux = zeros((nz,ny,nx),Float)  # x for cartesian wind
        vx = zeros((nz,ny,nx),Float)
        uxa = zeros((nz,ny,nx),Float)  # Asymmetric parts have trailing a
        vxa = zeros((nz,ny,nx),Float)

        for kk in range(1,nz+1):
            # These w's are really omega's!
            wm = Am * exp(z[kk]*pm - i*lam)
            wm[fixup] = nan
            w0 = A0 * exp(z[kk]*p0)
            w0[fixup] = nan
            wp = Ap * exp(z[kk]*pp + i*lam)
            wp[fixup] = nan

            u[kk] = albe*(wm + w0 + wp).real
            v[kk] = V + (wm + w0 + wp).imag
            ua[kk] = albe*(wm + wp).real
            va[kk] = (wm + wp).imag

            ux[kk] = (squeeze(u[kk])*x - squeeze(v[kk])*y) / r
            vx[kk] = (squeeze(v[kk])*x + squeeze(u[kk])*y) / r
            uxa[kk] = (squeeze(ua[kk])*x - squeeze(va[kk])*y) / r
            vxa[kk] = (squeeze(va[kk])*x + squeeze(ua[kk])*y) / r

        ucon = arange(-20,11,2)
        vcon = arange(0,41,5)

        figure(3)
        clf()
        subplot(221)
        cla()
        contourf(xx,z,squeeze(u[:,lp+1,:]),ucon)
        hold(True)
        [cs,h] = contour(xx,z,squeeze(u[:,lp+1,:]),[0,0],'k')
        set(h,'LineWidth',2)
        xlabel('Storm-relative u WE section')
        title('V_m=' + repr(vm) + ' r_m=' + repr(rm*1e-3) + ' b=' + repr(b) + \
              ' lat=' + repr(lat) + ' K=' + repr(K) + ' C=' + repr(C) + \
              ' U_t=' + repr(Ut))
        subplot(222)
        cla()
        contourf(xx,z,squeeze(u[:,:,lp+1]),ucon)
        hold(True)
        [cs,h] = contour(xx,z,squeeze(u[:,:,lp+1]),[0,0],'k')
        set(h,'LineWidth',2)
        xlabel('Storm-relative u SN section')
        subplot(223)
        cla()
        contourf(xx,z,squeeze(v[:,lp+1,:]),vcon)
        hold(True)
        [cs,h] = contour(xx,z,squeeze(v[:,lp+1,:]),[0,0],'k')
        set(h,'LineWidth',2)
        xlabel('Storm-relative v WE section')
        subplot(224)
        cla()
        contourf(xx,z,squeeze(v[:,:,lp+1]),vcon)
        hold(True)
        [cs,h] = contour(xx,z,squeeze(v[:,:,lp+1]),[0,0],'k')
        set(h,'LineWidth',2)
        xlabel('Storm-relative v SN section')

        self.rm = rm
        self.vm = vm
        self.b = b
        self.lat = lat
        self.rho = rho
        self.dp = dp
        self.pe = pe
        self.f = f
        self.Ut = Ut

