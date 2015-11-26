import unittest

from kepert import Kepert_linear_assym
from numpy import *

class Test_Kepert(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_kepert_inputs(self):

        x = resize(arange(-150e3,151e3,5e3), (61, 61))
        y = transpose(x)

        p = Kepert_linear_assym(x, y, lat=15, rm=4e4, vm=40.0, b=1.3, \
                                dp=3680.13292308, pe=1e5, rho=1.1, Ut=-5)

        assert p.rm == 4e4
        assert p.vm == 40.0
        assert p.b == 1.3
        assert p.lat == 15
        assert p.rho == 1.1
        assert p.dp == 3680.13292308
        assert p.pe == 1e5
        assert p.Ut == -5

    def test_kepert_V(self):

        from numpy import ones, float as Float
        from string import split

        f = open('test_data/V.csv', 'r')
        l = ones((61,61), Float)

        for i in range(61):
            tls = []
            tlf = []
            fline = f.readline()
            tls = split(fline, ',')
            for j in range(61):
                l[i,j] = float(tls[j])

        f.close()

        x = resize(arange(-150e3,151e3,5e3), (61, 61))
        y = transpose(x)

        p = Kepert_linear_assym(x, y, lat=15, rm=4e4, vm=40.0, b=1.3, \
                                dp=3680.13292308, pe=1e5, rho=1.1, Ut=-5)

        assert allclose(p.V, l, rtol=1.e-4, atol=1.e-8)

    def test_kepert_Z(self):

        from numpy import ones, float as Float
        from string import split

        f = open('test_data/Z.csv', 'r')
        l = ones((61,61), Float)

        for i in range(61):
            tls = []
            tlf = []
            fline = f.readline()
            tls = split(fline, ',')
            for j in range(61):
                l[i,j] = float(tls[j])

        f.close()

        x = resize(arange(-150e3,151e3,5e3), (61, 61))
        y = transpose(x)

        p = Kepert_linear_assym(x, y, lat=15, rm=4e4, vm=40.0, b=1.3, \
                                dp=3680.13292308, pe=1e5, rho=1.1, Ut=-5)

        assert allclose(p.Z, l, rtol=1.e-4, atol=1.e-8)

    def test_kepert_us(self):

        from numpy import ones, float as Float
        from string import split

        f = open('test_data/us.csv', 'r')
        l = ones((61,61), Float)

        for i in range(61):
            tls = []
            tlf = []
            fline = f.readline()
            tls = split(fline, ',')
            for j in range(61):
                l[i,j] = float(tls[j])

        f.close()

        x = resize(arange(-150e3,151e3,5e3), (61, 61))
        y = transpose(x)

        p = Kepert_linear_assym(x, y, lat=15, rm=4e4, vm=40.0, b=1.3, \
                                dp=3680.13292308, pe=1e5, rho=1.1, Ut=-5)

        assert allclose(p.us, l, rtol=1.e-4, atol=1.e-8)

    def test_kepert_vs(self):

        from numpy import ones, float as Float
        from string import split

        f = open('test_data/vs.csv', 'r')
        l = ones((61,61), Float)

        for i in range(61):
            tls = []
            tlf = []
            fline = f.readline()
            tls = split(fline, ',')
            for j in range(61):
                l[i,j] = float(tls[j])

        f.close()

        x = resize(arange(-150e3,151e3,5e3), (61, 61))
        y = transpose(x)

        p = Kepert_linear_assym(x, y, lat=15, rm=4e4, vm=40.0, b=1.3, \
                                dp=3680.13292308, pe=1e5, rho=1.1, Ut=-5)

        assert allclose(p.vs, l, rtol=1.e-4, atol=1.e-8)

    def test_kepert_usf(self):

        from numpy import ones, float as Float
        from string import split

        f = open('test_data/usf.csv', 'r')
        l = ones((61,61), Float)

        for i in range(61):
            tls = []
            tlf = []
            fline = f.readline()
            tls = split(fline, ',')
            for j in range(61):
                l[i,j] = float(tls[j])

        f.close()

        x = resize(arange(-150e3,151e3,5e3), (61, 61))
        y = transpose(x)

        p = Kepert_linear_assym(x, y, lat=15, rm=4e4, vm=40.0, b=1.3, \
                                dp=3680.13292308, pe=1e5, rho=1.1, Ut=-5)

        assert allclose(p.usf, l, rtol=1.e-4, atol=1.e-8)

    def test_kepert_vsf(self):

        from numpy import ones, float as Float
        from string import split

        f = open('test_data/vsf.csv', 'r')
        l = ones((61,61), Float)

        for i in range(61):
            tls = []
            tlf = []
            fline = f.readline()
            tls = split(fline, ',')
            for j in range(61):
                l[i,j] = float(tls[j])

        f.close()

        x = resize(arange(-150e3,151e3,5e3), (61, 61))
        y = transpose(x)

        p = Kepert_linear_assym(x, y, lat=15, rm=4e4, vm=40.0, b=1.3, \
                                dp=3680.13292308, pe=1e5, rho=1.1, Ut=-5)

        assert allclose(p.vsf, l, rtol=1.e-4, atol=1.e-8)

    def test_kepert_Uf(self):

        from numpy import ones, float as Float
        from string import split

        f = open('test_data/Uf.csv', 'r')
        l = ones((61,61), Float)

        for i in range(61):
            tls = []
            tlf = []
            fline = f.readline()
            tls = split(fline, ',')
            for j in range(61):
                l[i,j] = float(tls[j])

        f.close()

        x = resize(arange(-150e3,151e3,5e3), (61, 61))
        y = transpose(x)

        p = Kepert_linear_assym(x, y, lat=15, rm=4e4, vm=40.0, b=1.3, \
                                dp=3680.13292308, pe=1e5, rho=1.1, Ut=-5)

        assert allclose(p.Uf, l, rtol=1.e-4, atol=1.e-8)

    def test_kepert_Vf(self):

        from numpy import ones, float as Float
        from string import split

        f = open('test_data/Vf.csv', 'r')
        l = ones((61,61), Float)

        for i in range(61):
            tls = []
            tlf = []
            fline = f.readline()
            tls = split(fline, ',')
            for j in range(61):
                l[i,j] = float(tls[j])

        f.close()

        x = resize(arange(-150e3,151e3,5e3), (61, 61))
        y = transpose(x)

        p = Kepert_linear_assym(x, y, lat=15, rm=4e4, vm=40.0, b=1.3, \
                                dp=3680.13292308, pe=1e5, rho=1.1, Ut=-5)

        assert allclose(p.Vf, l, rtol=1.e-4, atol=1.e-8)

    def test_kepert_swf1(self):

        from numpy import ones, float as Float
        from string import split

        f = open('test_data/swf1.csv', 'r')
        l = ones((61,61), Float)

        for i in range(61):
            tls = []
            tlf = []
            fline = f.readline()
            tls = split(fline, ',')
            for j in range(61):
                l[i,j] = float(tls[j])

        f.close()

        x = resize(arange(-150e3,151e3,5e3), (61, 61))
        y = transpose(x)

        p = Kepert_linear_assym(x, y, lat=15, rm=4e4, vm=40.0, b=1.3, \
                                dp=3680.13292308, pe=1e5, rho=1.1, Ut=-5)

        assert allclose(p.swf1, l, rtol=1.e-4, atol=1.e-8)

    def test_kepert_swf2(self):

        from numpy import ones, float as Float
        from string import split

        f = open('test_data/swf2.csv', 'r')
        l = ones((61,61), Float)

        for i in range(61):
            tls = []
            tlf = []
            fline = f.readline()
            tls = split(fline, ',')
            for j in range(61):
                l[i,j] = float(tls[j])

        f.close()

        x = resize(arange(-150e3,151e3,5e3), (61, 61))
        y = transpose(x)

        p = Kepert_linear_assym(x, y, lat=15, rm=4e4, vm=40.0, b=1.3, \
                                dp=3680.13292308, pe=1e5, rho=1.1, Ut=-5)

        assert allclose(p.swf2, l, rtol=1.e-4, atol=1.e-8)

    def test_kepert_swf3(self):

        from numpy import ones, float as Float
        from string import split

        f = open('test_data/swf3.csv', 'r')
        l = ones((61,61), Float)

        for i in range(61):
            tls = []
            tlf = []
            fline = f.readline()
            tls = split(fline, ',')
            for j in range(61):
                l[i,j] = float(tls[j])

        f.close()

        x = resize(arange(-150e3,151e3,5e3), (61, 61))
        y = transpose(x)

        p = Kepert_linear_assym(x, y, lat=15, rm=4e4, vm=40.0, b=1.3, \
                                dp=3680.13292308, pe=1e5, rho=1.1, Ut=-5)

        assert allclose(p.swf3, l, rtol=1.e-4, atol=1.e-8)

    def test_kepert_swf4(self):

        from numpy import ones, float as Float
        from string import split

        f = open('test_data/swf4.csv', 'r')
        l = ones((61,61), Float)

        for i in range(61):
            tls = []
            tlf = []
            fline = f.readline()
            tls = split(fline, ',')
            for j in range(61):
                l[i,j] = float(tls[j])

        f.close()

        x = resize(arange(-150e3,151e3,5e3), (61, 61))
        y = transpose(x)

        p = Kepert_linear_assym(x, y, lat=15, rm=4e4, vm=40.0, b=1.3, \
                                dp=3680.13292308, pe=1e5, rho=1.1, Ut=-5)

        assert allclose(p.swf4, l, rtol=1.e-4, atol=1.e-8)



#-------------------------------------------------------------
if __name__ == "__main__":
    unittest.main()
    #suite = unittest.makeSuite(Test_Kepert,'test_kepert_V')
    #runner = unittest.TextTestRunner()
    #runner.run(suite)
