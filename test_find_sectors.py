import numpy as np
import matplotlib.pyplot as plt
from sympy import Point, Polygon, Line
from scipy.special import ellipj, ellipk

centroid = np.array([0.505, 0.497])
boundary_nodes = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
# for i in boundary_nodes:
#     plt.plot(i,'bx',alpha=0.5,ms=10)
# plt.plot(centroid[0],centroid[1],'rx',alpha=0.5,ms=10)
# plt.plot(centroid[0],0,'gx',ms=10)
# sectorPoint = np.array([centroid[0]+centroid[1]*np.tan(np.pi/3),0])
# deltas = boundary_nodes - centroid
# dist_2 = np.einsum("ij,ij->i", deltas, deltas)
# min_idx = np.argmin(dist_2)

# theta1 = np.arctan(centroid[1]/(1-centroid[0]))
# theta2 = np.arctan(centroid[1]/(sectorPoint[0]-centroid[0]))
# m = centroid[1]/(centroid[0]-sectorPoint[0])
# if(theta1-theta2>0):
#     x = boundary_nodes[min_idx][0]
#     y = m*(1-sectorPoint[0])
#     #plt.plot(x,y,'gx',ms=10)
#     #sectorPoint[0],sectorPoint[1] = x,y
# #else:
#     #plt.plot(sectorPoint[0],sectorPoint[1],'gx',ms=10)

# p1, p2, p3, p4 = map(Point, [(0, 0), (1, 0), (0, 1), (1, 1)])
# poly1 = Polygon(p1, p2, p3, p4)
# pc, ps = map(Point,[(centroid[0],centroid[1]), (1.5*np.sin(np.pi/3)+centroid[0],-1.5*np.cos(np.pi/3)+centroid[1])])
# s1 = Line(pc,ps)
# print(pc.evalf(),ps.evalf())
# # using intersection()
# intx = poly1.intersection(s1)
# for p in poly1.vertices:
#     plt.plot(p.coordinates[0],p.coordinates[1],'bx',ms=10)

# plt.plot(pc.coordinates[0],pc.coordinates[1], 'cx',ms=10)
# plt.plot(ps.coordinates[0],ps.coordinates[1], 'cx',ms=10)

# for i in intx:
#     print(i)
#     plt.plot(i.coordinates[0],i.coordinates[1], 'gx',ms=10)

# plt.show()

#
# The scipy implementation of ellipj only works on reals;
# this gives cn(z,k) for complex z.
# It works with array or scalar input.
#
def cplx_cn(z, k):
    z = np.asarray(z)
    if z.dtype != complex:
        return ellipj(z, k)[1]

    # # note, 'dn' result of ellipj is buggy, as of Mar 2015
    # # see https://github.com/scipy/scipy/issues/3904
    ss, cc = ellipj(z.real, k)[:2]
    dd = np.sqrt(1 - k * ss ** 2)  # make our own dn
    s1, c1 = ellipj(z.imag, 1 - k)[:2]
    d1 = np.sqrt(1 - k * s1 ** 2)

    # UPDATE: scipy bug seems to have been fixed mid 2016, so
    # four lines above could be done as these two, if you have that.
    # ss,cc,dd = ellipj( z.real, k )
    # s1,c1,d1 = ellipj( z.imag, 1-k )

    ds1 = dd * s1
    den = 1 - ds1 ** 2
    rx = cc * c1 / den
    ry = ss * ds1 * d1 / den
    return rx - 1j * ry


#
# Kval is the first solution to cn(x,1/2) = 0
# This is K(k) (where 4*K(k) is the period of the function).
Kval = ellipk(0.5)  # 1.8540746773013719

#######################################################
# map a complex point in unit square to unit circle
# The following points are the corners of the square (and map to themselves):
#     1    -1     j    -j
#  The origin also maps to itself.
# Points which are in : abs( re(z)) <=1, abs(im(z)) <=1, but outside the square, will map to
# points outside the unit circle, but are still consistent with mapping a full-sphere
# peirce projection to a full-sphere stereographic projection; however that means that
# the corners 1+j, 1-j, -1+j -1-j all map to the 'south pole' at infinity. You will get
# a divide-by-zero, or near to it, at or near those points.
# It works with array or scalar input.
#
def peirce_map(z):
    return cplx_cn(Kval * (1 - z), 0.5)


t = np.linspace(0, 2 * np.pi, 100)
x = 1.5 * np.cos(t)
y = 1.5 * np.sin(t)
plt.plot(x, y, "r-")
for i in range(len(x)):
    z = peirce_map(x[i] + 1j * y[i])
    plt.plot(np.real(z), np.imag(z), "gx")

plt.show()
