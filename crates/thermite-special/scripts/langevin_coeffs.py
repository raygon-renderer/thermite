import mpmath as mp, struct
mp.mp.dps = 50
def g(t):
    if t == 0: return mp.mpf(1)/3
    x = mp.sqrt(t); return (mp.coth(x) - 1/x)/x
L = lambda x: mp.coth(x) - 1/x
def Linv(y):
    if y == 0: return mp.mpf(0)
    return mp.findroot(lambda x: L(x)-y, y*(3-y*y)/(1-y*y))
qc={}
def q(s):
    if s in qc: return qc[s]
    if s == 0: v = mp.mpf(3)
    else:
        y = mp.sqrt(s); v = Linv(y)*(1-s)/y
    qc[s]=v; return v
def remez_poly(f, a, b, n, iters=10, M=500):
    N = n+2
    xs = [ (a+b)/2 + (b-a)/2*mp.cos(mp.pi*(N-1-i)/(N-1)) for i in range(N)]
    grid = [a + (b-a)*k/M for k in range(M+1)]
    fg = [f(x) for x in grid]
    for it in range(iters):
        A = mp.matrix(N, N); rhs = mp.matrix(N,1)
        for i,x in enumerate(xs):
            fx = f(x)
            for j in range(n+1): A[i,j] = x**j
            A[i,n+1] = (-1)**i * fx
            rhs[i] = fx
        sol = mp.lu_solve(A, rhs)
        c = [sol[j] for j in range(n+1)]
        vals = [(mp.polyval(c[::-1], x) - fx)/fx for x,fx in zip(grid,fg)]
        ext=[]; i=0
        while i <= M:
            j=i; s=mp.sign(vals[i]); best=i
            while j<=M and mp.sign(vals[j])==s:
                if abs(vals[j])>abs(vals[best]): best=j
                j+=1
            ext.append(grid[best]); i=j
        while len(ext)>N:
            if abs(vals[grid.index(ext[0])])<abs(vals[grid.index(ext[-1])]): ext.pop(0)
            else: ext.pop()
        if len(ext)<N: break
        # local refine each extremum by golden search on |err|
        newxs=[]
        for e in ext:
            k = grid.index(e); lo = grid[max(k-1,0)]; hi = grid[min(k+1,M)]
            err = lambda x: -abs((mp.polyval(c[::-1], x) - f(x))/f(x))
            try:
                xm = mp.findroot(lambda x: mp.diff(err, x), (lo, hi), solver='bisect') if lo<hi else e
                if not (lo <= xm <= hi): xm = e
            except Exception: xm = e
            newxs.append(xm)
        xs=newxs
    maxe = max(abs(v) for v in vals)
    return c, maxe
def rf64(x): return float(x)
def rf32(x): return struct.unpack('f', struct.pack('f', float(x)))[0]
def check(f, c, a, b, M=4000):
    grid = [a + (b-a)*k/M for k in range(1, M+1)]
    return max(abs((mp.polyval([mp.mpf(x) for x in c[::-1]], t) - f(t))/f(t)) for t in grid)
def hilo(x):
    hi = float(x); lo = float(x - hi); return hi, lo

out = {}
print("== f64 forward [0,4] deg 15")
c,e = remez_poly(g, mp.mpf(0), mp.mpf(4), 15); cf=[rf64(x) for x in c]
print("fit err", mp.nstr(e,3), "rounded err", mp.nstr(check(g,cf,0,4),3)); out['F64_SMALL']=cf
print("== f32 forward [0,4] deg 7")
c,e = remez_poly(g, mp.mpf(0), mp.mpf(4), 7); cf=[rf32(x) for x in c]
print("fit err", mp.nstr(e,3), "rounded err", mp.nstr(check(g,cf,0,4),3)); out['F32_SMALL']=cf
print("== DD forward [0,1] deg 20")
c,e = remez_poly(g, mp.mpf(0), mp.mpf(1), 20, iters=8, M=400); 
print("fit err", mp.nstr(e,3)); out['DD_SMALL']=[hilo(x) for x in c]
print("== inv seed f64 [0,0.85^2] deg 8")
b = mp.mpf('0.85')**2
c,e = remez_poly(q, mp.mpf(0), b, 8, iters=10, M=300); cf=[rf64(x) for x in c]
print("fit err", mp.nstr(e,3), "rounded", mp.nstr(check(q,cf,0,b,600),3)); out['F64_INV']=cf
print("== inv seed f32 [0,0.85^2] deg 4")
c,e = remez_poly(q, mp.mpf(0), b, 4, iters=8, M=300); cf=[rf32(x) for x in c]
print("fit err", mp.nstr(e,3), "rounded", mp.nstr(check(q,cf,0,b,600),3)); out['F32_INV']=cf
print("== forward, short tables for the Worst/Medium tiers")
c,e = remez_poly(g, mp.mpf(0), mp.mpf(4), 10); cf=[rf64(x) for x in c]
print("f64 lo fit", mp.nstr(e,3), "rounded", mp.nstr(check(g,cf,0,4),3)); out['F64_SMALL_LO']=cf
c,e = remez_poly(g, mp.mpf(0), mp.mpf(4), 4); cf=[rf32(x) for x in c]
print("f32 lo fit", mp.nstr(e,3), "rounded", mp.nstr(check(g,cf,0,4),3)); out['F32_SMALL_LO']=cf
import json; json.dump(out, open('crates/thermite-special/scripts/langevin_coeffs.json','w'), indent=1)
for k,v in out.items(): print(k, v)
