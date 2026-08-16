import mpmath as mp, json
mp.mp.dps = 60
def g(t):
    if t == 0: return mp.mpf(1)/3
    x = mp.sqrt(t); return (mp.coth(x) - 1/x)/x
def hilo(x):
    hi = float(x); lo = float(x - hi); return hi, lo
for n in [18, 20, 22, 24]:
    c, err = mp.chebyfit(g, [0, 1], n, error=True)
    c = c[::-1]  # ascending
    grid = [mp.mpf(k)/3000 for k in range(1,3001)]
    rel = max(abs((mp.polyval(c[::-1], t)-g(t))/g(t)) for t in grid)
    print(n, mp.nstr(err,3), mp.nstr(rel,3), flush=True)
    if rel < 2e-33:
        d = json.load(open('crates/thermite-special/scripts/langevin_coeffs.json'))
        d['DD_SMALL'] = [hilo(x) for x in c]
        json.dump(d, open('crates/thermite-special/scripts/langevin_coeffs.json','w'), indent=1)
        print(d['DD_SMALL']); break
