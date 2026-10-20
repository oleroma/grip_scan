import json, sys
for t in sys.argv[1:]:
    q = json.load(open(f'/tmp/claude-0/sdf/smooth/final/solve/{t}.json'))['q']
    z = q['zones']; s = q['smooth']
    print(f"{t}: solve {q['solve_s']}s h {q['h']} | zones " + ' '.join(f"{v[0]:.2f}" for v in z.values()) + f" | dig {q['dig'][0]}")
    print(f"   seats R2 {q['seats']['R2']} R4 {q['seats']['R4']}")
    print('   smooth ' + ' '.join(f"{k.replace('pct_area_concave_r_','').replace('_deg_per_mm','').replace('_mm','')}={v}" for k, v in s.items() if k != 'n'))
