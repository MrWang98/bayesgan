import json
import time

d_losses=[0.01,0.02]
g_losses=[0.03,0.04]

tmp1=list(map(float, d_losses))
tmp2=list(map(float, g_losses))


s_acc=92.25
ss_acc=86.79
results = {"disc_losses": tmp1,
            "gen_losses": tmp2,
            "supervised_acc": float(s_acc),
            "semi_supervised_acc": float(ss_acc),
            "timestamp": time.time()}
with open('results_test.json', 'w') as fp:
    json.dump(results, fp)