#!/usr/bin/env python3
"""Le resultados/tempos.csv e gera a tabela comparativa (media das repeticoes),
speedup S(p) = T(1)/T(p), eficiencia E(p) = S(p)/p e o grafico de speedup."""
import csv
import sys
from collections import defaultdict
from statistics import mean, stdev

arq = sys.argv[1] if len(sys.argv) > 1 else "resultados/tempos.csv"
dados = defaultdict(list)
with open(arq, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        dados[(float(r["atraso"]), int(r["linhas"]), int(r["processos"]))].append(float(r["t_total_ms"]))

linhas_md = []
for atraso in sorted({k[0] for k in dados}, reverse=True):
    linhas_md.append(f"\n### Latencia artificial = {atraso} s x rank\n")
    linhas_md.append("| N | Processos | Tempo medio (ms) | Desvio (ms) | Speedup | Eficiencia |")
    linhas_md.append("|---|---|---|---|---|---|")
    for n in sorted({k[1] for k in dados if k[0] == atraso}):
        base = mean(dados[(atraso, n, 1)]) if (atraso, n, 1) in dados else None
        for p in sorted({k[2] for k in dados if k[0] == atraso and k[1] == n}):
            ts = dados[(atraso, n, p)]
            m = mean(ts)
            d = stdev(ts) if len(ts) > 1 else 0.0
            s = base / m if base else float("nan")
            linhas_md.append(f"| {n} | {p} | {m:.1f} | {d:.1f} | {s:.2f} | {100 * s / p:.0f}% |")
print("\n".join(linhas_md))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for n in sorted({k[1] for k in dados if k[0] == 0.0}):
        ps = sorted({k[2] for k in dados if k[0] == 0.0 and k[1] == n})
        base = mean(dados[(0.0, n, 1)])
        ax.plot(ps, [base / mean(dados[(0.0, n, p)]) for p in ps], marker="o", label=f"N = {n}")
    ax.plot([1, 8], [1, 8], "k--", lw=1, label="ideal (linear)")
    ax.set_xlabel("processos MPI"); ax.set_ylabel("speedup T(1)/T(p)")
    ax.set_title("Speedup sem latencia artificial"); ax.grid(alpha=.3); ax.legend()
    fig.savefig("resultados/grafico_speedup.png", dpi=150, bbox_inches="tight")
    print("\nGrafico salvo em resultados/grafico_speedup.png")
except ImportError:
    print("\n(matplotlib nao instalado: pip install matplotlib para gerar o grafico)")
