#!/usr/bin/env python3
"""Le resultados/tempos.csv e gera a tabela comparativa (media das repeticoes),
speedup S(p) = T(1)/T(p), eficiencia E(p) = S(p)/p e o grafico de speedup.

T_total    = do inicio do programa ate o relatorio (inclui a geracao sequencial da imagem no root)
T_paralelo = da difusao dos parametros ate o relatorio (exclui a geracao, que e etapa sequencial)
"""
import csv
import sys
from collections import defaultdict
from statistics import mean, stdev

arq = sys.argv[1] if len(sys.argv) > 1 else "resultados/tempos.csv"
tot, par = defaultdict(list), defaultdict(list)
with open(arq, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        k = (r.get("analise", "vetorizada"), float(r["atraso"]), int(r["linhas"]), int(r["processos"]))
        tot[k].append(float(r["t_total_ms"]))
        par[k].append(float(r["t_paralelo_ms"]))

cenarios = sorted({(k[0], k[1]) for k in tot}, key=lambda c: (c[0] != "vetorizada", -c[1]))
saida = []
for modo, atraso in cenarios:
    saida.append(f"\n### Analise {modo} | latencia artificial = {atraso} s x rank\n")
    saida.append("| N | Proc. | T_total medio (ms) | Desvio | Speedup total | Efic. | T_paralelo (ms) | Speedup paralelo |")
    saida.append("|---|---|---|---|---|---|---|---|")
    for n in sorted({k[2] for k in tot if k[:2] == (modo, atraso)}):
        b = (modo, atraso, n, 1)
        bt = mean(tot[b]) if b in tot else None
        bp = mean(par[b]) if b in par else None
        for p in sorted({k[3] for k in tot if k[:3] == (modo, atraso, n)}):
            k = (modo, atraso, n, p)
            mt, mp = mean(tot[k]), mean(par[k])
            d = stdev(tot[k]) if len(tot[k]) > 1 else 0.0
            st = bt / mt if bt else float("nan")
            sp = bp / mp if bp else float("nan")
            saida.append(f"| {n} | {p} | {mt:.1f} | {d:.1f} | {st:.2f} | {100 * st / p:.0f}% | {mp:.1f} | {sp:.2f} |")
print("\n".join(saida))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, modo in zip(axs, ["vetorizada", "laco"]):
        for n in sorted({k[2] for k in par if k[0] == modo and k[1] == 0.0}):
            ps = sorted({k[3] for k in par if k[0] == modo and k[1] == 0.0 and k[2] == n})
            base = mean(par[(modo, 0.0, n, 1)])
            ax.plot(ps, [base / mean(par[(modo, 0.0, n, p)]) for p in ps], marker="o", label=f"N = {n}")
        ax.plot([1, 8], [1, 8], "k--", lw=1, label="ideal")
        ax.set_title(f"Speedup da fase paralela (analise {modo})")
        ax.set_xlabel("processos MPI"); ax.grid(alpha=.3); ax.legend()
    axs[0].set_ylabel("speedup T(1)/T(p)")
    import os; os.makedirs("resultados", exist_ok=True)
    fig.savefig("resultados/grafico_speedup.png", dpi=150, bbox_inches="tight")
    print("\nGrafico salvo em resultados/grafico_speedup.png")
except ImportError:
    print("\n(matplotlib nao instalado: pip install matplotlib para gerar o grafico)")
