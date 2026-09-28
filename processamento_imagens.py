#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Atividade Lab - Processamento Distribuido de Imagens Medicas com MPI
Disciplina: Computacao Distribuida - Universidade Presbiteriana Mackenzie
Aluno: Artur Campi - RA 10436740

Triagem simulada de radiografias de torax distribuida com mpi4py.
Usa obrigatoriamente os 5 mecanismos coletivos do MPI:
  MPI_Bcast   -> comm.bcast   (parametros globais)
  MPI_Scatter -> comm.scatter (faixas horizontais da imagem)
  MPI_Barrier -> comm.Barrier (sincronizacao de fases)
  MPI_Reduce  -> comm.reduce  (metricas numericas globais)
  MPI_Gather  -> comm.gather  (relatorios descritivos por no)

AVISO: simulacao computacional para fins didaticos. NAO e um sistema de
diagnostico medico real.

Uso:
  mpirun -np 4 python3 processamento_imagens.py                 # 2000 x 2000
  mpirun -np 4 python3 processamento_imagens.py 4000            # 4000 x 4000
  mpirun -np 4 python3 processamento_imagens.py --linhas 2003 --colunas 2000
  mpirun -np 4 python3 processamento_imagens.py 4000 --atraso 0 --csv resultados.csv
"""

import argparse
import socket
import sys
import time

import numpy as np
from mpi4py import MPI

# ---------------------------------------------------------------------------
# Etapa 1 - Inicializacao do ambiente MPI
# ---------------------------------------------------------------------------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
ROOT = 0

# Parametros padrao de teste
LINHAS = 2000
COLUNAS = 2000
LIMIAR_SUSPEITO = 200      # pixel > 200  -> suspeito (alterado)
LIMIAR_ALTO = 230          # pixel > 230  -> altamente suspeito (lesao densa)
PCT_CRITICO = 5.0          # % minimo de suspeitos para faixa CRITICA
PCT_ATENCAO = 1.0          # % minimo de suspeitos para faixa em ATENCAO
PCT_ALTO_CRITICO = 1.0     # % de altamente suspeitos que torna a faixa CRITICA


def ler_argumentos():
    p = argparse.ArgumentParser(description="Triagem distribuida de radiografias com MPI")
    p.add_argument("n", nargs="?", type=int, default=None,
                   help="dimensao da imagem quadrada N x N (atalho para --linhas N --colunas N)")
    p.add_argument("--linhas", type=int, default=None)
    p.add_argument("--colunas", type=int, default=None)
    p.add_argument("--atraso", type=float, default=0.5,
                   help="fator de latencia artificial: ranks impares dormem atraso*rank segundos (0 desliga)")
    p.add_argument("--cenario", choices=["nenhuma", "padrao", "extensa"], default="padrao",
                   help="quantidade de focos de consolidacao injetados na radiografia")
    p.add_argument("--seed", type=int, default=2026, help="semente do gerador (mesma imagem em todos os testes)")
    p.add_argument("--csv", default=None, help="arquivo CSV onde o root acrescenta os tempos medidos")
    p.add_argument("--analise", choices=["vetorizada", "laco"], default="vetorizada",
                   help="vetorizada: mascaras NumPy (padrao) | laco: percorre pixel a pixel em Python puro (carga de CPU alta)")
    p.add_argument("--quiet", action="store_true", help="nao imprime a auditoria por processo")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Etapa 2 - Geracao da radiografia simulada (apenas no root)
# ---------------------------------------------------------------------------
def gerar_radiografia(linhas, colunas, cenario="padrao", seed=2026):
    """Gera uma matriz uint8 que imita uma radiografia de torax PA.

    Estrutura (coordenadas normalizadas em [-1, 1]):
      - fundo externo (ar fora do corpo): escuro (~25)
      - partes moles do torax: intermediario (~110)
      - mediastino e coluna: claros, mas abaixo do limiar (~150-180)
      - campos pulmonares: escuros/normais (~55-95) com sombra das costelas
      - diafragma: base clara dos pulmoes
      - focos de consolidacao: manchas gaussianas muito claras (200-255)
    """
    rng = np.random.default_rng(seed)
    y = np.linspace(-1.0, 1.0, linhas, dtype=np.float32)[:, None]
    x = np.linspace(-1.0, 1.0, colunas, dtype=np.float32)[None, :]

    img = np.full((linhas, colunas), 25.0, dtype=np.float32)          # fundo externo

    torax = (x / 0.95) ** 2 + ((y - 0.08) / 1.02) ** 2 <= 1.0
    img[torax] = 112.0                                                 # partes moles

    # Campos pulmonares (duas elipses). Na matriz: pulmao "esquerdo" = colunas < N/2
    pulmao_esq = ((x + 0.43) / 0.33) ** 2 + ((y + 0.02) / 0.72) ** 2 <= 1.0
    pulmao_dir = ((x - 0.43) / 0.33) ** 2 + ((y + 0.02) / 0.72) ** 2 <= 1.0
    pulmoes = pulmao_esq | pulmao_dir
    # gradiente craniocaudal: apice um pouco mais claro que a base aerada
    base_pulmao = np.broadcast_to(62.0 - 10.0 * y, img.shape)
    img[pulmoes] = base_pulmao[pulmoes]

    # Costelas: faixas periodicas levemente mais claras sobre os pulmoes
    costelas = (np.sin((y * 11.0 + np.abs(x) * 2.2) * np.pi) > 0.82)
    img[pulmoes & costelas] += 28.0

    # Mediastino + coluna vertebral
    mediastino = (np.abs(x) < 0.13) & torax & (y > -0.75)
    img[mediastino] = 152.0
    coluna = (np.abs(x) < 0.045) & torax
    img[coluna] = 176.0

    # Diafragma (cupulas claras abaixo dos pulmoes)
    diafragma = torax & (y > 0.70 - 0.12 * np.cos(np.abs(x) * 3.0)) & ~mediastino
    img[diafragma] = 138.0

    # Ruido de aquisicao
    img += rng.normal(0.0, 6.0, size=img.shape).astype(np.float32)

    # Focos de consolidacao (lesoes) - (centro_y, centro_x, sigma, amplitude)
    focos = {
        "nenhuma": [],
        "padrao": [(-0.12, 0.45, 0.150, 165.0),   # foco principal - pulmao direito (terco medio)
                   (0.34, 0.40, 0.070, 160.0),    # foco satelite - pulmao direito (base)
                   (-0.42, -0.45, 0.040, 155.0)], # pequeno foco - pulmao esquerdo (apice)
        "extensa": [(-0.12, 0.45, 0.20, 175.0), (0.30, 0.40, 0.14, 170.0),
                    (-0.30, -0.45, 0.16, 170.0), (0.25, -0.40, 0.12, 165.0)],
    }[cenario]
    for cy, cx, sig, amp in focos:
        # calcula a gaussiana apenas numa janela de +-3 sigma (economiza memoria)
        r0 = max(0, int(((cy - 3 * sig) + 1) / 2 * (linhas - 1)))
        r1 = min(linhas, int(((cy + 3 * sig) + 1) / 2 * (linhas - 1)) + 2)
        c0 = max(0, int(((cx - 3 * sig) + 1) / 2 * (colunas - 1)))
        c1 = min(colunas, int(((cx + 3 * sig) + 1) / 2 * (colunas - 1)) + 2)
        yy, xx = y[r0:r1], x[:, c0:c1]
        # perfil super-gaussiano: nucleo denso quase plano (consolidacao) com bordas suaves
        mancha = amp * np.exp(-((((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sig ** 2)) ** 2))
        textura = rng.normal(0.0, 10.0, size=mancha.shape).astype(np.float32)
        img[r0:r1, c0:c1] += (mancha + textura * (mancha > 40)).astype(np.float32)

    np.clip(img, 0, 255, out=img)
    return img.astype(np.uint8)


def dividir_em_faixas(imagem, nprocs):
    """Etapa 5 - Fatiamento balanceado (estrategia para o desafio de divisibilidade).

    np.array_split gera nprocs blocos CONTIGUOS de linhas cujos tamanhos diferem
    em no maximo 1 linha: os primeiros (linhas % nprocs) blocos recebem 1 linha
    extra. Nao ha padding, nem descarte, nem redimensionamento: nenhum pixel e
    perdido e nenhum pixel artificial e analisado. Cada bloco vai acompanhado do
    indice global da sua primeira linha, para a auditoria por faixa.
    """
    blocos = np.array_split(imagem, nprocs, axis=0)
    faixas, inicio = [], 0
    for b in blocos:
        faixas.append({"inicio": inicio, "bloco": np.ascontiguousarray(b)})
        inicio += b.shape[0]
    return faixas


def classificar_faixa(pct_suspeitos, pct_altos, p):
    """Etapa 7 - Classificacao heuristica da faixa local."""
    if pct_suspeitos >= p["pct_critico"] or pct_altos >= p["pct_alto_critico"]:
        return "CRITICA"
    if pct_suspeitos >= p["pct_atencao"]:
        return "ATENCAO"
    return "NORMAL"


def main():
    args = ler_argumentos()
    t_inicio = MPI.Wtime()

    imagem_completa = None
    parametros = None

    # -----------------------------------------------------------------------
    # Etapas 2 e 3 - root gera a radiografia e define os parametros
    # -----------------------------------------------------------------------
    if rank == ROOT:
        linhas = args.linhas or args.n or LINHAS
        colunas = args.colunas or args.n or COLUNAS
        print(f"[Master] Inicializando analise com {size} processos MPI "
              f"({linhas} x {colunas} pixels, cenario '{args.cenario}')...", flush=True)
        imagem_completa = gerar_radiografia(linhas, colunas, args.cenario, args.seed)
        parametros = {
            "linhas": linhas,
            "colunas": colunas,
            "limiar_suspeito": LIMIAR_SUSPEITO,
            "limiar_alto": LIMIAR_ALTO,
            "pct_critico": PCT_CRITICO,
            "pct_atencao": PCT_ATENCAO,
            "pct_alto_critico": PCT_ALTO_CRITICO,
            "atraso": args.atraso,
            "analise": args.analise,
        }
    t_geracao = MPI.Wtime()

    # Etapa 3 - Difusao dos parametros via Broadcast
    parametros = comm.bcast(parametros, root=ROOT)

    # -----------------------------------------------------------------------
    # Etapa 4 - Sincronizacao inicial (todos parametrizados antes do trafego pesado)
    # -----------------------------------------------------------------------
    comm.Barrier()
    t_config = MPI.Wtime()

    # -----------------------------------------------------------------------
    # Etapa 5 - Particionamento e distribuicao via Scatter
    # -----------------------------------------------------------------------
    faixas = dividir_em_faixas(imagem_completa, size) if rank == ROOT else None
    minha_faixa = comm.scatter(faixas, root=ROOT)
    del faixas
    bloco_local = minha_faixa["bloco"]
    linha_inicio = minha_faixa["inicio"]
    linha_fim = linha_inicio + bloco_local.shape[0] - 1
    t_scatter = MPI.Wtime()

    # -----------------------------------------------------------------------
    # Etapa 6 - Analise local (vetorizada com NumPy, pixel a pixel)
    # -----------------------------------------------------------------------
    total_local = int(bloco_local.size)
    col_meio = parametros["colunas"] // 2                    # floor(N/2)
    lim_s, lim_a = parametros["limiar_suspeito"], parametros["limiar_alto"]

    if parametros["analise"] == "laco":
        # Inspecao literal pixel a pixel em Python puro (sem vetorizacao):
        # custo de CPU proporcional ao numero de pixels -> evidencia o ganho do paralelismo.
        soma_local = max_local = suspeitos_esq = suspeitos_dir = altos_local = 0
        for linha in bloco_local.tolist():
            for j, v in enumerate(linha):
                soma_local += v
                if v > max_local:
                    max_local = v
                if v > lim_s:                                # pixel suspeito
                    if j < col_meio:
                        suspeitos_esq += 1                   # coluna <  N/2 -> pulmao esquerdo
                    else:
                        suspeitos_dir += 1                   # coluna >= N/2 -> pulmao direito
                    if v > lim_a:                            # altamente suspeito (lim_a > lim_s)
                        altos_local += 1
    else:
        # Mesma inspecao pixel a pixel, porem vetorizada pelo NumPy (laco em C)
        soma_local = int(bloco_local.sum(dtype=np.uint64))
        max_local = int(bloco_local.max()) if total_local else 0
        suspeito_mask = bloco_local > lim_s
        suspeitos_esq = int(np.count_nonzero(suspeito_mask[:, :col_meio]))   # coluna <  N/2
        suspeitos_dir = int(np.count_nonzero(suspeito_mask[:, col_meio:]))   # coluna >= N/2
        altos_local = int(np.count_nonzero(bloco_local > lim_a))
    suspeitos_local = suspeitos_esq + suspeitos_dir

    # Etapa 7 - Classificacao heuristica da faixa
    pct_suspeitos = 100.0 * suspeitos_local / total_local if total_local else 0.0
    pct_altos = 100.0 * altos_local / total_local if total_local else 0.0
    classificacao_local = classificar_faixa(pct_suspeitos, pct_altos, parametros)
    t_analise_local = MPI.Wtime()

    # -----------------------------------------------------------------------
    # Etapa 8 - Simulacao de heterogeneidade de hardware (stragglers)
    # -----------------------------------------------------------------------
    if rank % 2 != 0 and parametros["atraso"] > 0:
        time.sleep(parametros["atraso"] * rank)
    t_pronto = MPI.Wtime()

    # -----------------------------------------------------------------------
    # Etapa 9 - Sincronizacao pre-consolidacao
    # -----------------------------------------------------------------------
    comm.Barrier()
    t_barreira = MPI.Wtime()
    espera_barreira_ms = (t_barreira - t_pronto) * 1000.0

    # -----------------------------------------------------------------------
    # Etapa 10 - Consolidacao numerica global com Reduce
    # -----------------------------------------------------------------------
    total_pixels_global = comm.reduce(total_local, op=MPI.SUM, root=ROOT)
    soma_global = comm.reduce(soma_local, op=MPI.SUM, root=ROOT)
    max_global = comm.reduce(max_local, op=MPI.MAX, root=ROOT)
    suspeitos_global = comm.reduce(suspeitos_local, op=MPI.SUM, root=ROOT)
    altos_global = comm.reduce(altos_local, op=MPI.SUM, root=ROOT)
    esq_global = comm.reduce(suspeitos_esq, op=MPI.SUM, root=ROOT)
    dir_global = comm.reduce(suspeitos_dir, op=MPI.SUM, root=ROOT)
    t_max_analise = comm.reduce((t_analise_local - t_scatter) * 1000.0, op=MPI.MAX, root=ROOT)

    # -----------------------------------------------------------------------
    # Etapa 11 - Coleta dos relatorios descritivos com Gather
    # -----------------------------------------------------------------------
    relatorio_local = {
        "rank": rank,
        "host": socket.gethostname(),
        "linhas_inicio": linha_inicio,
        "linhas_fim": linha_fim,
        "n_linhas": bloco_local.shape[0],
        "pixels": total_local,
        "suspeitos": suspeitos_local,
        "altos": altos_local,
        "esq": suspeitos_esq,
        "dir": suspeitos_dir,
        "pct": pct_suspeitos,
        "classificacao": classificacao_local,
        "max_local": max_local,
        "analise_ms": (t_analise_local - t_scatter) * 1000.0,
        "espera_ms": espera_barreira_ms,
    }
    todos_relatorios = comm.gather(relatorio_local, root=ROOT)

    # -----------------------------------------------------------------------
    # Etapa 12 - Relatorio final consolidado e diagnostico simulado (root)
    # -----------------------------------------------------------------------
    if rank == ROOT:
        t_fim = MPI.Wtime()
        t_total = (t_fim - t_inicio) * 1000.0
        t_paralelo = (t_fim - t_geracao) * 1000.0
        media_intensidade = soma_global / total_pixels_global
        taxa_comprometida = 100.0 * suspeitos_global / total_pixels_global
        taxa_altos = 100.0 * altos_global / total_pixels_global
        faixas_criticas = sum(r["classificacao"] == "CRITICA" for r in todos_relatorios)
        faixas_atencao = sum(r["classificacao"] == "ATENCAO" for r in todos_relatorios)

        if taxa_comprometida >= parametros["pct_critico"] or taxa_altos >= parametros["pct_alto_critico"]:
            diagnostico = "QUADRO CRITICO / ALTA CONCENTRACAO DE ALTERACOES"
        elif taxa_comprometida >= parametros["pct_atencao"] or faixas_criticas or faixas_atencao:
            diagnostico = "ATENCAO CLINICA"
        else:
            diagnostico = "SEM INDICIOS RELEVANTES"

        if dir_global > esq_global:
            lado_critico = "Direito"
        elif esq_global > dir_global:
            lado_critico = "Esquerdo"
        else:
            lado_critico = "Equilibrado"
        maior, menor = max(esq_global, dir_global), min(esq_global, dir_global)
        razao = f"{maior / menor:.1f}x o outro lado" if menor else "lado oposto sem suspeitas"

        L, C = parametros["linhas"], parametros["colunas"]
        print("\n" + "=" * 72)
        print("        RELATORIO CONSOLIDADO DE TRIAGEM DISTRIBUIDA (SIMULACAO)")
        print("=" * 72)
        print("--- Informacoes Gerais ---")
        print(f"Dimensoes do Exame       : {L} x {C} pixels ({L * C:,} px)".replace(",", "."))
        print(f"Processos MPI Utilizados : {size}  (hosts: {', '.join(sorted({r['host'] for r in todos_relatorios}))})")
        print(f"Limiares Adotados        : suspeito > {parametros['limiar_suspeito']} | "
              f"alto > {parametros['limiar_alto']} | faixa critica >= {parametros['pct_critico']:.1f}%")
        print(f"Latencia Artificial      : {parametros['atraso']} s x rank (ranks impares) | analise: {parametros['analise']}")
        print(f"Tempo Total de Execucao  : {t_total:.2f} ms")
        print(f"  geracao(root) {(t_geracao - t_inicio) * 1000:.1f} ms | bcast+barrier {(t_config - t_geracao) * 1000:.1f} ms | "
              f"scatter {(t_scatter - t_config) * 1000:.1f} ms | analise(max) {t_max_analise:.1f} ms | "
              f"barreira->fim {(t_fim - t_barreira) * 1000:.1f} ms")
        print("--- Metricas Globais ---")
        print(f"Intensidade Media Global : {media_intensidade:.2f} (Maxima: {max_global})")
        print(f"Total de Pixels Suspeitos: {suspeitos_global} ({taxa_comprometida:.2f}%)")
        print(f"Altamente Suspeitos      : {altos_global} ({taxa_altos:.2f}%)")
        print("--- Lateralidade Pulmonar ---")
        print(f"  - Pulmao Esquerdo      : {esq_global} suspeitos")
        print(f"  - Pulmao Direito       : {dir_global} suspeitos")
        print(f"Maior Concentracao       : Pulmao {lado_critico} ({razao})")
        if not args.quiet:
            print("--- Auditoria por Processo (Faixas) ---")
            for rel in todos_relatorios:
                print(f"P{rel['rank']:02d}@{rel['host']:<7} | Linhas [{rel['linhas_inicio']:04d}-{rel['linhas_fim']:04d}] "
                      f"| Susp: {rel['suspeitos']:06d} ({rel['pct']:5.2f}%) | Max: {rel['max_local']:03d} "
                      f"| Espera barreira: {rel['espera_ms']:7.1f} ms | Faixa: {rel['classificacao']}")
        print("-" * 72)
        print(f"CLASSIFICACAO GERAL DA RADIOGRAFIA: {diagnostico}")
        print(f"(faixas criticas: {faixas_criticas}/{size} | faixas em atencao: {faixas_atencao}/{size})")
        print("=" * 72 + "\n", flush=True)

        if args.csv:
            import os
            novo = not os.path.exists(args.csv)
            with open(args.csv, "a", encoding="utf-8") as f:
                if novo:
                    f.write("analise,linhas,colunas,processos,atraso,t_total_ms,t_paralelo_ms,t_geracao_ms,"
                            "t_scatter_ms,t_analise_max_ms,suspeitos,esq,dir,diagnostico\n")
                f.write(f"{parametros['analise']},{L},{C},{size},{parametros['atraso']},{t_total:.2f},{t_paralelo:.2f},"
                        f"{(t_geracao - t_inicio) * 1000:.2f},{(t_scatter - t_config) * 1000:.2f},"
                        f"{t_max_analise:.2f},{suspeitos_global},{esq_global},{dir_global},{diagnostico}\n")


if __name__ == "__main__":
    main()
