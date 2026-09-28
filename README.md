# Lab MPI: Processamento Distribuído de Imagens Médicas

Atividade Lab da disciplina **Computação Distribuída** (Universidade Presbiteriana Mackenzie, Faculdade de Computação e Informática).
Prof. Alcides / Prof. Mário. Aluno: **Artur Campi** (RA 10436740).

Sistema distribuído em Python com `mpi4py` que faz a triagem **simulada** de radiografias de tórax,
dividindo a imagem em faixas horizontais entre os processos de um cluster Docker de 4 nós
(`master`, `worker1`, `worker2`, `worker3`) executado no GitHub Codespaces.

> Simulação computacional com fins didáticos. Não é um sistema de diagnóstico médico real.

## Mecanismos MPI utilizados

| Etapa | Primitiva | Uso |
|---|---|---|
| 3 | `comm.bcast` | difusão dos parâmetros (dimensões, limiares, percentuais, latência) |
| 4 e 9 | `comm.Barrier` | início pós-configuração e sincronização pré-consolidação |
| 5 | `comm.scatter` | distribuição das faixas de linhas (fatiamento balanceado com `np.array_split`) |
| 10 | `comm.reduce` | totais globais (`MPI.SUM`) e intensidade máxima (`MPI.MAX`) |
| 11 | `comm.gather` | coleta dos relatórios descritivos de cada faixa no root |

## Arquivos

| Arquivo | Descrição |
|---|---|
| `processamento_imagens.py` | programa MPI (12 etapas do roteiro) |
| `Dockerfile` / `docker-compose.yml` / `hosts` | cluster de 4 containers com Open MPI + SSH sem senha |
| `preparar_cluster.sh` | sobe o cluster, inicia o SSH e copia o script para os nós |
| `executar_experimentos.sh` | bateria de medições (N = 500 a 6000; 1, 2, 4, 8 processos; análise vetorizada e laço Python) |
| `analisar_resultados.py` | tabela de tempos médios, speedup, eficiência e gráfico |
| `resultados/` | logs das execuções, CSV de tempos e gráfico de speedup |
| `prints/` | capturas de tela do terminal do Codespace |

## Como executar (Codespaces ou qualquer máquina com Docker)

```bash
./preparar_cluster.sh

# execuções do roteiro (2, 4 e 8 processos)
docker compose exec master su - mpiuser -c "mpirun --hostfile hosts -np 2 python3 processamento_imagens.py"
docker compose exec master su - mpiuser -c "mpirun --hostfile hosts -np 4 python3 processamento_imagens.py"
docker compose exec master su - mpiuser -c "mpirun --hostfile hosts -np 8 --oversubscribe python3 processamento_imagens.py"

# imagem 4000 x 4000
docker compose exec master su - mpiuser -c "mpirun --hostfile hosts -np 4 python3 processamento_imagens.py 4000"

# desafio de divisibilidade (2003 linhas em 4 processos)
docker compose exec master su - mpiuser -c "mpirun --hostfile hosts -np 4 python3 processamento_imagens.py --linhas 2003 --colunas 2000"

# medições de desempenho + tabela/gráfico
./executar_experimentos.sh
python3 analisar_resultados.py
```

Opções do programa: `N` (imagem N x N), `--linhas`, `--colunas`, `--atraso` (fator da latência artificial,
padrão 0.5 s x rank nos ranks ímpares; 0 desliga), `--cenario {nenhuma,padrao,extensa}`, `--analise {vetorizada,laco}`, `--seed`, `--csv`, `--quiet`.

## Observação sobre o Codespaces

No Codespaces, o `iptables-legacy` do host vem com `FORWARD DROP` e só libera a `docker0`; sem ajuste, o `ssh`/`mpirun`
entre os containers da rede `mpinet` trava. O `preparar_cluster.sh` já aplica a correção
(`sudo iptables-legacy -I DOCKER-USER -j ACCEPT`).

## Divisibilidade de linhas

O root usa `np.array_split(imagem, size, axis=0)`: blocos contíguos cujo tamanho difere em no máximo 1 linha
(as `linhas % size` primeiras faixas recebem uma linha extra). Cada bloco é enviado junto com o índice global
da sua primeira linha. Não há padding, descarte nem redimensionamento, então nenhum pixel é perdido
(ex.: 2003 linhas em 4 processos: 501 + 501 + 501 + 500).
