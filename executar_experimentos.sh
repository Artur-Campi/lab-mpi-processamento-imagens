#!/usr/bin/env bash
# Bateria de experimentos de desempenho no cluster Docker.
# Grava todas as execucoes em resultados/tempos.csv (REPS repeticoes por configuracao).
set -e
cd "$(dirname "$0")"
REPS=${REPS:-3}
mkdir -p resultados
docker compose exec -T master rm -f /home/mpiuser/tempos.csv

rodar () {  # $1=N  $2=processos  $3=atraso
  docker compose exec -T master su - mpiuser -c \
    "mpirun --hostfile hosts -np $2 --oversubscribe python3 processamento_imagens.py $1 --atraso $3 --quiet --csv tempos.csv" > /dev/null
  printf "N=%-5s np=%s atraso=%-3s ok\n" "$1" "$2" "$3"
}

# 1) cenario do roteiro: com latencia artificial (0.5 s x rank nos ranks impares)
for N in 2000 4000; do for NP in 1 2 4 8; do for r in $(seq $REPS); do rodar $N $NP 0.5; done; done; done
# 2) sem latencia artificial: mede o speedup "real" da computacao distribuida
for N in 500 2000 4000 6000; do for NP in 1 2 4 8; do for r in $(seq $REPS); do rodar $N $NP 0; done; done; done

docker cp master:/home/mpiuser/tempos.csv resultados/tempos.csv
echo "Resultados salvos em resultados/tempos.csv"
