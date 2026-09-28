#!/usr/bin/env bash
# Sobe o cluster Docker, inicia o SSH e copia o script para todos os nos.
set -e
cd "$(dirname "$0")"
docker compose up -d --build
for no in master worker1 worker2 worker3; do
  docker compose exec $no service ssh start
  docker cp processamento_imagens.py $no:/home/mpiuser/
done
# teste rapido de conectividade SSH/MPI entre os nos
docker compose exec master su - mpiuser -c "mpirun --hostfile hosts -np 4 hostname"
