#!/usr/bin/env bash
# Sobe o cluster Docker, inicia o SSH e copia o script para todos os nos.
set -e
cd "$(dirname "$0")"
docker compose up -d --build

# GitHub Codespaces (docker-in-docker): as tabelas iptables-legacy do host tem
# politica FORWARD DROP e so liberam a docker0, entao o trafego entre containers
# da rede "mpinet" era descartado (ssh/mpirun travavam). Libera a chain DOCKER-USER.
if command -v iptables-legacy >/dev/null 2>&1 && sudo -n iptables-legacy -S FORWARD 2>/dev/null | grep -q -- '-P FORWARD DROP'; then
  sudo iptables-legacy -C DOCKER-USER -j ACCEPT 2>/dev/null || sudo iptables-legacy -I DOCKER-USER -j ACCEPT
fi
for no in master worker1 worker2 worker3; do
  docker compose exec $no service ssh start
  docker cp processamento_imagens.py $no:/home/mpiuser/
done
# teste rapido de conectividade SSH/MPI entre os nos
docker compose exec master su - mpiuser -c "mpirun --hostfile hosts -np 4 hostname"
