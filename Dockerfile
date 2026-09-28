# Imagem unica usada pelos 4 nos do cluster (master, worker1, worker2, worker3)
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        openmpi-bin libopenmpi-dev python3 python3-numpy python3-mpi4py \
        openssh-server openssh-client iputils-ping procps \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /var/run/sshd

# Parametros MCA do Open MPI para rodar dentro de containers Docker:
#  - vader sem single-copy (CMA/ptrace e bloqueado por padrao no Docker)
#  - trafego TCP apenas pela interface da rede do compose (eth0)
RUN printf "btl_vader_single_copy_mechanism = none\nbtl_tcp_if_include = eth0\noob_tcp_if_include = eth0\n" \
        >> /etc/openmpi/openmpi-mca-params.conf

# Usuario sem privilegios que executa o MPI
RUN useradd -m -s /bin/bash mpiuser

# Chave SSH sem senha compartilhada: como todos os nos usam a MESMA imagem,
# a chave publica ja esta no authorized_keys de todos eles.
USER mpiuser
RUN mkdir -p /home/mpiuser/.ssh \
    && ssh-keygen -q -t ed25519 -N "" -f /home/mpiuser/.ssh/id_ed25519 \
    && cp /home/mpiuser/.ssh/id_ed25519.pub /home/mpiuser/.ssh/authorized_keys \
    && printf "Host *\n  StrictHostKeyChecking no\n  UserKnownHostsFile /dev/null\n  LogLevel ERROR\n" > /home/mpiuser/.ssh/config \
    && chmod 700 /home/mpiuser/.ssh && chmod 600 /home/mpiuser/.ssh/*
COPY --chown=mpiuser:mpiuser hosts /home/mpiuser/hosts

USER root
WORKDIR /home/mpiuser
EXPOSE 22
# Sobe o sshd automaticamente (os comandos "service ssh start" do roteiro continuam validos)
CMD ["/bin/bash", "-c", "service ssh start && sleep infinity"]
