FROM continuumio/conda-ci-linux-64-python3.8:latest

USER root

SHELL [ "/bin/bash"]

WORKDIR /app/scripts

COPY ./Detection /app/Detection
COPY ./docker/autophoto/build/autophoto-build.sh /app/build/
RUN /app/build/autophoto-build.sh

COPY ./docker/.build/apt-build.sh /app/build/
RUN /app/build/apt-build.sh

COPY ./docker/.build/pip-build.sh /app/build/
RUN /app/build/pip-build.sh

COPY ./docker/.build/start.sh /app/build/
CMD ["/app/build/start.sh"]
