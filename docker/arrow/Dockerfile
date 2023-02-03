FROM continuumio/conda-ci-linux-64-python3.8:latest

USER root

SHELL [ "/bin/bash"]

WORKDIR /app/scripts

COPY ./arrow/requirements.txt /usr/src/app/
COPY ./arrow/build/pyarrow-build.sh /app/build/
RUN /app/build/pyarrow-build.sh

COPY ./.build/apt-build.sh /app/build/
RUN /app/build/apt-build.sh

COPY ./.build/pip-build.sh /app/build/
RUN /app/build/pip-build.sh

COPY ./.build/start.sh /app/build/

CMD ["/app/build/start.sh"]
