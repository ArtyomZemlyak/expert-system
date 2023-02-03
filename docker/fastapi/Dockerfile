FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

USER root

ENV DOCKER=true

WORKDIR /app/scripts

COPY ./fastapi/requirements.txt /usr/src/app/
COPY ./fastapi/build/fastapi-build.sh /app/build/
RUN chmod +x /app/build/fastapi-build.sh && \
    /app/build/fastapi-build.sh

COPY ./.build/apt-build.sh /app/build/
RUN chmod +x /app/build/apt-build.sh && \
    /app/build/apt-build.sh

COPY ./.build/pip-build.sh /app/build/
RUN chmod +x /app/build/pip-build.sh && \
    /app/build/pip-build.sh

COPY ./.build/start.sh /app/build/
RUN chmod +x /app/build/start.sh

CMD ["/app/build/start.sh"]
