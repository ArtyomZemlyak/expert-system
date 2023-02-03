FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

USER root

ENV DOCKER=true

WORKDIR /app/scripts

COPY ./esgui/requirements.txt /usr/src/app/
COPY ./esgui/build/esgui-build.sh /app/build/
RUN chmod +x /app/build/esgui-build.sh && \
    /app/build/esgui-build.sh

COPY ./esgui/build/start.sh /app/build/
RUN chmod +x /app/build/start.sh

CMD [ "/app/build/start.sh" ]
