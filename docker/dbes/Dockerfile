FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

USER root

ENV DOCKER=true

WORKDIR /app/scripts

COPY ./dbes/requirements.txt /usr/src/app/
COPY ./dbes/build/dbes-build.sh /app/build/
RUN chmod +x /app/build/dbes-build.sh && \
    /app/build/dbes-build.sh

COPY ./dbes/build/start.sh /app/build/
RUN chmod +x /app/build/start.sh

CMD [ "/app/build/start.sh" ]
