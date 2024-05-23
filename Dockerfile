FROM public.ecr.aws/lambda/python:3.12-x86_64 as base

FROM base as builder

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 POETRY_VIRTUALENVS_CREATE=false

RUN pip install poetry-plugin-export

COPY pyproject.toml poetry.lock ./
RUN poetry export --without-hashes -o requirements.txt && \
  pip install --no-deps -r requirements.txt --target /app-deps


FROM base as final

COPY --from=builder /app-deps ${LAMBDA_TASK_ROOT}
COPY . .
RUN mv src/* ./

CMD [ "pyne.main.handler" ]
