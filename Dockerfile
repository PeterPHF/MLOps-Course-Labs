FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
RUN pip install --no-cache-dir \
    joblib==1.4.2 \
    "litestar[standard]>=2.21.1" \
    "pandas>=2.2.0" \
    "pydantic>=2.13.4" \
    "scikit-learn==1.6.1"

COPY . .

EXPOSE 8000

CMD ["litestar", "--app", "main:app", "run", "--host", "0.0.0.0", "--port", "8000"]