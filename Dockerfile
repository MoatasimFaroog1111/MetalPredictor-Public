FROM python:3.14-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1 LIVE_DB_PATH=/data/live_predictions.sqlite3 LIVE_ARTIFACT_DIR=/run/secrets/metalpredictor_artifacts
WORKDIR /app
COPY pyproject.toml requirements-live.txt ./
COPY src ./src
RUN python -m pip install --no-cache-dir -r requirements-live.txt && python -m pip install --no-cache-dir .
COPY live_web ./live_web
COPY run_live.py ./run_live.py
COPY docker_entrypoint_live.py /usr/local/bin/docker_entrypoint_live.py
RUN mkdir -p /data /run/secrets/metalpredictor_artifacts && useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app /data /run/secrets/metalpredictor_artifacts
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=3).read()" || exit 1
ENTRYPOINT ["python","/usr/local/bin/docker_entrypoint_live.py"]
CMD ["python","run_live.py"]
