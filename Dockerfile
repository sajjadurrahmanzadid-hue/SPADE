# ─────────────────────────────────────────────────────────────────────────────
# SPADE — Statistical Platform for Agronomic Data Evaluation
# Reproducible container image.
#
# Builds the exact environment pinned in environment.yml (Python 3.11 + the
# conda/pip stack) on top of micromamba, then launches the Streamlit app.
# The image is self-contained: no local Python, conda, or R required on the host.
#
# Build:  docker build -t spade:1.0 .
# Run:    docker run --rm -p 8501:8501 spade:1.0
#         then open http://localhost:8501
#
# Pinned to a specific micromamba release for reproducibility. micromamba reads
# environment.yml natively, so the container and the Conda environment are
# guaranteed to resolve from the same specification.
# ─────────────────────────────────────────────────────────────────────────────
FROM mambaorg/micromamba:1.5.8

# --- System libraries -------------------------------------------------------
# Kaleido (static Plotly export) ships a bundled headless Chromium that needs a
# set of shared libraries not present in the slim base image. Without these,
# figure/table PNG export hangs or fails silently. curl is added for the
# container HEALTHCHECK. Installed as root, then we drop back to the unprivileged
# micromamba user.
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 libcups2 \
        libdrm2 libgbm1 libxkbcommon0 libxcomposite1 libxdamage1 \
        libxext6 libxfixes3 libxrandr2 libxshmfence1 libasound2 \
        libpango-1.0-0 libcairo2 libatspi2.0-0 \
    && rm -rf /var/lib/apt/lists/*
USER $MAMBA_USER

# --- Python environment -----------------------------------------------------
# Copy only the spec first so the (slow) dependency-resolution layer is cached
# and not rebuilt when application code changes. Installed into `base` so the
# micromamba entrypoint activates it automatically at runtime.
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml \
    && micromamba clean --all --yes
ARG MAMBA_DOCKERFILE_ACTIVATE=1

# --- Application ------------------------------------------------------------
WORKDIR /app
COPY --chown=$MAMBA_USER:$MAMBA_USER . /app

# Streamlit network/runtime configuration
ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

# Liveness check against Streamlit's internal health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

# Entry point is provided by the micromamba base image (it activates the env and
# execs this CMD). Change the filename here if the app entry script is renamed.
CMD ["streamlit", "run", "wheat_dashboard.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
