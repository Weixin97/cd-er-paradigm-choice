# Reproducibility container for the cross-domain ER paradigm-choice artifact.
# Based on jupyter/scipy-notebook, includes Python 3.10, JupyterLab, numpy,
# pandas, scikit-learn, matplotlib, and other standard scientific packages.

FROM jupyter/scipy-notebook:latest

# System dependencies that some Python packages need at build time
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Install additional Python dependencies from requirements.txt
USER jovyan
COPY --chown=jovyan:users requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Repo will be mounted at /home/jovyan/work by docker-compose
WORKDIR /home/jovyan/work

# Ensure REPO_ROOT is set for config resolution
ENV REPO_ROOT=/home/jovyan/work
ENV PYTHONPATH=/home/jovyan/work
