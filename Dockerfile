ARG BASE_IMAGE=vllm-ascend:0.18.0
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG WORKSPACE=/vllm-workspace

ENV VLLM_WORKSPACE=${WORKSPACE} \
    VLLM_TARGET_DIR=${WORKSPACE}/vllm \
    VLLM_ASCEND_TARGET_DIR=${WORKSPACE}/vllm-ascend \
    DYNAMICPD_ENABLED=1 \
    PYTHONUNBUFFERED=1

WORKDIR ${WORKSPACE}

COPY vllm ${WORKSPACE}/vllm
COPY vllm-ascend ${WORKSPACE}/vllm-ascend
COPY dynamicPD ${WORKSPACE}/dynamicPD

RUN if ! command -v git >/dev/null 2>&1; then \
      echo "git is required to apply dynamicPD patches" >&2; \
      exit 1; \
    fi

RUN ${WORKSPACE}/dynamicPD/scripts/apply_patch.sh

RUN TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
    VLLM_TARGET_DEVICE=empty \
    python -m pip install --no-cache-dir --no-build-isolation --no-deps -e ${WORKSPACE}/vllm && \
    TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
    python -m pip install --no-cache-dir --no-build-isolation --no-deps -e ${WORKSPACE}/vllm-ascend && \
    TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
    python -m pip install --no-cache-dir --no-build-isolation --no-deps -e ${WORKSPACE}/dynamicPD

WORKDIR ${WORKSPACE}/dynamicPD/scripts

ENTRYPOINT ["vllm"]
CMD ["--help"]
