import os
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional
from llm_datadist import (LLMDataDist, LLMRole)
from vllm.config import KVTransferConfig, VllmConfig
from vllm.distributed.parallel_state import get_tp_group, get_world_group
from vllm.utils import get_ip, logger
from vllm_ascend.utils import get_ascend_soc_version


from vllm_ascend.distributed.llmdatadist_c_mgr_connector import LLMDataDistCMgrConnectorWorker, LLMDataDistCMgrAgentMetadata

from dynamicPD.patching import dynamicPDPatch

class LLMDataDistCMgrConnectorWorkerPatch(dynamicPDPatch[LLMDataDistCMgrConnectorWorker]):
    def __init__(self, vllm_config: VllmConfig):
        assert vllm_config.kv_transfer_config is not None
        logger.info("Initialize the LLMDataDistCMgrConnectorWorker")
        # we assume the local node only contains dp and tp, and tp will not communicate inter-node.
        # for any scenario beyond this scope, the functionality of this connector is not guaranteed.
        self.local_rank_on_node = get_world_group().rank % (
            vllm_config.parallel_config.data_parallel_size_local *
            vllm_config.parallel_config.tensor_parallel_size)
        self.local_rank = get_world_group().local_rank
        if vllm_config.parallel_config.data_parallel_external_lb:
            self.local_dp_rank = vllm_config.parallel_config.data_parallel_rank
        else:
            self.local_dp_rank = vllm_config.parallel_config.data_parallel_rank_local
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.tp_rank = get_tp_group().rank_in_group
        self.rank = get_world_group().rank
        self.local_ip = get_ip()
        self.kv_transfer_config: KVTransferConfig = vllm_config.kv_transfer_config
        self.local_agent_metadata: Optional[
            LLMDataDistCMgrAgentMetadata] = None
        self.vllm_config = vllm_config
        self.executor = ThreadPoolExecutor(8)
        self.thread_lock = threading.Lock()

        self.llm_datadist_role = None
        self.llm_datadist_remote_role = None
        if self.kv_transfer_config.kv_role == "kv_producer":
            self.llm_datadist_role = LLMRole.PROMPT
            self.llm_datadist_remote_role = LLMRole.DECODER
        elif self.kv_transfer_config.kv_role == "kv_consumer":
            self.llm_datadist_role = LLMRole.DECODER
            self.llm_datadist_remote_role = LLMRole.PROMPT
        else:
            raise RuntimeError(
                f"LLMDataDistWorker: Receive unexpected kv role in LLMDataDistWorker, this worker now only support kv_producer and kv_consumer, but receiving {vllm_config.kv_transfer_config.kv_role}"
            )

        # linked_cluster record the cluster that already build the connection its format should be {"cluster_id": "comm_name"}
        self.linked_cluster: dict[Any, Any] = {}
        self.prefill_device_list: list[tuple[int, int]] = []
        self.decode_device_list: list[tuple[int, int]] = []
        global_rank_table = self.read_offline_rank_table()
        self.local_agent_metadata = self.read_agent_metadata(global_rank_table)
        self.llm_datadist = LLMDataDist(self.llm_datadist_role,
                                        self.local_agent_metadata.cluster_id)
        self.init_llm_datadist()
        self.finished_reqs: set[str] = set()
        self.soc_info = get_ascend_soc_version()
        # Set hccl deterministic for model execute
        os.environ["HCCL_DETERMINISTIC"] = "true"
        self.done_receiving_counts: defaultdict[str,
                                                set[int]] = defaultdict(set)
        self.reqs_to_send: dict[str, float] = {}
